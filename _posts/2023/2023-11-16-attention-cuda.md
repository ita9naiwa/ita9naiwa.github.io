---
layout: article
title: "Attention을 쿠다로 구현해보기"
category: "ML"
tag: "ML"
comment: false
key: 20231110
mathjax: true
---

### CUDA를 공부하기.
쿠다를 공부하려고 오래 전 부터 마음먹었는데 실제로는 한 달 쯤 전부터 (정말 필요해진 순간에!) 공부하게 되었다. 회사에서 LLM 인퍼런스 모델을 이해해야 할 일이 생겨서, 동기가 강제로 부여되게 되어 겨우 쿠다 첫 걸음을 떼었다. 사실 아직 뗐다고 말 할 수 있는지도 잘 모르겠다.

![sigmoid](https://kjhov195.github.io/post_img/200107/image2.png)

사진 출처:https://kjhov195.github.io/2020-01-07-activation_function_1
인터넷에 많이 있는 쿠다 자료의 특징은, 난이도가 Sigmoid 함수를 따른다는 점이다. 예를 들면 다음과 같다. 첫 예제에서 벡터 곱셈을 배우고, 두번째 예제에서 쿠다로 Quick Sort를 구현하고, 세 번째 예제에서는 CNN을 (심지어 backprop도) 구현하는 느낌의 난이도 배치가 되어 있다. 정말 학습자의 의욕을 정말 깎아먹지 않을 수가 없는 배치이다. 그래서 든 생각이, 내가 아는 쿠다 지식으로 뭘 만들어보면 그나마 조금이라도 익숙해지지 않을까 싶었다.

### Attention
(이 글을 읽는 사람은 셀프-어텐션을 이해하고 있다고 가정한다)
난이도도 적절하고, LLM inference에서 실질적으로 많이 optimize되고 재구현되고 있는 모듈이다. 따라서 이를 구현해보기로 결심했다.

실제 구현체는 여기 있다. https://github.com/ita9naiwa/attention-impl

#### Warp Reduce Sum and Block Reduce Sum
출처: https://github.com/vllm-project/vllm/blob/main/csrc/reduction_utils.cuh
```cpp
template<typename T>
__inline__ __device__ T warpReduceSum(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(0xffffffff, val, mask, 32);
  return val;
}
```
쿠다에서 공유 메모리를 제외하고 다른 스레드의 값을 가져오는 유일한 방법은 __shfl_* 함수를 사용하는 것이다. __shfl은 warp 내에서 다른 스레드의 값을 가져오는 연산이다. xor은 다음과 같은 신기한 성질이 있어 이를 이용해 각 warp(32개 스레드의 그룹)의 합을 다음과 같이 계산할 수 있다.
![그래프]({{ "/assets/images/att/butterfly_reduction.png" | absolute_url }})
출처:https://people.maths.ox.ac.uk/~gilesm/cuda/lecs/lec4.pdf

그럼, 이 워프 계산을 마치면 0,..., 31까지는 같은 값이 있다(0-31까지의 합), 32,...,63까지는 또 같은 값이 있다.
그럼 다시, 이 32개의 값을 다시 한번 더 더해주면, 0-63까지의 합을 구할 수 있다. 이를 반복하면, 0-1023까지의 합을 구할 수 있다. 이를 이용해 다음과 같은 코드를 구현할 수 있다. 0, 32, 64, ...,에 있는 값을 `shared[]`에 모아주고, 얘를 다시 warp reduce sum한다는 내용의 코드이고 이로써 0-1023까지 모든 값을 더할 수 있다.
```cpp
template<typename T>
__inline__ __device__ T blockReduceSum(T val) {
    static __shared__ T shared[32];
    if(threadIdx.x == 0) {
        for(int i = 0; i < 32;++i){
            shared[i] = 0.0;
        }
    }
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;
    val = warpReduceSum<T>(val);

    if (lane == 0)
        shared[wid] = val;
    __syncthreads();
    T ret = 0;
    for(int j = 0; j < 32;++j){
        ret += shared[j];
    }
  return ret;
}
```

참고 자료:
- https://people.maths.ox.ac.uk/~gilesm/cuda/lecs/lec4.pdf
- https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/


#### Naive Attention
naive attention은 파이썬으로 구현하기 어렵지 않다.
```python
def MHA(Q, K, V, mask=None):
    _Q = Q.reshape(batch_size, context_size, num_heads, dim // num_heads).permute(0, 2, 1, 3).reshape(batch_size * num_heads, context_size, dim // num_heads)
    _K = K.reshape(batch_size, context_size, num_heads, dim // num_heads).permute(0, 2, 1, 3).reshape(batch_size * num_heads, context_size, dim // num_heads)
    scale = (dim / num_heads) ** -0.5
    S = torch.bmm(_Q, _K.permute(0, 2, 1)).reshape(batch_size, num_heads, context_size, context_size)
    S *= scale
    if mask is not None:
        S += (mask.unsqueeze(1) - 1) * 100000.0
    P = S.softmax(dim=-1)
    O = torch.matmul(P, V.reshape(batch_size, context_size, num_heads, dim // num_heads).permute(0, 2, 1, 3))
    O = O.permute(0, 2, 1, 3).reshape(batch_size, context_size, dim)
    return S, P, O
```

얘는 대강 이런 입력을 받는다
Q, K, V는 각각 (batch_size, context_size, dim) 차원을 갖는 실수 텐서이며, mask는 (batch_size, context_size, context) 차원을 갖는 정수 텐서이다.
이러한 입력을 받는 c++ 코드는 이렇게 된다. 코드마다 주석을 달아 두었으니 이해하기 어렵지 않을 것이다.

```cpp
std::vector<torch::Tensor> naive_attention_forward(
    torch::Tensor &Q,       // [batch_size, context_len, dim]
    torch::Tensor &K,       // [batch_size, context_len, dim]
    torch::Tensor &V,       // [batch_size, context_len, dim]
    torch::Tensor &mask,    // [batch_size, context_len, context_len]
    int num_heads
) {
    auto batch_size = Q.size(0);
    auto context_len = Q.size(1);
    auto dim = Q.size(2);

    // 텐서를 쿠다에 바로 이니셜라이즈하기 위해, 옵션을 미리 만들어 둔다.
    auto options = torch::TensorOptions().dtype(Q.scalar_type()).device(torch::kCUDA);
    auto S = torch::zeros({batch_size, num_heads, context_len, context_len}, options);
    auto P = torch::zeros({batch_size, num_heads, context_len, context_len}, options);
    auto O = torch::zeros_like(Q);
    // context_size가 1024보다 클 수도 있지만, 쿠다의 max threads 크기는 1024
    const int threads = std::min((int)context_len, 1024);

    // 각 배치와 각 헤드마다 같은 연산이 반복되므로, 얘내를 블록으로 보아 계산한다
    const dim3 blocks(batch_size, num_heads);

    // 토치 코드를 실행하기 위해 필요한 스트림을 가져온다.
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // 이건 그냥 MHA에 필요한 거다
    float scale = 1.0 / std::sqrt(float(dim) / num_heads);

    // 파이토치의 AT_DISPATCH_FLOATING_TYPES_AND_HALF 매크로는 바인딩해준다. 템플릿 함수를 16bit float, 32bit float에 대해 realize해준다.
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        Q.scalar_type(),
        "naive_attention_forward_kernel",
        ([&] {
            naive_attention_forward_kernel<<<blocks, threads, 0, stream>>>(
                context_len,
                dim / num_heads,
                scale,
                Q.data_ptr<scalar_t>(), // data의 pointer를 리턴한다. 이러면 쿠다에서 바로 사용할 수 있는 디바이스 메모리를 리턴한다.
                K.data_ptr<scalar_t>(),
                V.data_ptr<scalar_t>(),
                mask.data_ptr<scalar_t>(),
                S.data_ptr<scalar_t>(),
                P.data_ptr<scalar_t>(),
                O.data_ptr<scalar_t>()
            );
        })
    );
    return {S, P, O};
}
```

naive_attention_forward는 다음과 같이 이루어져 있다.

```cpp
template <typename scalar_t>
__global__ void naive_attention_forward_kernel(
    const int context_len,
    const int dim,
    const float scale,
    scalar_t* __restrict__ Q, // __restrict__ 키워드는 포인터가 가리키는 메모리가 다른 포인터에 의해 접근되지 않는다는 것을 컴파일러에게 알려준다.
    scalar_t* __restrict__ K, // 캐싱할 유리해져서, 속도 향상에 도움을 줄 수 있다.
    scalar_t* __restrict__ V,
    scalar_t* __restrict__ mask,
    scalar_t* __restrict__ S,
    scalar_t* __restrict__ P,
    scalar_t* __restrict__ O
) {
    const int thread_id     = threadIdx.x;
    const int block_dim     = blockDim.x;
    const int batch_id      = blockIdx.x;
    const int head_id       = blockIdx.y;
    const int batch_size    = gridDim.x;
    const int num_heads     = gridDim.y;

    // Q has shape      [batch_size, context_len, num_heads, dim]
    // K has shape      [batch_size, context_len, num_heads, dim]
    // mask has shape   [batch_size, context_len, context_len]
    for(int i = thread_id; i < context_len; i += block_dim) {
        for(int j = 0; j < context_len; ++j) {
            // 여기 indexing이 되게 지리멸렬한데, 익숙해지면 (쓰는 사람은) 나름 불편하지 않게 사용할 수 있다.
            int S_idx = (num_heads * context_len * context_len) * batch_id + \
                        (context_len * context_len) * head_id + \
                        (context_len) * i + j;
            // S[S_idx]는 사실 S[batch_id][head_id][i][j]를 의미한다.

            if (mask[(context_len * context_len) * batch_id + context_len * i + j] > 0){
                for(int k = 0; k < dim; ++k) {
                    // 여기 Q_idx, K_idx도 마찬가지다.
                    int Q_idx = (context_len * num_heads * dim) * batch_id + \
                                (dim) * head_id + \
                                (num_heads * dim) * i + k;
                    // Q[batch_id][i][head_id][k]를 의미한다.
                    int K_idx = (context_len * num_heads * dim) * batch_id + \
                                (dim) * head_id + \
                                (num_heads * dim) * j + k;
                    // K[batch_id][j][head_id][k]를 의미한다.
                    S[S_idx] += (scalar_t)((Q[Q_idx] * K[K_idx]) * scale);
                }
            } else {
                S[S_idx] = -100000.0;
            }
        }
    }
    ...
    ...
}
```

쿠다 코드에서 3차원 이상의 텐서는 이렇게 인덱싱해 곱하고 더해주고 할 수 있다. 이것만 기억하면 (느리겠지만) 아무쪼록 쿠다 코드를 짤 수 있다. 위 코드에서는 S (Q*K)를 구하는 부분만 보여주었지만, 나머지는 사실 저 부분과 크게 차이가 나지 않는 곱셈의 연속이다.


#### 실제로 구현한 것들
위의 가장 쉬운 naive attention 이외에도, 세 개를 더 구현해뒀다.
- KV-cache Attention
- Continuous Batching Attention
- Paged KV cache Attention

위 코드를 이해했다면, 차근차근 본다면 그리 어렵지 않을 것이라 생각한다.


### 내가 본 자료들
1. [Learn CUDA Programming](https://github.com/PacktPublishing/Learn-CUDA-Programming)
그나마 난이도가 현실적이고 순차적이고 계단적이게 되어 있다!

2. [CUSTOM C++ AND CUDA EXTENSIONS](https://pytorch.org/tutorials/advanced/cpp_extension.html)
여기 cuda 구현체가 설명이 상당히 자세해서 이해하기 쉽다. 대부분 머신러닝을 하는 사람은 pyTorch같은 곳에 응용하길 원할 거 같은데 이 자료는 그것도 설명을 아주 자세히 해 두었다!

3. [vLLM](https://blog.vllm.ai/2023/06/20/vllm.html)
pagedAttention 모듈은 어려운데, 이 외 부분은 이해하기 쉽게 작성되어 있다. 초보를 갓 벗어난(아직초보일지도모름) 사람에게 딱 적절한 난이도의 실제 코드를 볼 수 있다.

4. [대학교 강의자료](https://people.maths.ox.ac.uk/~gilesm/cuda/lecs/lec4.pdf)
전반적으로 어려운데, 그 때 그 때 내가 참고할 수 있는 부분이 있어서 그 부분만 따로 읽어 보았다. reduction 파트