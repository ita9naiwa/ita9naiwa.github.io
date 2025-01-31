---
layout: article
title: "Implementing FlashAttnetion V1 naively"
category: "mlsys"
tag: "mlsys"
comment: true
key: 20250127
mathjax: true
---

### Warning
This is not a comprehensive tutorial. It's more a note for myself to write what descisions I made while implementing naive FlashAttention V1. So sadly this also describes my limitation of skills.

I already posted [an introductory post about CUDA]{{"_posts/2023/2023-11-16-attention-cuda.md" | absolute_url}} a year ago. I've been not using CUDA actively after writing this post. It would be great if I continue to develop Parallel Computing since then. Anyway now I am again studying Parallel Computing. I wrote two posts about [matmul]({{"_posts/2025/2025-01-27-matmul-basics.md" | absolute_url}}) and [Attention]({{"_posts/2025/2025-01-27-matmul-basics.md" | absolute_url}}).

Now we (at least I) are ready to implement FlashAttention V1.

But, soon I realized that [the original implementation](https://github.com/Dao-AILab/flash-attention) is **very highly** optimized and I can't even understand it. Luckily, there's an easy and comprehensive [implementation in vLLM](https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/prefix_prefill.py) to refer to when I got stuck.


Implementation is [here](https://github.com/ita9naiwa/playground/blob/master/kernels/flash_attn.cu)

### Some decisions

- I used same block size for row.
- I set block_size to be `(BLOCK_SIZE, BLOCK_SIZE)`, where `BLOCK_SIZE=16`


These limitations made following implementation details.


Since there's shuffle reduction. I didn't know that `__shfl` reads `threadIdx.x` first.

```cpp
int tx = threadIdx.x / BLOCK_SIZE;
int ty = threadIdx.x % BLOCK_SIZE;
```

I did this to enable `__shfl` among `ty`s. However, one bad thing is that **by setting `ROW_BLOCK_SIZE=COL_BLOCK_SIZE`**, I wasn't able to increase `COL_BLOCK_SIZE` to be 32, thus using `__shfl_xor_sync` become difficult. Instead, I performed reduction in following way.

```cpp
float S_ij = S_ij_orig;
for (int offset = 8; offset > 0; offset >>= 1) {
  float val = __shfl_down_sync(0xffff, S_ij, offset, 16);
  S_ij = fmaxf(S_ij, val);
}

if (ty == 0) {
  shared_vals[tx][0] = S_ij;
}
__syncthreads();
float max_ij = shared_vals[tx][0];
```
This makes code really messy but seems like performance decrease is not significant.


```cpp
__shared__ scalar_t Q_shared[BLOCK_SIZE][DIM_SPACE];
__shared__ scalar_t K_shared[BLOCK_SIZE][DIM_SPACE];
__shared__ scalar_t V_shared[BLOCK_SIZE][DIM_SPACE];
```

`DIM_SPACE` can be larger than `BLOCK_SIZE`, thus loading Q looks like (K, V similarly)
```cpp
for (int t = ty; t < D; t += BLOCK_SIZE) {
  int idx = blockIdx.x * N * D + i * BLOCK_SIZE * D + tx * D + t;
  Q_shared[tx][t] = Q[idx];
}
```

Also, it only handles `H <= 64`, and `N, H` to be multiples of 16.


Actually, those decisions are not decisions, rather than limitations of the implementation and my skills. I think I can fix, but implementing again will be easier. I spent a few day implementing this, and don't feel like to do right now.


### Simple performance comparison
I very simply measured performance against pytorch reference implementation.

```python
def test_perf():
    B, H, N, D = 128, 32, 64, 64
    Q = torch.randn(B, H, N, D).to(torch.float16).cuda()
    K = torch.randn(B, H, N, D).to(torch.float16).cuda()
    V = torch.randn(B, H, N, D).to(torch.float16).cuda()
    start = time.time()
    for i in range(100):
        torch._scaled_dot_product_attention_math(Q, K, V, scale=1.0)[0]
    print("Time taken for 100 iterations: ", time.time() - start)
    start = time.time()
    for i in range(100):
        flash_attention_v1(Q, K, V)
    print("Time taken for 100 iterations: ", time.time() - start)


test_perf()
```

Luckily my naive implementation won against pytorch one.


```
Time taken for 100 iterations:  0.7478551864624023
Time taken for 100 iterations:  0.6320099830627441
```