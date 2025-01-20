---
layout: article
title: "Flash Attention Idea"
category: "mlsys"
tag: "mlsys"
comment: true
key: 20250118
mathjax: true
---


이 글은 기본적으로 [이 글](https://gist.github.com/Groverkss/7c5eccc6547c8d6c817a263c1d9c7bc9)의 요약이다. 내가 이해가 안 갔던 부분은 조금 내가 찾아서 공부하고 조금 덧붙인 내용이 있지만, 어려운 부분이나 허접한 부분이 있다면 내 탓이다. 훌륭한 튜토리얼을 만든 [Kunwar Grover](https://github.com/Groverkss)에게 감사합니당.


### Attention with score modifier

```
Attention(Q, K, V) = Softmax(score_mod(Q @ K.T)) @ V
```

where `@` is matmul

score modifier `score_mod` exhibits attention variants. Some are common

1. masked(causal) attention
```
score_mod = mask_func(dim0, dim1) ? x : -inf
```
where mask_func = arbitrary binary function to say it's masked or not.

2. Scale

```
score_mod = x / scale
```

### Safe Softmax

Softmax(v)
```python
    s = 0
    for i in range(n):
        y_i = exp(v_i)
        s += y_i
    for i in range(n):
        y_i = y_i / s
    return {y_0, y_1, ..., y_n}
```

인데, `exp(v_i)`, `s`의 값이 짱 커질수도 있어서, 얘를 적당히 normalize해줘야 한다. fp16의 max값은 ~=65504로, log(65504) ~=4.81 정도로, 생각보다 아주 쉽게 overflow가 발생한다. 따라서, 계산적으로 동일하지만 좀 더 stable한 다음 연산을 사용해준다.

safeSoftmax(v)
```python
    m = -inf
    for i in range(n):
        m = max(m, v_i)

    s = 0
    for i in range(n):
        y_i = exp(v_i - m)
        s += y_i
    for i in range(n):
        y_i = y_i / s
    return {y_0, y_1, ..., y_n}
```

`-inf <= v_i - m <= 0` 이므로 항상 exp 값은 0과 1 사이에 바운드되게 된다. 저 safeSoftMax 계산에서 문제점이 있다고 하면, **loop가 3개 필요하다는 점이다.** 그리고 저 loop 중 위의 두개는 parallelize되기 조금 불편한, reduction loop다(값을 accumulate하는 loop).

### Fast Safe Softmax

https://arxiv.org/abs/1805.02867 에서 제안된, 빠른 softmax 구하는 방법이다. first, second loop를 online approach로 걔선했다.

fastSafeSoftmax(v)
```python
    m_0 = -inf
    d_0 = 0
    for i in 1 ... n:
        m_i = max(m_{i-1}, x_j)
        d_j = d_{j-1} * exp(m_{j-1} - m_j) + exp(x_j - m_j)
    for i in 1 ... n:
        y_i = exp(x_i - m_n) / m_n
    return y
```

### Parallel, Reduction

이 용어가 mlir 용어인지, 어떤 용어인지 정확하게 모르는데, 아무튼 대략적인 설명은 다음과 같다
v = [1,2,3,4, ..., n]
가 있다고 할때,
exp(v) 는 병렬 연산이고, blocker가 없다. 4개 스레드가 각각 v_1, v_2, v_3, v_4, ..., v_n에 대한 exp(v_i)를 계산할 수 있다. 이런 연산을 흔히 Parallel이라고 부른다.

sum(v)는 병렬화 하기엔 조금 힘든 연산이다, "축을 줄인다"는 점에서 reduction이라고 부른다. 이 연산을 그래도 병렬화하는 방법이 있는데, `v_1 + v_2 + v_3 + v_4`의 계산 순서를 잘 생각해보자.
```
tmp = v_1
tmp += v_2
tmp += v_3
tmp += v_4
```
얘는 대략 O(n)의 순차적 연산이 필요하다.

```
# 첫번째 병렬 연산
tmp_1 = v_1 + v_2
tmp_2 = v_3 + v_4

# 두번째 병렬 연산
tmp = tmp_1 + tmp_2
```
얘는 대략 O(logn)의 순차적 연산이 필요하다. 물론 프로세서의 스레드 개수에 대한 한계는 존재하겠지만, 연산이 겹치지 않는 부분, 결합법칙과 교환법칙이 성립하는 곳에선 병렬화를 어느 정도 잘 할 수 있다.

저 FastSafeSoftmax 연산의 첫번째 for loop는 1개의 reduction이고, 두번째는 parallel 루프이다. 그래서 이걸 attention 구현에 적용하면 밑과 같아지는데..

```
    Q := M by K matrix
    K := L by K matrix
    V := L by N matrix

    Attention(Q, K, V) = Softmax(Q @ K.T) @ V
```

```python
    S = Q @ K.T // S := M by K2

```