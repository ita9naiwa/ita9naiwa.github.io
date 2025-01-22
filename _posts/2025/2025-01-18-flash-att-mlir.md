---
layout: article
title: "Flash Attention Idea"
category: "mlsys"
tag: "mlsys"
comment: true
key: 20250118
mathjax: true
---

This post is basically a commentary, or more introductory version of this [gist](https://gist.github.com/Groverkss/7c5eccc6547c8d6c817a263c1d9c7bc9). I added some which helped me to understand the article, but more verbosity and confusion may be introduced. Thanks to [Kunwar Grover](https://github.com/Groverkss) for making great tutorial to FlashAttention and mlir.linalg.

The goal of this post is to give conceptual understanding on FlashAttention to people who have no knowledge of High Performance Computing.

### Parallel, Reduction Loops
> I've seen these often but never studied well, and I was unable to find great references so mostly my memory-based. Thus it might be inaccurate.

#### Parallel - Parallalizable loop

```python
a = [0, 1, 2, ..., 99]
# a = log(a)
for i in range(100):
    a[i] = log(a[i])
```
Inside the loop, each write doesn't overwrap. No race condition. If we have 100 threads, this for loop is completely parallalizable. This kind of loop is called `Parallel`.

#### Reduction - Reduce dimension.

```python
# s = sum(a)
a = [0, 1, 2, ..., 99]
s = 0
for i in range(100):
    s += a[i]
```
This kind of `Reduction` is somewhat parallelizable, but with some constraints.

- Parallalization approach 1
mutex on variable `s` and make only one thread can write to `s`. If computation is complicated and each thread's calculation differ (thus less race condition), it would be useful.

- Parallelization approach 2
It employs some property of reduction operations's Associativitiy.
`a + b + c d = ((a + b) + (c + d))`

- Realization 1
```
s = 0
s += a[0]
s += a[1]
...
s += a[99]
```
It requires `O(n)` sequential addition.

- Realization 2
```
// chunk 1
tmp_1, ..., tmp_50 = 0,0, ..., 0
tmp_1 = a[0] + a[1]
tmp_2 = a[2] + a[3]
...
tmp_50 = a[98] + a[99]


// chunk 2
new_tmp_1, new_tmp_2 = ..., new_tmp_25 = 0, 0, ..., 0
new_tmp_1 = tmp_1 + tmp_2
new_tmp_2 = tmp_3 + tmp_4
...
new_tmp_25 = tmp_49 + tmp_50
```
Each chunk calculation can be parallelized. The total number of operations is still `O(n)`, but operations in each chunk can be parallelized and the number of chunk call `O(logn)`.

In summary, Reduction also can be parallelized but with some constraints.
For example, Operations like `Sum`, `Min`, `Max` are `Reduction`.

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

So all you need to remember is that Attention procedure is calculated by following order:

1. matmul
2. softmax
3. matmul


### softMax

Let's focust first on softmax.

#### safeSoftmax

`softmax(v)`
```python
    n = len(v)
    s = 0
    y = [0 for _ in range(n)]
    for i in 1 ... n: # Reduction Loop
        y_i = exp(v_i)
        s += y_i
    for i in 1 ... n: # Parallel Loop
        y[i] = y[i] / s
    return y
```

`exp(v)` might be really large. `Max<FP16>() ~= 65504` and `log(65504) ~= 4.81`, so overflow can occur very easily. One solution is to bound `exp` function by `exp(v - max(v))`.

`safeSoftmax(v)`
```python
    m = -inf
    for i in 1 ... n: # Reduction Loop
        m = max(m, v[i])

    s = 0
    for i in 1 ... n: # Reduction Loop
        y[i] = exp(v[i] - m)
        s += y[i]

    for i in 1 ... n: # Parallel Loop
        y[i] = y[i] / s

    return y
```

Since `-inf <= v_i - m <= 0`, `exp` value is always bounded between 0 and 1. One problem of SafeSoftmax is that there are three loops and two of these loops are Reduction loops, which is slower than parallel loop.

### fastSafeSoftmax
Proposed in https://arxiv.org/abs/1805.02867, FastSafeSoftmax replaces first two reduction loops to single reduction loop, making significant performance improvement.

`fastSafeSoftmax(v)`
```python
    m_0 = -inf
    d_0 = 0
    for i in 1 ... n: # Reduction Loop
        # Each chunk calculates:
        m_i = max(m_{i-1}, x_j)
        d_j = d_{j-1} * exp(m_{j-1} - m_j) + exp(x_j - m_j)
        # m, d are reduced finally

    for i in 1 ... n: # Parallel Loop
        y_i = exp(x_i - m) / d
    return y
```

### Attention
Let's go back to the Attention Operation.
I just assume here that readers know basic semantics of Attention.
`O = attention(Q, K, V) = softmax(Q @ K.T) @ V`

- `Q: M by K_1` matrix - `M` query vectors of size `K_1`
- `K: K_2 by K_1` matrix - `K_2` key candidates of size `K_1`
- `V: K_2 by N` matrix - `K_2` val candidates  of size `N`
- `O: M by N` matrix - `M` response vectors of size `N`

```python
    P = Q @ K.T # M by K_2 matrix - unnormalized weights over K_2 candidates
    S = M by K_2 matrix
    for i in range(m): # Parallel loop
        S[i][:] = softmax(P[i][:])
    O = S @ V
    return O
```

Let's decompose Softmax Operation, as mentioned above.

```python
    P = Q @ K.T # M by K_2 matrix - unnormalized weights over K_2 candidates
    S = 0 # M by K_2 matrix initialized by zeros
    for i in range(m): # Parallel loop
        m = -inf # size of M by K_2 array
        d = 0    # size of M by K_2 array
        for j in 1 ... n: # Reduction Loop
            # Each chunk calculates:
            m[i][j] = max(m[i][j - 1], P[i][j])
            d[i][j] = d[i][j - 1] * exp(m[i][j-1] - m[i][j]) + exp(P[i][j] - m[i][j])
            # m[i], d[i] are reduced as max of each row.
        for j in 1 ... n: # Parallel Loop
            S[i][j] = exp(P[i] - m[i]) / d[i]

    O = S @ V
    return O
```