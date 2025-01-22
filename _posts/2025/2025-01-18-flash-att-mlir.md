---
layout: article
title: "Flash Attention Idea"
category: "mlsys"
tag: "mlsys"
comment: true
key: 20250118
mathjax: true
---

This post is basically a commentary, or more introductory version of this [gist](https://gist.github.com/Groverkss/7c5eccc6547c8d6c817a263c1d9c7bc9). I added some which helped me to understand the article, but more verbosity and confusion may be introduced. Thanks to [Kunwar Grover](https://github.com/Groverkss) for making a great tutorial on FlashAttention and mlir.linalg.

The goal of this post is to provide conceptual understanding of FlashAttention for people who have no or little background of High Performance Computing. (like me).

### Parallel, Reduction Loops
> I've seen these often but never studied deeply, and I couldn't find great references so this is mostly my memory-based. It might be inaccurate.

#### Parallel - Parallelizable loop

```python
# a = log(a)
a = [0, 1, 2, ..., 99]
for i in range(100):
    a[i] = log(a[i])
```

Inside the loop, each write doesn't conflict (no race condition). If we have 100 threads, this for loop is completely parallelizable. This kind of loop is called `Parallel`.

#### Reduction - Reducing dimension

```python
# s = sum(a)
a = [0, 1, 2, ..., 99]
s = 0
for i in range(100):
    s += a[i]
```

This kind of `Reduction` is somewhat parallelizable, but with some constraints.

- Parallalization approach 1
Place a mutex on  `s` allowing only one thread to write to `s` at a time. If computation per thread is large, it would be useful but for a simple sum it can be inefficient.

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

... until it reduces to a single summation.
```

Operations in each chunk can run in parallel. The total number of operations is still `O(n)`.
However, operations in each chunk don't overlap and can be parallelized and the number of chunk call `O(logn)`.
In short, Reduction also can be parallelized but with some constraints.
For example, Operations like `Sum`, `Min`, `Max` are `Reduction`.

#### Note
>  I use sequential notation for reduction loops to illustrate logic. Actual implementations often parallelize them.

### Attention(Q, K, V) = Softmax(score_mod(Q @ K.T)) @ V

Here, `@` denotes matrix multiplication.

```
    Attention(Q, K, V) = Softmax(score_mod(Q @ K.T)) @ V
```

The function `score_mod` includes different forms of attention:

1. masked(causal) attention
```
    score_mod = mask_func(dim0, dim1) ? x : -inf
```
where mask_func = arbitrary binary function to say it's masked or not.

2. Scale

```
    score_mod = x / scale
```

multiple score modification can be applied. For example, it's common to apply both Scale and causal mask. So all you need to remember is that Attention procedure is calculated by following order:

1. matmul
1. (Optional) score modifiers
1. softmax
1. matmul

For simplicity, we omit `score_mod` in examples below. It's relatively to apply score modification since they're elementwise function and can run in parallel easily.

### matmul

Matrix Multiplication, as you know, is presented in `3-nested loops`

`C = A @ B`
- `A: M by K` matrix
- `B: K by N` matrix
- `C: M by N` matrix

```python
    for i in 1 ... M: # Parallel Loop
        for j in 1 ... N: # Parallel Loop
            C[i][j] = 0
            for k in 1 ... K: # Reduction Loop
                C[i][j] += A[i][k] * B[k][j]
```

or, it can be written `sum of outer products`.

```python
    C[:][:] = 0
    for i in 1 ... M: # Reduction Loop
        C += outerProduct(A[:][i], B[i][:])
```

To me, at first glimpse, this `sum-of-outer-products` view looks strange, but it offers tiling and cache friendly optimizations. For example, we can split C into 4 subblocks and perform matmul in smaller tiles (more cache hit against C and A, B). Further decomposing `outerProduct` is somewhat implementation detail and also related to the outer reduction loop. In the rest of this post, we use `sum-of-outer-products` view of matrix multiplication. How inner `outerProduct` is decomposed is omitted, otherwise mentioned.

### softMax

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

While straightforward, this can overflow if exp(v[i]) is large.


#### safeSoftmax

`exp(v)` can be really large. `Max<FP16>() ~= 65504` and `log(65504) ~= 4.81`, so overflow can occur very easily. One solution is to bound `exp` function by `exp(v - max(v))`.

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

Since `-inf <= v_i - m <= 0`, `exp` value is always bounded between 0 and 1. One problem of SafeSoftmax is that there are three loops and two of these loops are Reduction loops, which is usually slower than parallel loop.

### fastSafeSoftmax
Proposed in https://arxiv.org/abs/1805.02867, FastSafeSoftmax merges two loop blocks:

`fastSafeSoftmax(v)`
```python
    m = -inf # n sized array
    d = 0    # n sized array
    for i in 1 ... n: # Reduction Loop
        # Each chunk calculates:
        m[i] = max(m[i - 1], x[i])
        d[i] = d[i - 1] * exp(m[i - 1] - m[i]) + exp(x[i] - m[i])
        # m[n - 1], d[n - 1] are reduced finally

    for i in 1 ... n: # Parallel Loop
        y[i] = exp(x[i] - m[n - 1]) / d[n - 1]
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
    P = Q @ K.T # M by K_2 matrix
        # - unnormalized weights over K_2 candidates
    S = M by K_2 matrix
    for i in 1 ... m: # Parallel loop
        S[i][:] = softmax(P[i][:])
    O = S @ V
    return O
```

Let's decompose Softmax Operation, as mentioned above.

```python
    # loop block 1
    P = Q @ K.T # M by K_2 matrix
        # - unnormalized weights over K_2 candidates of M queries

    S = 0 # M by K_2 matrix initialized by zeros
          # - normalized weights over K_2 candidates of M queries

    # loop block 2
    for i in 1 ... M: # Parallel loop
        m = -inf # size of M by K_2 array
        d = 0    # size of M by K_2 array
        for j in 1 ... K_2: # Reduction Loop
            # Each chunk calculates:
            m[i][j] = max(m[i][j - 1], P[i][j])
            d[i][j] = d[i][j - 1] * exp(m[i][j-1] - m[i][j]) + exp(P[i][j] - m[i][j])
        # m[i][K_2 - 1], d[i][K_2 - 1] are final value generated by this loop.

    # loop block 3
    for i in 1 ... M: # Parallel Loop
        for j in 1 ... K_2: # Parallel Loop
            S[i][j] = exp(P[i][j] - m[i][K_2 - 1]) / d[i][K_2 - 1]

    # loop block 4
    O = S @ V
    return O
```

### FlashAttention

FlashAttention v1 merges loop blocks 2, 3, and 4. The idea is very similar to merging two reduction loops in `fastSafeSoftmax`. Concretely:


```python
    # loop block 1
    P = Q @ K.T # M by K_2 matrix
        # - unnormalized weights over K_2 candidates of M queries

    S = 0 # M by K_2 matrix initialized by zeros
          # - normalized weights over K_2 candidates of M queries

    # loop block 2
    for i in 1 ... M: # Parallel loop
        m = -inf # size of M by K_2 array
        d = 0    # size of M by K_2 array
        for j in 1 ... K_2: # Reduction Loop
            # Each chunk calculates:
            m[i][j] = max(m[i][j - 1], P[i][j])
            d[i][j] = d[i][j - 1] * exp(m[i][j-1] - m[i][j]) + exp(P[i][j] - m[i][j])
        # m[i][K_2 - 1], d[i][K_2 - 1] are final value generated by this loop.

    # loop block 3
    for i in 1 ... M: # Parallel Loop
        for j in 1 ... K_2: # Parallel Loop
            S[i][j] = exp(P[i][j] - m[i][K_2 - 1]) / d[i][K_2 - 1]

    # loop block 4
    ## <- Before
    O = S @ V
```

Let's see the loop block 2 first. The first parallel loop `for i in 1 ... M:` can be omitted, if we think as if we're using SIMD/SIMT operations. This is what we were already doing when viewing `outerProduct`.

#### <- before
```python
# loop block 2
for i in 1 ... M: # Parallel loop
    m = -inf # size of M by K_2 array
    d = 0    # size of M by K_2 array
    for j in 1 ... K_2: # Reduction Loop
        # Each chunk calculates:
        m[i][j] = max(m[i][j - 1], P[i][j])
        d[i][j] = d[i][j - 1] * exp(m[i][j-1] - m[i][j]) + exp(P[i][j] - m[i][j])
    # m[i][K_2 - 1], d[i][K_2 - 1] are final value generated by this loop.
```

#### <- After

```python
# loop block 2
    m = -inf # size of M by K_2 array
    d = 0    # size of M by K_2 array
    for j in 1 ... K_2: # Reduction Loop
        # Each chunk calculates:
        m[:][j] = max(m[:][j - 1], P[:][j])
        d[:][j] = d[:][j - 1] * exp(m[:][j-1] - m[:][j]) + exp(P[:][j] - m[:][j])
    # m[:][K_2 - 1], d[i][K_2 - 1] are final value generated by this loop.
```

Then we go to the loop block 3 and 4. Actually What loop block 3 does is `S = exp(P - m) / d` and loop block 4 does is `S @ V`. We mathematically view this, and represent in a single loop block.
```
    S @ V =
    (exp(P - m) / d) @ V
    (exp(P - m) @ V)  / d
```

Then, $exp(P - m) @ V / d$ can be represented in

```python
    # 3-4 merged block
    ret = 0 # M by N Matrix
    for i in 1 ... K_2: # Reduction Loop
        ret += outerProduct(exp(P - m)[:][i], V[i][:])
    return ret / d
```

The core idea of fusing 3-4 merged block into the loop block 2 is that

> While we are constructing `m`, `d`, we can use intermediate values of `m`, `d` and calculate partial outcome and run Reduction. So, our final Attention procedure is that:

```python
    # loop block 1
    P = Q @ K.T # M by K_2 matrix
                # - unnormalized weights over K_2 candidates of M queries
    # 2-3-4 merged block
    m = -inf # size of M by K_2 array
    d = 0    # size of M by K_2 array
    O = 0    # M by N Matrix
    for j in 1 ... K_2: # Reduction Loop
        # Each chunk calculates:
        m[:][j] = max(m[:][j - 1], P[:][j])
        d[:][j] = d[:][j - 1] * exp(m[:][j-1] - m[:][j]) + exp(P[:][j] - m[:][j])
        O = O * exp(m[:][j - 1] - m[:][j]) + outerProduct(exp(P[:][j] - m[:][j]), V[j][:])
    return O / d[:][K_2 - 1]
```

In practice, FlashAttention kernels break this loop into tile-sized chunks (in the sequence dimension) and also tile the outerProduct for GPU efficiency. The key insight is doing all computations in a single pass, reducing intermediate reads/writes and improving numerical stability by keeping values in higher-precision registers.

### Conclusion

This is a conceptual explanation of how FlashAttention fuses softmax and matrix multiplication into one loop. Real implementations split or tile these loops for hardware efficiency.

