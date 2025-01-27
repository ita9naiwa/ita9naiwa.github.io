---
layout: article
title: "Tiled Matmul 101"
category: "mlsys"
tag: "mlsys"
comment: true
key: 20250127
mathjax: true
---


I'm extremly poor at thinking about matrices. I've seen many people graphically think and draw matrix strides, multiplications, etc....
Yet as a person who work in ML, I think I should understand matmul, even in high-level concepts. This post is about my struggle to understand matmul in HPC environment.

for convenience, I set the notation:
- A is M by K sized matrix
- B is K by N sized matrix
- C = A @ B, is M by N sized matrix


### Loops 1
```python
for i in range(M):
  for j in range(N):
    for k in range(K):
      C[i][j] += A[i][k] * B[k][j]
```
Note that we change the order of for loop. The result is same.

**Loops 2**
```python
for k in range(K):
  for i in range(M):
    for j in range(N):
      C[i][j] += A[i][k] * B[k][j]
```


Or, we can unroll the loop as we want.

**Loops 3**
```python
for i in range(M, 16):
    for i0 in range(16):
      for j in range(N, 16):
        for j0 in range(16):
          for k in range(K, 16):
            for k0 in range(16):
              C[i][j] += A[i][k + k0] * B[k + k0][j]
```

We can just reorder this loops again;

**Loops 4**
```python
for i in range(M, 16):                # Two loops are parallelized
    for j in range(N, 16):            # by grids
      for k in range(K, 16):          # Not parallelized
        for i0 in range(16):          # Two loops are parallelized
          for j0 in range(16):        # by blocks
            for k0 in range(16):      # Not parallelized
              C[i][j] += A[i][k + k0] * B[k + k0][j]
```
This Loops 4 form is what we want, Slightly different image showing same procedure in sequential manner.

- We can parallelize loops not involving k, and k0.
    - first two loops having i, j can be parallelized by Grids
    - i0, j0 loops are parallelized by Blocks
- We can tile `A[i:i + 16][k * 16:(k + 1) * 16]`, and `B[k*16:(k+1) * 16][j:j + 16]` and store them to the shared memory

Visualized figure is given below;

![tiled_matmul]({{ "/assets/images/tiled_matmul/tiled_matmul.png" | absolute_url }})


Cuda implementation is given below;
```cpp
__global__ void matmul_smem(int *A, int *B, int *C, int M, int K, int N) {
    __shared__ int tile_A[16][16];
    __shared__ int tile_B[16][16];

    int i = blockIdx.x * blockDim.x; // Grid parallelizing i (rows)
    int j = blockIdx.y * blockDim.y; // Grid parallelizing j (columns)
    int i0 = threadIdx.x;
    int j0 = threadIdx.y;
    int sum = 0;

    for (int k = 0; k < K; k += 16) { // Tiling over K dimension
        // Load A's tile into shared memory
        if (i + i0 < M && k + j0 < K)
            tile_A[i0][j0] = A[(i + i0) * K + (k + j0)];
        else
            tile_A[i0][j0] = 0;

        // Load B's tile into shared memory
        if (j + j0 < N && k + i0 < K)
            tile_B[i0][j0] = B[(k + i0) * N + (j + j0)];
        else
            tile_B[i0][j0] = 0;
        __syncthreads();

        // Perform computation within the tile
        for (int k0 = 0; k0 < 16; k0++) { // Iterate over tile's K dimension
            sum += tile_A[i0][k0] * tile_B[k0][j0];
        }
        __syncthreads();
    }

    // Write the result to C
    if (i + i0 < M && j + j0 < N) {
        C[(i + i0) * N + (j + j0)] = sum;
    }
}
```


### Conclusion
I've struggled but finally I grasped one of the most important idea in HPC computing...!

### References:
- [Learn CUDA Programming](https://www.amazon.com/CUDA-Cookbook-Effective-parallel-programming/dp/1788996240)
- [CUTLASS: Fast Linear Algebra in CUDA C++](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)
