---
layout: article
title: "Adding Scaled Dot Product to Triton "
category: "mlsys"
tag: "mlsys"
comment: true
key: 20250829
mathjax: true
---

Recently, my first contribution to Triton ([PR #7918](https://github.com/triton-lang/triton/pull/7918)) was merged.
This features a new matrix multiply operation introduced in sm_120 NVIDIA architectures (RTX 5000, A6000 series).
Thanks to [@masahi](https://github.com/masahi), and [@mobiacham](https://github.com/mobicham), I was able to make my first contribution to triton.

With this addition, fast quantization and accelerated inference using Scaled Dot Product is now available even on consumer GPUs (see projects like [gemlite](https://github.com/mobiusml/gemlite))

---

### What is the Scaled Dot Product?
With Ampere/Hopper and beyond, GPUs began supporting FP8 and FP4 datatypes. Along with this, NVIDIA introduced a new type of matrix multiply: **Scaled Dot Product (mma with block scaling)**.
Please refer to [Official PTX docs](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions) for the detail.
Mathematically, it looks like this:
```
D = (A * SF_A) * (B * SF_B) + C
```

Here **SF_A, SF_B** are scale factors applied per row (A) and per column (B). This normalization allows low-precision values to be multiplied with better stability.

The PTX ISA defines scaling granularity modes:
- **scale_vec::1X** → A: M×1, B: 1×N (per-row / per-col)
- **scale_vec::2X** → A: M×2, B: 2×N (split K in half)

In other words, for larger K dimensions, scaling becomes more fine-grained.

---

### Definition in PTX ISA
PTX ISA 9.0 introduces instructions such as:
```
mma.sync.aligned.m16n8k32.mxf8f6f4.block_scale.scale_vec::1X ...
```

Key aspects:
- `.block_scale`: indicates the presence of scale factors as separate fragments
- `.scale_vec::1X/2X`: determines scaling granularity

Within a warp, specific lanes supply scale factors. For example, in an `m16n8k32` shape:
- A has 16 row scales
- B has 8 column scales
For scale_vec::2X, the K dimension is further split.

---

### Mapping in Triton (IR → PTX)
In Triton, I introduced a new `tt.dot_scaled` operation. It works like `tt.dot`, but also takes scale tensors for A and B.

The lowering pipeline:
1. **TTG IR layer**
   - Choose layouts for scale tensors (`chooseScaledNvidiaScaleLayout`)
   - Ensure they match the PTX warp-fragment layout
2. **NVVM lowering**
   - `ttg.dot_scaled` lowered into NVVM intrinsics
3. **PTX backend**
   - Generates final PTX instruction with `.block_scale.scale_vec` suffix

The tricky part: **scale tensor layout mismatch**.
- A/B operands already follow fixed warp-fragment layouts
- Scale tensors must exactly match the expected provider lanes

To solve this, I implemented a dedicated layout-selection logic.

---

### Changes
- Added `tt.dot_scaled` operation to Triton
- Implemented automatic scale tensor layout selection (`chooseScaledNvidiaScaleLayout`)
- Extended PTX backend to emit MMA v5 instructions with block scaling
- Added unit tests and PTX validation

Example MLIR snippet:
```mlir
%res = tt.dot_scaled %a, %b, %scale_a, %scale_b
       : (tensor<16x32xi8>, tensor<32x8xi8>,
          tensor<16xi8>, tensor<8xi8>) -> tensor<16x8xf32>
```

Generated PTX:
```ptx
mma.sync.aligned.m16n8k32.mxf8f6f4.block_scale.scale_vec::1X
  {f32, f32, f32, f32}, {b32, b32}, {b32, b32}, {f32, f32, f32, f32},
  {b32, b32}, {b32, b32};
```

---

### Benchmark
E2E vLLM Benchmark: Llama3-8B-Instruct - in_len=1024 out_len=1024 batch_size=128 (5090 RTX)

(Thanks to @mobicham, he conducted this benchmark)

```
fp8 x fp8 = 42.83 sec
mxfp8 x mxfp8 (using native dot) = 44.45 sec
mxfp8 x mxfp8 (main using emulation) = 76.44 sec
```

### Next Steps

I plan to continue bridging the gap between Triton IR and the latest PTX capabilities.

- Handling multi-CTA cases
- Supporting more scale modes (e.g., X2, X4)
- Adding FP4 and additional precision modes

---

Reference:

- CUDA official docs: https://docs.nvidia.com/cuda/parallel-thread-execution
- PTX official docs: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions