---
layout: article
title: "Linear Layout in Triton (2)"
category: "mlsys"
tag: "mlsys"
comment: true
key: 20251026
mathjax: true
---
### Introduction
We continue our discussion here, and I assume you know some (1) triton frontend, and (2) CUDA. I used mostly NVIDIA GPU terms but same logic can apply to AMD GPUs.

## 4. Dimension Naming Conventions and Semantics

### 4.1 Input Dimensions

Input dimensions represent the **hardware hierarchy**.

#### Register Distributed Layout
```
"register" → Index of register elements held by a single thread
"lane"     → Thread ID within a warp (0~31 for NVIDIA, 0~63 for AMD)
"warp"     → Warp ID within a CTA
"block"    → Block (CTA) ID within a cluster
```

**Example: BlockedEncodingAttr Conversion**
```cpp
// sizePerThread=[4,2], threadsPerWarp=[8,4], warpsPerCTA=[2,2]
auto layout = BlockedEncodingAttr::get(...);
auto ll = layout.toLinearLayout(shape);

// ll has the following input dimensions:
// - register: 0~7 (4×2-1)
// - lane: 0~31 (8×4-1)
// - warp: 0~3 (2×2-1)
// - block: ...
```

#### Shared Memory Layout
```
"offset" → Linear offset within shared memory (in elements)
"block"  → Block ID (for multi-CTA cases)
```

**Example: SwizzledSharedEncodingAttr**
```cpp
auto shared = SwizzledSharedEncodingAttr::get(
  ctx, vec=8, perPhase=4, maxPhase=8, order={1,0}, ...);
auto ll = swizzledSharedToLinearLayout(shape, shared);

// ll inputs: offset, block
// ll outputs: dim0, dim1
// offset → (dim0, dim1) mapping expresses the swizzle pattern
```

### 4.2 Output Dimensions

Output dimensions represent the **logical tensor axes**.

```
"dim0"  → First dimension of the tensor (based on original shape, regardless of order)
"dim1"  → Second dimension of the tensor
"dim2"  → Third dimension of the tensor
...
```

**Important:** Output dimensions are independent of the layout's `order` field. The `order` determines memory layout order, but LinearLayout's output dimension names always use standard order (`dim0, dim1, ...`).

### 4.3 Dimension Ordering and Reshape

Dimensions follow a **minor-to-major order**:
- The first dimension is most minor (changes fastest in memory)
- The last dimension is most major

**Important for reshaping:**
```cpp
// Input dimensions: ["register", "lane"] (in this order)
// When flattened: register is minor, lane is major
auto flattened = layout.flattenIns();
// Result: single dimension, size = register_size × lane_size
```

---

## 5. Core API Detailed Guide

### 5.1 Basic Constructors

#### 5.1.1 `identity1D` - Identity Function

```cpp
static LinearLayout identity1D(
  int32_t size,        // Input size (power of 2)
  StringAttr inDim,    // Input dimension name
  StringAttr outDim    // Output dimension name
);
```

**Meaning:** For range `x ∈ [0, size)`, `L(x) = x`

**Example:**
```cpp
auto L = LinearLayout::identity1D(8, S("lane"), S("dim0"));
// L(0) = 0, L(1) = 1, ..., L(7) = 7
// Bases: [L(1)=1, L(2)=2, L(4)=4]
```

**When to use:**
- Continuous index mapping
- Basic building block for other layouts

#### 5.1.2 `strided1D` - Strided Layout

```cpp
static LinearLayout strided1D(
  int32_t size,
  int32_t stride,
  StringAttr inDim,
  StringAttr outDim
);
```

**Meaning:** For range `x ∈ [0, size)`, `L(x) = stride × x`

**Example:**
```cpp
auto L = LinearLayout::strided1D(4, 2, S("lane"), S("dim0"));
// L(0) = 0, L(1) = 2, L(2) = 4, L(3) = 6
// Bases: [L(1)=2, L(2)=4]
```

**Note:** Stride must be a power of 2 (due to GF(2) arithmetic properties).

#### 5.1.3 `zeros1D` - Broadcast Layout

```cpp
static LinearLayout zeros1D(
  int32_t size,
  StringAttr inDim,
  StringAttr outDim,
  int32_t outDimSize = 1
);
```

**Meaning:** For all `x`, `L(x) = 0` (broadcasting)

**Example:**
```cpp
auto L = LinearLayout::zeros1D(8, S("lane"), S("dim1"));
// L(0) = L(1) = ... = L(7) = 0
// Bases: [0, 0, 0]  (all bases are 0)
```

**Use case:** Broadcast pattern where all threads read the same value along one axis

### 5.2 Direct Construction from Bases

```cpp
LinearLayout(
  BasesT bases,                           // Map from input dimensions to basis vectors
  ArrayRef<StringAttr> outDimNames        // Output dimension names
);
```

**Example: 2D Swizzle**
```cpp
LinearLayout swizzle({
  {S("offset"), {
    {0, 1},  {0, 2},  {0, 4},  {0, 8},   // col bases
    {1, 0},  {2, 0},  {4, 4},  {8, 8}    // row bases (swizzled)
  }}
}, {S("dim0"), S("dim1")});
```

**Basis format:**
- `bases[inDim][i]` is the value of `L(inDim=2^i, other=0)`
- `bases[inDim][i][j]` is the value for the `j`-th output dimension

### 5.3 Layout Composition: `operator*`

```cpp
friend LinearLayout operator*(LinearLayout inner, LinearLayout outer);
```

**Meaning:** Direct sum - independently combine two layouts

**Rules:**
1. If input dimensions don't overlap: direct sum of two input spaces
2. If output dimensions overlap: both contribute to the same output (via XOR)
3. Order: `inner` dimensions are more minor

**Example 1: Building a 2D Identity**
```cpp
auto L = LinearLayout::identity1D(4, S("lane"), S("dim1"))
       * LinearLayout::identity1D(8, S("register"), S("dim0"));

// Inputs: (register, lane)
// Outputs: (dim0, dim1)
// L(reg=3, lane=2) = (dim0=3, dim1=2)
```

**Example 2: Sharing Output Dimensions**
```cpp
auto L1 = LinearLayout::identity1D(4, S("lane"), S("dim0"));
auto L2 = LinearLayout::identity1D(8, S("register"), S("dim0"));
auto L = L1 * L2;

// L(lane=2, register=3) = (dim0 = 2 ⊕ 3 = 1)
// Both inputs combine via XOR into the same output dimension
```

**Example 3: Combining with Broadcast**
```cpp
auto L = LinearLayout::zeros1D(4, S("lane"), S("dim1"))
       * LinearLayout::identity1D(8, S("register"), S("dim0"));

// L(lane=?, register=5) = (dim0=5, dim1=0)
// dim1 is always 0 regardless of lane value
```

### 5.4 Function Composition: `compose`

```cpp
LinearLayout compose(const LinearLayout &outer) const;
```

**Meaning:** `(outer ∘ this)(x) = outer(this(x))`

**Requirements:**
- `this`'s output dimensions = `outer`'s input dimensions
- `this->getOutDimSize(d) ≤ outer.getInDimSize(d)`

**Example:**
```cpp
// L1: (register) → (offset)
auto L1 = LinearLayout::identity1D(32, S("register"), S("offset"));

// L2: (offset) → (dim0, dim1)
auto L2 = /* some swizzled layout */;

// L3: (register) → (dim0, dim1)
auto L3 = L1.compose(L2);
```

### 5.5 Invert and Compose: `invertAndCompose`

```cpp
LinearLayout invertAndCompose(const LinearLayout &outer) const;
```

**Meaning:** Compute `C(x)` such that `this(x) = outer(C(x))`

**Key Use Case: Register → Shared Memory Store**
```cpp
// regLayout: (register,lane,warp) → (dim0,dim1)
// memLayout: (offset,block) → (dim0,dim1)

auto cvt = regLayout.invertAndCompose(memLayout);
// cvt: (register,lane,warp) → (offset,block)

// Meaning: Which shared offset should register[r,l,w] be stored to?
```

**Requirements:**
- `outer` must be surjective (covers all outputs)
- `outer`'s codomain ≥ `this`'s codomain
- `outer` can be non-injective (chooses smallest solution)

**Example Scenario:**
{% raw %}
```cpp
// 1) Register distribution: thread 0, reg 0 holds tensor[2,3]
auto regLayout = toLinearLayout(blockedEncoding);
assert(regLayout.apply({{S("register"),0}, {S("lane"),0}, {S("warp"),0}})
       == {{S("dim0"),2}, {S("dim1"),3}});

// 2) Shared memory: offset 10 corresponds to tensor[2,3]
auto memLayout = toLinearLayout(sharedEncoding);
assert(memLayout.apply({{S("offset"),10}})
       == {{S("dim0"),2}, {S("dim1"),3}});

// 3) Conversion: thread 0, reg 0 should store to offset 10
auto cvt = regLayout.invertAndCompose(memLayout);
assert(cvt.apply({{S("register"),0}, {S("lane"),0}, {S("warp"),0}})
       == {{S("offset"),10}});
```
{% endraw %}

### 5.6 Shape Transformations

#### `flattenIns/Outs`
```cpp
LinearLayout flattenIns() const;
LinearLayout flattenOuts() const;
```

Merge all input/output dimensions into a single dimension.

```cpp
// Input: (register:4, lane:8, warp:2)
auto flat = layout.flattenIns();
// Output: (register:64)  // 4×8×2, in minor-to-major order
```

#### `reshapeIns/Outs`
```cpp
LinearLayout reshapeIns(
  ArrayRef<std::pair<StringAttr, int32_t>> newInDims
) const;
```

Reshape dimensions: first flatten, then unstack.

```cpp
auto reshaped = layout.reshapeIns({
  {S("register"), 8},
  {S("lane"), 32}
});
```

#### `transposeIns/Outs`
```cpp
LinearLayout transposeOuts(ArrayRef<StringAttr> newOrder) const;
```

Reorder dimensions (typically used before reshape).

```cpp
auto transposed = layout.transposeOuts({S("dim1"), S("dim0")});
```

### 5.7 Apply Operations

#### Applying with Integers
```cpp
SmallVector<std::pair<StringAttr, int32_t>>
apply(ArrayRef<std::pair<StringAttr, int32_t>> ins) const;
```

{% raw %}
```cpp
auto result = layout.apply({
  {S("register"), 3},
  {S("lane"), 5},
  {S("warp"), 1}
});
// result: {{S("dim0"), ...}, {S("dim1"), ...}}
```
{% endraw %}

#### Applying with MLIR Values
```cpp
// include/triton/Conversion/TritonGPUToLLVM/Utility.h
SmallVector<std::pair<StringAttr, Value>>
applyLinearLayout(
  Location loc,
  RewriterBase &rewriter,
  const LinearLayout &layout,
  ArrayRef<std::pair<StringAttr, Value>> indices
);
```

Used for actual address calculation during lowering:
```cpp
Value regId = b.i32_val(0);
Value laneId = getLaneId(rewriter, loc);
Value warpId = getWarpId(rewriter, loc);

auto offsets = applyLinearLayout(loc, rewriter, cvt, {
  {S("register"), regId},
  {S("lane"), laneId},
  {S("warp"), warpId}
});
// offsets[0].second is the shared memory offset (Value)
```

---

## 6. Practical Examples: Step-by-Step Construction

### 6.1 Example 1: Simple 1D Distribution

**Scenario:** Distribute 32 elements across 4 threads, 8 elements per thread

{% raw %}
```cpp
// Each thread: 8 registers
// Thread 0: elements [0,4,8,12,16,20,24,28]
// Thread 1: elements [1,5,9,13,17,21,25,29]
// Thread 2: elements [2,6,10,14,18,22,26,30]
// Thread 3: elements [3,7,11,15,19,23,27,31]

auto layout =
  LinearLayout::identity1D(8, S("register"), S("dim0")) *
  LinearLayout::strided1D(4, 1, S("lane"), S("dim0"));

// Verification
assert(layout.apply({{S("register"),0}, {S("lane"),0}}) == {{S("dim0"),0}});
assert(layout.apply({{S("register"),1}, {S("lane"),0}}) == {{S("dim0"),4}});
assert(layout.apply({{S("register"),0}, {S("lane"),1}}) == {{S("dim0"),1}});
assert(layout.apply({{S("register"),2}, {S("lane"),3}}) == {{S("dim0"),11}});
```
{% endraw %}

**Explanation:**
- `register`: stride=4 (bases: [4,8,16,...])
- `lane`: stride=1 (bases: [1,2])
- XOR combination: `dim0 = register_contribution ⊕ lane_contribution`

### 6.2 Example 2: 2D Blocked Layout

**Scenario:**
- 16×16 tensor
- sizePerThread = [2, 2]: 4 elements per thread
- threadsPerWarp = [4, 4]: 16 threads per warp
- warpsPerCTA = [2, 2]: 4 warps per CTA

```cpp
// 1) Register layout: 2×2 blocks
auto regLayout =
  LinearLayout::identity1D(2, S("register"), S("dim0")) *
  LinearLayout::identity1D(2, S("register"), S("dim1"));

// 2) Lane layout: 4×4 blocks
auto laneLayout =
  LinearLayout::strided1D(4, 2, S("lane"), S("dim0")) *  // stride=2
  LinearLayout::strided1D(4, 2, S("lane"), S("dim1"));   // stride=2

// 3) Warp layout: 2×2 blocks
auto warpLayout =
  LinearLayout::strided1D(2, 8, S("warp"), S("dim0")) *  // stride=8
  LinearLayout::strided1D(2, 8, S("warp"), S("dim1"));   // stride=8

// 4) Combine
auto layout = regLayout * laneLayout * warpLayout;

// Verification: lane=5(=0b0101), register=2(=0b10)
// dim0: reg_contrib=0, lane_contrib=2×1=2, warp_contrib=0 → 2
// dim1: reg_contrib=2, lane_contrib=2×0=0, warp_contrib=0 → 2
{% raw %}
assert(layout.apply({
  {S("register"), 2},
  {S("lane"), 5},
  {S("warp"), 0}
}) == {{S("dim0"), 2}, {S("dim1"), 2}});
{% endraw %}
```

**Automatic BlockedEncodingAttr Conversion:**
```cpp
auto blocked = BlockedEncodingAttr::get(
  ctx,
  /*sizePerThread=*/{2,2},
  /*threadsPerWarp=*/{4,4},
  /*warpsPerCTA=*/{2,2},
  /*order=*/{1,0},  // row-major
  CTALayoutAttr::get(...)
);

auto layout = blocked.toLinearLayout({16,16});
// Produces the same result as manual construction above
```

### 6.3 Example 3: Shared Memory Swizzle

**Scenario:** 128×32 shared memory, 128-byte swizzle

```cpp
int numRows = 128;
int numCols = 32;  // For FP32
int vec = 8;       // 8 elements = 32 bytes
int perPhase = 4;  // 4 rows per phase
int maxPhase = 8;  // 8 phases

// Construct bases
std::vector<std::vector<int>> bases2D;

// Col bases (no swizzle)
for (int col = 1; col < numCols; col *= 2) {
  bases2D.push_back({0, col});
}

// Row bases (with swizzle)
for (int row = 1; row < numRows; row *= 2) {
  int phase = (row / perPhase) % maxPhase;
  int colSwizzle = vec * phase;
  bases2D.push_back({row, colSwizzle % numCols});
}

LinearLayout swizzled({
  {S("offset"), bases2D}
}, {S("dim0"), S("dim1")});

// Example: offset=17 (row=4, col=1)
// row basis: row=4 → phase=4/4%8=1, swizzle=8×1=8
// L(offset=17) = L(16) ⊕ L(1)
//              = (4, 8) ⊕ (0, 1)
//              = (4, 9)
```

### 6.4 Example 4: Register → Shared Conversion

**Complete Workflow:**

{% raw %}
```cpp
// 1) Define register layout
auto regLayout = BlockedEncodingAttr::get(...).toLinearLayout(shape);
// Inputs: (register, lane, warp, block)
// Outputs: (dim0, dim1)

// 2) Define shared memory layout
auto memLayout = swizzledSharedToLinearLayout(shape, sharedAttr);
// Inputs: (offset, block)
// Outputs: (dim0, dim1)

// 3) Compute conversion layout
auto cvtLayout = regLayout.invertAndCompose(memLayout);
// Inputs: (register, lane, warp, block)
// Outputs: (offset, block)

// 4) Address calculation during LLVM lowering
auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
Value blockId = getBlockId(rewriter, loc);

for (int regId = 0; regId < numRegs; regId++) {
  auto offsetPairs = applyLinearLayout(loc, rewriter, cvtLayout, {
    {S("register"), b.i32_val(regId)},
    {S("lane"), laneId},
    {S("warp"), warpId},
    {S("block"), blockId}
  });

  Value offset = offsetPairs[0].second;  // offset value
  Value ptr = gep(smemBase, offset);
  store(registerValues[regId], ptr);
}
```
{% endraw %}

### Conclusion
Now we have verified several common layouts in LL.
