---
layout: article
title: "Linear Layout in Triton (2): LL Examples"
category: "mlsys"
tag: "mlsys"
comment: true
key: 20251026
mathjax: true
---

**Prerequisites**
- Part 1 of this series (concepts, GF(2) intuition, and basis vectors)
- A working knowledge of CUDA-thread hierarchy (CTA → warp → lane) or an equivalent AMD mental model

**What you will learn here**
- How to map hardware dimensions to layout inputs without memorising folklore
- When to pick each constructor (`identity1D`, `strided1D`, `zeros1D`, and manual bases)
- How to combine layouts safely through composition, inversion, and reshape calls
- End-to-end examples that you can adapt straight into lowering code


---
## 1. Dimension Naming Conventions and Semantics

Understanding dimension names is the gateway to reading and writing `LinearLayout` code fluently. This section maps Triton's encoding attributes to the input/output dimension vocabulary you will use throughout the API.

### 1.1. Input Dimensions — Hardware Hierarchy

Input dimensions describe **where** a value lives in hardware. The standard progression follows the GPU execution model from finest to coarsest granularity:

#### Distributed (Register-based) Layouts
```
"register" → Index into per-thread register slots (values held by one thread)
"lane"     → Thread ID within a warp (0–31 on NVIDIA, 0–63 on AMD)
"warp"     → Warp ID within a CTA (Cooperative Thread Array)
"block"    → Block (CTA) ID within the grid/cluster
```

**Why this order?**
- **`register`** is the most minor dimension. It represents data you can permute cheaply inside a single thread without cross-lane communication.
- **`lane`** is the next level. Lanes execute in lockstep within a warp, so this dimension often maps to contiguous memory strides.
- **`warp`** and **`block`** scale the same pattern across the SM and grid. They typically control coarser-grained tiling.

**Conversion example: BlockedEncodingAttr → LinearLayout**

```cpp
// sizePerThread=[4,2], threadsPerWarp=[8,4], warpsPerCTA=[2,2]
auto blocked = BlockedEncodingAttr::get(ctx,
  /*sizePerThread=*/{4, 2},
  /*threadsPerWarp=*/{8, 4},
  /*warpsPerCTA=*/{2, 2},
  /*order=*/{1, 0},
  CTALayoutAttr::get(/*...*/));
auto ll = blocked.toLinearLayout(shape);

// Resulting input dimensions (ins):
// - register: 0..7   (4×2 - 1 values per thread)
// - lane:     0..31  (8×4 - 1 threads per warp)
// - warp:     0..3   (2×2 - 1 warps per CTA)
// - block:    size depends on launch grid
```

These attributes decompose into basis vectors. For a quick sanity check, evaluate `ll.apply` at a few known points:

{% raw %}
```cpp
// Thread 0, register 0 should map to tensor index 0
assert(ll.apply({{S("register"), 0}, {S("lane"), 0}, {S("warp"), 0}})[0].second == 0);

// Thread 0, register 1 should map to the next element along the first axis
assert(ll.apply({{S("register"), 1}, {S("lane"), 0}, {S("warp"), 0}})[0].second == 1);
```
{% endraw %}

#### Shared Memory Layouts

Shared encodings replace the fine-grained hardware axes (`register`, `lane`, `warp`) with a single `offset` dimension:

```
"offset" → Linear index in shared memory (measured in elements, not bytes)
"block"  → CTA ID (optional; used when reasoning about multi-CTA scenarios)
```

**Swizzled shared example**

```cpp
auto shared = SwizzledSharedEncodingAttr::get(
  ctx,
  /*vec=*/8,
  /*perPhase=*/4,
  /*maxPhase=*/8,
  /*order=*/{1, 0},
  CTALayoutAttr::get(/*...*/));
auto ll = swizzledSharedToLinearLayout(shape, shared);

// Input dimensions:  offset, block
// Output dimensions: dim0, dim1
// The offset → (dim0, dim1) mapping encodes the swizzle pattern to avoid bank conflicts
```

Use this form when you need to reason about shared-memory bank conflicts at the IR level, before lowering to LLVM.

### 1.2. Output Dimensions — Logical Tensor Axes

Output dimensions always correspond to the **logical tensor axes** in their original order. The layout's `order` field controls internal memory layout (row-major vs. column-major), but the output dimension names remain canonical:

```
"dim0" → First tensor axis
"dim1" → Second tensor axis
"dim2" → Third tensor axis
...
```

**Key property:** All layouts—regardless of their internal `order`—speak the same output language. This uniformity is why `LinearLayout` composes so cleanly: every transformation operates on `dim0`, `dim1`, etc., and you never need to track multiple coordinate systems.

**Example:**
```cpp
// Both layouts have order={1,0} (column-major), but outputs are still named dim0, dim1
auto blocked = BlockedEncodingAttr::get(/*..., order={1,0}*/).toLinearLayout(shape);
auto shared  = swizzledSharedToLinearLayout(shape, sharedAttr);

// Both use the same output names:
// blocked:  (register,lane,warp) → (dim0, dim1)
// shared:   (offset) → (dim0, dim1)
```

### 1.3. Dimension Ordering, Flattening, and Reshaping

All `LinearLayout` operations interpret dimensions in **minor-to-major** order:
- The **first** dimension is the most minor (changes fastest in memory).
- The **last** dimension is the most major (changes slowest).

This convention affects how `flattenIns`, `flattenOuts`, `reshapeIns`, and `reshapeOuts` combine or split dimensions.

#### Flatten Rule of Thumb

```cpp
// Input dimensions: ["register", "lane", "warp"]
// Minor-to-major: register is most minor, warp is most major

auto flattened = layout.flattenIns();
// Result: single dimension with size = register_size × lane_size × warp_size
// Elements are ordered: register changes fastest, then lane, then warp
```

If you want a different flattening order (e.g., lane-major), transpose first:

```cpp
auto laneMajor = layout.transposeIns({S("lane"), S("register"), S("warp")}).flattenIns();
// Now lane changes fastest
```

#### Reshape Checklist

Before calling `reshapeIns` or `reshapeOuts`:

1. **Verify total size:** The product of new dimension sizes must equal the product of old dimension sizes.
2. **Plan transposes:** `reshapeIns` always flattens in minor-to-major order before splitting. If you need a different minor axis, transpose first.
3. **Check dimension counts:** Use `getInDimSize(name)` or `getOutDimSize(name)` to confirm sizes before reshaping. Silent dimension drops are a common source of bugs.

**Example workflow:**

{% raw %}
```cpp
// Original: (register:4, lane:8, warp:2) — total size 64
auto flat = layout.flattenIns();
// Flat: (register:64)

auto reshaped = flat.reshapeIns({{S("thread"), 32}, {S("block"), 2}});
// Reshaped: (thread:32, block:2)

// If you wanted warp to be minor instead of register, do this:
auto transposed = layout.transposeIns({S("warp"), S("lane"), S("register")});
auto flatWarpMajor = transposed.flattenIns();
// Now warp changes fastest
```
{% endraw %}

---
## 2. Practical Examples: Step-by-Step Construction

The following scenarios show how the APIs above compose into real Triton workflows. Each example walks through the mental model explicitly.

**Arithmetic view (bitfield intuition):** When two inputs contribute to the same output dimension via `operator*` and their basis bits do not overlap, the XOR accumulation is equivalent to standard integer addition over disjoint bitfields. For example,

```cpp
auto L1 = LinearLayout::identity1D(4, S("lane"), S("dim0"));
auto L2 = LinearLayout::strided1D(8, 4, S("register"), S("dim0"));  // shift register by 2 bits
auto L  = L1 * L2;

// Effective arithmetic form (disjoint bitfields):
// dim0 = (lane % 4) + ((register % 8) << 2)
//      = (lane % 4) + (register % 8) * 4
```

This works because `lane` occupies the lower 2 bits of `dim0`, while `register` occupies the next 3 bits (shifted by 2 via stride=4). If the bitfields overlap (e.g., you deliberately place both on the same bit positions), the combination is true XOR, not integer addition.

### 2.1. Example 1 — Simple 1D Distribution Across Lanes

**Scenario:** Distribute 32 logical elements across 4 threads so that each thread owns 8 elements in a strided pattern.

**Distribution pattern:**
- Thread 0: elements [0, 4, 8, 12, 16, 20, 24, 28]
- Thread 1: elements [1, 5, 9, 13, 17, 21, 25, 29]
- Thread 2: elements [2, 6, 10, 14, 18, 22, 26, 30]
- Thread 3: elements [3, 7, 11, 15, 19, 23, 27, 31]

**Construction:**

{% raw %}
```cpp
auto L1 = LinearLayout::identity1D(4, S("lane"), S("dim0"));
auto L2 = LinearLayout::identity1D(8, S("register"), S("dim0"));
auto layout = L1 * L2;

// Verification
assert(layout.apply({{S("register"), 0}, {S("lane"), 0}})[0].second == 0);
assert(layout.apply({{S("register"), 1}, {S("lane"), 0}})[0].second == 4);
assert(layout.apply({{S("register"), 0}, {S("lane"), 1}})[0].second == 1);
assert(layout.apply({{S("register"), 2}, {S("lane"), 3}})[0].second == 11);
```
{% endraw %}


```cpp
dim0 = (lane % 4) + ((register % 8) << 2)
     = (lane & 0b11) + ((register & 0b111) << 2)
```

### 2.2. Example 2 — 2D Blocked Layout (MMA Tile)

**Scenario:** A CTA cooperatively computes a 32×32 accumulator tile. Each warp handles a 16×8 MMA fragment (`m16n8k16`), and each lane packs two accumulator values in registers. We want a `LinearLayout` that maps `(register, lane)` coordinates back to logical `(row, col)` indices while respecting the register packing and warp tiling order used by NVIDIA's `nvidiaMmaTile`.

![mma_fig86]({{ "/assets/mma_fig86.png" | absolute_url }})
Image brought from https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#mma-16816-c-i8


**Goal:** Produce a layout equivalent to the relevant portion of `LinearLayout.cpp::nvidiaMmaTile`, but stripped down to just the accumulator mapping. The fragment is parameterised by
- `tileShape = {m=16, n=8}`
- `repOrder = {dimM, dimN}`
- `kWidth = 2`


```cpp
// Step 1: trivial CTA layout with the desired output dimension order
int rank = repOrder.size();
auto dimNames = standardOutDimNames(ctx, rank);
auto trivialShape = SmallVector<unsigned>(rank, 1);
LinearLayout ctaLayout =
    identityStandardND(S("register"), trivialShape, repOrder);

// Step 2: identify inner/outer logical axes
assert(rank == 2);
auto inner = order[0];
auto outer = order[1];

// Step 3: accumulate the blocked structure
assert(tileShape.size() == rank);
int m = tileShape[outer];   // rows handled per warp
int n = tileShape[inner];   // columns handled per warp

assert(m % 8 == 0);
assert(n % (kWidth * 4) == 0);
ctaLayout = ctaLayout *
            LinearLayout::identity1D(kWidth, S("register"), dimNames[inner]) *
            LinearLayout::identity1D(4, S("lane"), dimNames[inner]) *
            LinearLayout::identity1D(8, S("lane"), dimNames[outer]) *
            LinearLayout::identity1D(m / 8, S("register"), dimNames[outer]) *
            LinearLayout::identity1D(n / (kWidth * 4), S("register"),
                                      dimNames[inner]);
```

**What each factor contributes**
- `identity1D(kWidth, register → dimN)` walks across the `k`-width columns that a single lane updates.
- `identity1D(4, lane → dimN)` folds the 4-lane quad (threads 0–3, 4–7, …) into neighbouring columns.
- `identity1D(8, lane → dimM)` groups lanes into eight-row strips (`lane >> 2`).
- `identity1D(m/8, register → dimM)` repeats the strip across the vertical extent of the tile.
- `identity1D(n/(kWidth*4), register → dimN)` handles column repetitions when `n > kWidth*4`.

Together these six factors pack lane quads into rows, distribute row groups across registers, and make the layout cyclic over the 16×8 accumulator tile contained in each warp.

**Mapper / demapper sanity checks**

You can test the layout by reproducing the PTX warp-tile mapping in Python. The helper below mirrors how the layout turns `(register, lane)` pairs into `(outer=row, inner=col)` coordinates and back:

```python
import numpy as np
from collections import defaultdict

def mapper(register, lane, kWidth=8, m=16, n=8):
    """(lane, register) → (outer=row, inner=col)"""
    m_prime = m // 8                     # row repetition (two 8-row halves)
    n_prime = max(1, n // (kWidth * 4))  # column repetition (>=1 when n≥32)

    inner = (register % kWidth) * 4 + lane % 4
    outer = (lane // 4) % 8

    outer = outer * m_prime + (register // kWidth) % m_prime
    inner = inner * n_prime + (register // (kWidth * m_prime)) % n_prime
    return outer, inner

def demapper(outer, inner, kWidth=8, m=16, n=8):
    """(outer=row, inner=col) → (lane, register)"""
    m_prime = max(1, m // 8)
    n_prime = max(1, n // (kWidth * 4))

    inner_base = inner // n_prime
    rep_inner = inner % n_prime

    lane = (outer % 8) * 4 + (inner % 4)

    reg0 = (inner_base // 4) % kWidth
    reg1 = outer % m_prime
    reg2 = rep_inner

    register = reg0 + kWidth * (reg1 + m_prime * reg2)
    return lane, register
```

Running `demapper` across the 16×16 accumulator viewport prints the lane/register assignments for each row. You should see the familiar Hopper-style pattern where four consecutive columns belong to the same four-lane subgroup, and rows advance in steps of eight threads:

```python
kWidth, m, n = 8, 16, 16
array = [[demapper(row, col, kWidth=kWidth, m=m, n=n) for col in range(n)]
         for row in range(m)]

def tie(row):
    d = defaultdict(list)
    for reg, lane in row:
        d[reg].append(lane)
    return [(r, d[r]) for r in sorted(d.keys())]

for row, r in enumerate(array):
    print(f"row {row:2d}", tie(r))
```

The output packs each row into eight-lane groups (`lane >> 2`) and shows how registers cycle every `kWidth` columns. This is a direct confirmation that our simplified `ctaLayout` reproduces the PTX accumulator expectations—exactly what you need before calling `invertAndCompose` to stitch the accumulator to shared-memory staging buffers.
## 3. Core API Detailed Guide

This section walks through the core `LinearLayout` constructors and combinators, showing you how to build, manipulate, and apply layouts in practice, with extra commentary and quick sanity checks you can paste into your own code.

### 3.1. Basic Constructors — Build 1D Pieces

All three factory helpers return a **one-dimensional** layout. You stack them with `operator*` to produce richer, multi-dimensional structures.

#### 3.1.1. `identity1D` — Pass-Through Mapping

```cpp
static LinearLayout identity1D(
  int32_t size,        // Input domain size (must be power of two)
  StringAttr inDim,    // Input dimension name
  StringAttr outDim    // Output dimension name
);
```

**Semantics:** `L(x) = x` for `x ∈ [0, size)`

**Example:**
{% raw %}
```cpp
auto L = LinearLayout::identity1D(8, S("lane"), S("dim0"));
// L(0) = 0, L(1) = 1, ..., L(7) = 7
// Basis: [L(1)=1, L(2)=2, L(4)=4]

assert(L.apply({{S("lane"), 5}})[0].second == 5);
```
{% endraw %}

**When to use:**
- Contiguous, one-to-one mappings (e.g., `register` indices inside a single thread)
- As the "seed" layout when assembling more complex multi-dimensional structures

#### 3.1.2. `zeros1D` — Broadcast / Replication

```cpp
static LinearLayout zeros1D(
  int32_t size,
  StringAttr inDim,
  StringAttr outDim,
  int32_t outDimSize = 1
);
```

**Semantics:** `L(x) = 0` for all `x`

Every input value maps to zero in the chosen output dimension. This models **broadcasting**: all threads share the same tensor index along this axis.

**Example:**
{% raw %}
```cpp
auto L = LinearLayout::zeros1D(8, S("lane"), S("dim1"));
// L(0) = L(1) = ... = L(7) = 0
// Basis: [0, 0, 0] — all basis vectors are zero

assert(L.apply({{S("lane"), 5}})[0].second == 0);
```
{% endraw %}

**When to use:**
- Broadcasting a single value across all threads (e.g., a column vector broadcast)
- Modeling data replication in multi-CTA scenarios

**Optional parameter:** `outDimSize` lets you record the logical extent even though all values collapse to zero. This is useful when you later compose or invert the layout and need to track the full codomain.

### 3.2. Direct Construction from Basis Vectors

When the helper constructors can't express your pattern (swizzles, MMA layouts, custom permutations), build the layout manually from basis vectors.

```cpp
LinearLayout(
  BasesT bases,                           // map<inDim, vector<vector<int>>>
  ArrayRef<StringAttr> outDimNames        // ordered output dimension names
);
```

**Basis format:**
- `bases[inDim][i]` stores the value of `L(inDim = 2^i, others = 0)`
- `bases[inDim][i][j]` gives the contribution to the `j`-th output dimension (`outDimNames[j]`)

**Example: 2D Swizzle**
```cpp
LinearLayout swizzle({
  {S("offset"), {
    {0, 1},  {0, 2},  {0, 4},  {0, 8},   // Column basis (no swizzle)
    {1, 0},  {2, 0},  {4, 4},  {8, 8}    // Row basis (swizzled)
  }}
}, {S("dim0"), S("dim1")});

// Basis interpretation:
// L(offset=1) = (0, 1)  — column bit 0
// L(offset=2) = (0, 2)  — column bit 1
// ...
// L(offset=16) = (1, 0)  — row bit 0
// L(offset=32) = (2, 0)  — row bit 1
// L(offset=64) = (4, 4)  — row bit 2 with swizzle!
```

**Verification tip:** Whenever you hand-roll a basis, test it with a few manually computed points to ensure your phase/stride arithmetic is correct:

{% raw %}
```cpp
// offset=17 → binary decomposition: 16 + 1
auto result = swizzle.apply({{S("offset"), 17}});
// L(17) = L(16) ⊕ L(1) = (1,0) ⊕ (0,1) = (1,1)
assert(result == SmallVector{{S("dim0"), 1}, {S("dim1"), 1}});
```
{% endraw %}

### 3.3. Layout Combination with `operator*`

```cpp
friend LinearLayout operator*(LinearLayout inner, LinearLayout outer);
```

`operator*` builds the **direct sum** of two layouts. Think of it as:
- Concatenating the input spaces (inner inputs remain independent of outer inputs)
- XOR-accumulating contributions to shared output dimensions

**Rules of thumb:**
1. **Input dimensions can overlap** — overlapping in-dims are merged; use disjoint dims when you want Cartesian products.
2. **Shared output dimensions receive XOR contributions** from both operands.
3. **The left operand is more minor** — its dimensions change faster.

**Example 1: Building a 2D identity**
```cpp
auto L = LinearLayout::identity1D(4, S("lane"), S("dim1")) *
         LinearLayout::identity1D(8, S("register"), S("dim0"));

// Inputs: (register, lane)  — register is minor, lane is major
// Outputs: (dim0, dim1)
// L(register=3, lane=2) = (dim0=3, dim1=2)
```

**Example 2: Sharing an output dimension (XOR interaction)**
{% raw %}
```cpp
auto L1 = LinearLayout::identity1D(4, S("lane"), S("dim0"));
auto L2 = LinearLayout::identity1D(8, S("register"), S("dim0"));
auto L = L1 * L2;

// Both contribute to dim0, so their outputs XOR:
// L(lane=2, register=3) = (dim0 = 2 ⊕ 3 = 1)
assert(L.apply({{S("lane"), 2}, {S("register"), 3}})[0].second == 1);
```
{% endraw %}

**Example 3: Broadcast combined with identity**
```cpp
auto L = LinearLayout::zeros1D(4, S("lane"), S("dim1")) *
         LinearLayout::identity1D(8, S("register"), S("dim0"));

// L(lane=?, register=5) = (dim0=5, dim1=0)
// All lanes share dim1=0 (broadcast), but register contributes to dim0
```

**Mental model:** Use `operator*` whenever two hardware dimensions co-exist at runtime (e.g., `register × lane`, `warp × block`). The result is a layout that maps the Cartesian product of inputs to (possibly XOR-ed) outputs.

### 3.4. Function Composition — `compose`

```cpp
LinearLayout compose(const LinearLayout &outer) const;
```

**Semantics:** `(outer ∘ this)(x) = outer(this(x))`

Use `compose` when the **output** of one layout feeds directly into the **input** of another, creating a pipeline.

**Requirements:**
- `this`'s output dimensions must exactly match `outer`'s input dimensions (names and sizes)
- `this->getOutDimSize(d) ≤ outer.getInDimSize(d)` for all output dimensions `d`

**Example:**
{% raw %}
```cpp
// L1: (register) → (offset)
auto L1 = LinearLayout::identity1D(32, S("register"), S("offset"));

// L2: (offset) → (dim0, dim1)
auto L2 = /* some swizzled shared layout */;

// L3: (register) → (dim0, dim1)
auto L3 = L1.compose(L2);

// Now you can map register IDs directly to tensor coordinates:
auto coords = L3.apply({{S("register"), 5}});
// coords = {{S("dim0"), ...}, {S("dim1"), ...}}
```
{% endraw %}

**Why use `compose`?**
- **Modularity:** Test `L1` and `L2` independently before stitching them together.
- **Clarity:** Express multi-stage transformations (register → offset → tensor) explicitly in code.

### 3.5. Invert and Compose — `invertAndCompose`

```cpp
LinearLayout invertAndCompose(const LinearLayout &outer) const;
```

**Semantics:** Solve `this(x) = outer(C(x))` for `C`

This is the **workhorse function** for data layout conversions. It computes a layout `C` that maps from `this`'s input space to `outer`'s input space, ensuring both layouts produce the same tensor coordinates.

**Typical use case: Register → Shared Memory Store**

```cpp
// Source layout: (register, lane, warp) → (dim0, dim1)
auto regLayout = blocked.toLinearLayout(shape);

// Destination layout: (offset, block) → (dim0, dim1)
auto sharedLayout = swizzledSharedToLinearLayout(shape, sharedAttr);

// Conversion layout: (register, lane, warp) → (offset, block)
auto cvt = regLayout.invertAndCompose(sharedLayout);

// Meaning: "Which shared memory offset should (register=r, lane=l, warp=w) write to?"
```

**Requirements:**
- `outer` must be **surjective** (cover all possible outputs)
- `outer`'s codomain ≥ `this`'s codomain
- `outer` can be **non-injective** (multiple inputs map to same output); the algorithm picks the smallest pre-image

**Detailed example:**
{% raw %}
```cpp
// 1) Register layout: thread 0, register 0 holds tensor[2,3]
auto regLayout = toLinearLayout(blockedEncoding);
assert(regLayout.apply({{S("register"),0}, {S("lane"),0}, {S("warp"),0}})
       == SmallVector{{S("dim0"),2}, {S("dim1"),3}});

// 2) Shared layout: offset 10 corresponds to tensor[2,3]
auto memLayout = toLinearLayout(sharedEncoding);
assert(memLayout.apply({{S("offset"),10}})
       == SmallVector{{S("dim0"),2}, {S("dim1"),3}});

// 3) Conversion: (register=0, lane=0, warp=0) → offset=10
auto cvt = regLayout.invertAndCompose(memLayout);
assert(cvt.apply({{S("register"),0}, {S("lane"),0}, {S("warp"),0}})
       == SmallVector{{S("offset"),10}});
```
{% endraw %}

**Why it works:** Both `regLayout` and `memLayout` map to the same tensor space `(dim0, dim1)`. The inversion finds the shared memory offset that corresponds to the same tensor element.

### 3.6. Shape Transformations — Flatten, Reshape, Transpose

These utilities let you reorganize the input or output dimensions without changing the underlying mapping.

#### `flattenIns` / `flattenOuts`

```cpp
LinearLayout flattenIns() const;
LinearLayout flattenOuts() const;
```

Merge all input (or output) dimensions into a single dimension. Dimensions are flattened in **minor-to-major** order.

**Example:**
```cpp
// Input: (register:4, lane:8, warp:2)
auto flat = layout.flattenIns();
// Output: (register:64)  — size = 4×8×2
// Order: register changes fastest, then lane, then warp
```

#### `reshapeIns` / `reshapeOuts`

```cpp
LinearLayout reshapeIns(
  ArrayRef<std::pair<StringAttr, int32_t>> newInDims
) const;
LinearLayout reshapeOuts(
  ArrayRef<std::pair<StringAttr, int32_t>> newOutDims
) const;
```

Flatten all dimensions, then split into new named dimensions. The total size must remain the same.

**Example:**
```cpp
auto reshaped = layout.reshapeIns({
  {S("thread"), 32},
  {S("block"), 2}
});
// Flattens (register:4, lane:8, warp:2) → 64 elements
// Then splits into (thread:32, block:2)
```

#### `transposeIns` / `transposeOuts`

```cpp
LinearLayout transposeIns(ArrayRef<StringAttr> newOrder) const;
LinearLayout transposeOuts(ArrayRef<StringAttr> newOrder) const;
```

Reorder dimensions explicitly. Typically used **before** `reshape` when you need a non-default minor axis.

**Example:**
```cpp
// Original: (register:4, lane:8)
auto transposed = layout.transposeIns({S("lane"), S("register")});
// Result: (lane:8, register:4) — now lane is minor

auto flat = transposed.flattenIns();
// Flattens with lane changing fastest
```

**Workflow tip:**
```cpp
// If you want to flatten with warp as the minor axis:
auto transposed = layout.transposeIns({S("warp"), S("lane"), S("register")});
auto flat = transposed.flattenIns();
// Now warp changes fastest, then lane, then register
```

### 3.7. Evaluating Layouts — `apply` and `applyLinearLayout`

Two complementary helpers let you inspect or materialize a layout.

#### Integer `apply` — For Testing and Debugging

```cpp
SmallVector<std::pair<StringAttr, int32_t>>
apply(ArrayRef<std::pair<StringAttr, int32_t>> ins) const;
```

Feed integer inputs, get integer outputs. Perfect for:
- **Assertions** in unit tests
- **REPL exploration** to understand a layout
- **Manual verification** of basis correctness

**Example:**
{% raw %}
```cpp
auto result = layout.apply({
  {S("register"), 3},
  {S("lane"), 5},
  {S("warp"), 1}
});
// result: {{S("dim0"), ...}, {S("dim1"), ...}}

assert(result[0].second == expectedDim0);
assert(result[1].second == expectedDim1);
```
{% endraw %}

#### MLIR `applyLinearLayout` — For LLVM Lowering

```cpp
// Defined in include/triton/Conversion/TritonGPUToLLVM/Utility.h
SmallVector<std::pair<StringAttr, Value>>
applyLinearLayout(
  Location loc,
  RewriterBase &rewriter,
  const LinearLayout &layout,
  ArrayRef<std::pair<StringAttr, Value>> indices
);
```

Consumes MLIR `Value`s (SSA values), emits LLVM IR to compute the layout function at runtime.

**Example (during lowering):**
```cpp
Value regId = b.i32_val(0);
Value laneId = getLaneId(rewriter, loc);
Value warpId = getWarpId(rewriter, loc);

auto offsets = applyLinearLayout(loc, rewriter, cvtLayout, {
  {S("register"), regId},
  {S("lane"), laneId},
  {S("warp"), warpId}
});

Value sharedOffset = offsets[0].second;  // MLIR Value representing offset
// Use sharedOffset in a GEP or store instruction
```

**Critical reminder:** Always keep the dimension names in sync between `applyLinearLayout` inputs and the layout you created. Mismatched strings lead to silent wrong-code bugs!

---
## Conclusion

Armed with the patterns above, `LinearLayout` stops being a black box and becomes a practical tool you can reach for daily.
