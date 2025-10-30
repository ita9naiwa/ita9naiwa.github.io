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

## 4. Dimension Naming Conventions and Semantics

Understanding dimension names is the gateway to reading and writing `LinearLayout` code fluently. This section maps Triton's encoding attributes to the input/output dimension vocabulary you will use throughout the API.

### 4.1 Input Dimensions — Hardware Hierarchy

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

### 4.2 Output Dimensions — Logical Tensor Axes

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

### 4.3 Dimension Ordering, Flattening, and Reshaping

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

## 5. Core API Detailed Guide

This section walks through the core `LinearLayout` constructors and combinators, showing you how to build, manipulate, and apply layouts in practice, with extra commentary and quick sanity checks you can paste into your own code.

### 5.1 Basic Constructors — Build 1D Pieces

All three factory helpers return a **one-dimensional** layout. You stack them with `operator*` to produce richer, multi-dimensional structures.

#### 5.1.1 `identity1D` — Pass-Through Mapping

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

#### 5.1.2 `strided1D` — Stride Multiplication

```cpp
static LinearLayout strided1D(
  int32_t size,
  int32_t stride,      // Must be power of two
  StringAttr inDim,
  StringAttr outDim
);
```

**Semantics:** `L(x) = stride × x` for `x ∈ [0, size)`

Under the hood, everything is still GF(2) arithmetic, so the stride must be a power of two to preserve linearity.

**Example:**
{% raw %}
```cpp
auto L = LinearLayout::strided1D(4, 2, S("lane"), S("dim0"));
// L(0) = 0, L(1) = 2, L(2) = 4, L(3) = 6
// Basis: [L(1)=2, L(2)=4]

assert(L.apply({{S("lane"), 3}})[0].second == 6);
```
{% endraw %}

**When to use:**
- Warp-level tiling where lanes map to strided tensor indices
- Combining with `identity1D` on another axis to form 2D or 3D tiles

**Tip:** Combine `strided1D` across multiple dimensions to build blocked patterns:
```cpp
auto L = LinearLayout::strided1D(4, 2, S("lane"), S("dim0")) *
         LinearLayout::strided1D(4, 2, S("lane"), S("dim1"));
// Maps lanes to a 2D strided grid
```

#### 5.1.3 `zeros1D` — Broadcast / Replication

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

### 5.2 Direct Construction from Basis Vectors

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

### 5.3 Layout Combination with `operator*`

```cpp
friend LinearLayout operator*(LinearLayout inner, LinearLayout outer);
```

`operator*` builds the **direct sum** of two layouts. Think of it as:
- Concatenating the input spaces (inner inputs remain independent of outer inputs)
- XOR-accumulating contributions to shared output dimensions

**Rules of thumb:**
1. **Input dimensions must be disjoint** — otherwise you redefine a dimension.
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

### 5.4 Function Composition — `compose`

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

### 5.5 Invert and Compose — `invertAndCompose`

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

### 5.6 Shape Transformations — Flatten, Reshape, Transpose

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

### 5.7 Evaluating Layouts — `apply` and `applyLinearLayout`

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

## 6. Practical Examples: Step-by-Step Construction

The following scenarios show how the APIs above compose into real Triton workflows. Each example walks through the mental model explicitly.

**Arithmetic view (bitfield intuition):** When two inputs contribute to the same output dimension via `operator*` and their basis bits do not overlap, the XOR accumulation is equivalent to standard integer addition over disjoint bitfields. For example,

```cpp
auto L1 = LinearLayout::identity1D(4, S("lane"), S("dim0"));
auto L2 = LinearLayout::identity1D(8, S("register"), S("dim0"));
auto L  = L1 * L2;

// Effective arithmetic form (disjoint bitfields):
// dim0 = (lane % 4) + ((register % 8) << 2)
//      = (lane % 4) + (register % 8) * 4
```

This works because `lane` occupies the lower 2 bits of `dim0`, while `register` occupies the next 3 bits. If the bitfields overlap (e.g., you deliberately place both on the same bit positions), the combination is true XOR, not integer addition.

### 6.1 Example 1 — Simple 1D Distribution Across Lanes

**Scenario:** Distribute 32 logical elements across 4 threads so that each thread owns 8 elements in a strided pattern.

**Distribution pattern:**
- Thread 0: elements [0, 4, 8, 12, 16, 20, 24, 28]
- Thread 1: elements [1, 5, 9, 13, 17, 21, 25, 29]
- Thread 2: elements [2, 6, 10, 14, 18, 22, 26, 30]
- Thread 3: elements [3, 7, 11, 15, 19, 23, 27, 31]

**Construction:**

{% raw %}
```cpp
auto layout =
  LinearLayout::identity1D(8, S("register"), S("dim0")) *
  LinearLayout::strided1D(4, 1, S("lane"), S("dim0"));

// Verification
assert(layout.apply({{S("register"), 0}, {S("lane"), 0}})[0].second == 0);
assert(layout.apply({{S("register"), 1}, {S("lane"), 0}})[0].second == 4);
assert(layout.apply({{S("register"), 0}, {S("lane"), 1}})[0].second == 1);
assert(layout.apply({{S("register"), 2}, {S("lane"), 3}})[0].second == 11);
```
{% endraw %}

**Explanation:**
- **Register basis:** `identity1D(8, ...)` creates basis `[1, 2, 4]` for dim0
  - register=1 → 0 + 4×1 = 4
  - register=2 → 0 + 4×2 = 8
  - ...
- **Lane basis:** `strided1D(4, 1, ...)` creates basis `[1, 2]` for dim0
  - lane=1 → 1
  - lane=2 → 2
  - lane=3 → 3
- **XOR combination:** `dim0 = register_contrib ⊕ lane_contrib`
  - (register=2, lane=3) → 8 ⊕ 3 = 11 ✓

**Key takeaways:**
- Registers contribute the **coarse stride** (multiples of 4 in this case)
- Lanes inject the **fine-grained offset** inside the stride
- XOR combines both contributions—always verify with a few spot checks!

**Arithmetic view:** With disjoint bitfields, XOR equals integer add on those fields:

```cpp
// dim0 = (lane % 4) + ((register % 8) << 2)
//      = (lane & 0b11) + ((register & 0b111) << 2)
```

### 6.2 Example 2 — 2D Blocked Layout (BlockedEncodingAttr)

**Scenario:**
- Tensor shape: 16 × 16
- `sizePerThread = [2, 2]` → each thread handles a 2×2 patch (4 elements total)
- `threadsPerWarp = [4, 4]` → 16 threads per warp, arranged in a 4×4 grid
- `warpsPerCTA = [2, 2]` → 4 warps per CTA, arranged in a 2×2 grid

**Manual construction (step by step):**

```cpp
// Step 1: Register layout (2×2 block per thread)
auto regLayout =
  LinearLayout::identity1D(2, S("register"), S("dim0")) *
  LinearLayout::identity1D(2, S("register"), S("dim1"));
// register=0 → (0,0), register=1 → (1,0), register=2 → (0,1), register=3 → (1,1)

// Step 2: Lane layout (4×4 grid of threads, stride=2)
auto laneLayout =
  LinearLayout::strided1D(4, 2, S("lane"), S("dim0")) *
  LinearLayout::strided1D(4, 2, S("lane"), S("dim1"));
// lane covers [0,8) in each dimension with stride 2

// Step 3: Warp layout (2×2 grid of warps, stride=8)
auto warpLayout =
  LinearLayout::strided1D(2, 8, S("warp"), S("dim0")) *
  LinearLayout::strided1D(2, 8, S("warp"), S("dim1"));
// warp covers [0,16) in each dimension with stride 8

// Step 4: Combine all levels
auto layout = regLayout * laneLayout * warpLayout;
```

**Detailed verification:**

Let's check `(register=2, lane=5, warp=0)`:
- lane=5 = 0b0101
- register=2 = 0b10

Compute dim0:
- register contrib: bit 1 of register in dim0 position → 0
- lane contrib: bits of lane in dim0 position → 2×1 = 2 (from lane bit 0)
- warp contrib: 0
- **dim0 = 0 ⊕ 2 ⊕ 0 = 2**

Compute dim1:
- register contrib: bit 1 of register in dim1 position → 2
- lane contrib: bits of lane in dim1 position → 2×0 = 0
- warp contrib: 0
- **dim1 = 2 ⊕ 0 ⊕ 0 = 2**

```cpp
{% raw %}
assert(layout.apply({
  {S("register"), 2},
  {S("lane"), 5},
  {S("warp"), 0}
}) == (SmallVector{
  std::pair{S("dim0"), 2},
  std::pair{S("dim1"), 2}
}));
{% endraw %}
```

**Automatic conversion (using BlockedEncodingAttr):**

```cpp
auto blocked = BlockedEncodingAttr::get(ctx,
  /*sizePerThread=*/{2, 2},
  /*threadsPerWarp=*/{4, 4},
  /*warpsPerCTA=*/{2, 2},
  /*order=*/{1, 0},  // row-major
  CTALayoutAttr::get(/*...*/));

auto layoutFromAttr = blocked.toLinearLayout({16, 16});
```

The automatic conversion should reproduce the same layout. Verify equality by:
1. Comparing `layout.toString()` with `layoutFromAttr.toString()`
2. Probing a dozen random coordinates with `apply(...)`
3. Checking that both have the same basis vectors

**Arithmetic view:** Decompose each input into 2D bitfields and sum with strides:

```cpp
int regX  =  (register >> 0) & 0b1;    // sizePerThread.x = 2
int regY  =  (register >> 1) & 0b1;    // sizePerThread.y = 2
int laneX =  (lane >> 0) & 0b11;       // threadsPerWarp.x = 4
int laneY =  (lane >> 2) & 0b11;       // threadsPerWarp.y = 4
int warpX =  (warp >> 0) & 0b1;        // warpsPerCTA.x = 2
int warpY =  (warp >> 1) & 0b1;        // warpsPerCTA.y = 2

int dim0 = regX + (laneX << 1) + (warpX << 3);  // 1, 2, 8 strides
int dim1 = regY + (laneY << 1) + (warpY << 3);
```

### 6.3 Example 3 — Shared Memory Swizzle

**Goal:** Model a 128×32 shared-memory tile with a 128-byte swizzle pattern, common on NVIDIA GPUs to avoid bank conflicts.

**Background:**
- Shared memory on NVIDIA GPUs has 32 banks (4-byte wide)
- Consecutive elements in a row must map to different banks
- Swizzling XORs row information into the column index to spread accesses across banks

**Swizzle parameters:**
```cpp
int numRows = 128;
int numCols = 32;  // for FP32 elements
int vec = 8;       // 8 elements = 32 bytes = 128 bits
int perPhase = 4;  // 4 rows per phase
int maxPhase = 8;  // 8 phases total
```

**Construction:**

```cpp
std::vector<std::vector<int>> bases2D;

// Column basis (no swizzle applied to column bits)
for (int col = 1; col < numCols; col *= 2) {
  bases2D.push_back({0, col});  // contributes only to dim1
}

// Row basis (swizzle applied)
for (int row = 1; row < numRows; row *= 2) {
  int phase = (row / perPhase) % maxPhase;
  int colSwizzle = (vec * phase) % numCols;
  bases2D.push_back({row, colSwizzle});  // row with column XOR
}

LinearLayout swizzled({
  {S("offset"), bases2D}
}, {S("dim0"), S("dim1")});
```

**Swizzle formula explained:**
```
phase = (row / perPhase) % maxPhase
actualCol = baseCol ⊕ (vec × phase)
```

For each group of `perPhase` rows (4 rows), the phase increments. The column index gets XOR-ed with `vec × phase`, rotating the bank assignments.

**Manual verification:**

Let's check `offset = 17`:
- Binary decomposition: 17 = 16 + 1 = 0b10001
- Row contribution: offset bit 4 → row=16, phase=(16/4)%8=4, swizzle=8×4=32 mod 32=0? No wait, let me recalculate.

Actually, offset 17 in a row-major 128×32 layout:
- offset 17 = row 0, col 17
- Column bits: 1 = 0b00001
- L(offset=1) = (0, 1) — from column basis bit 0
- L(offset=16) = (0, 16) — from column basis bit 4
- L(17) = L(16) ⊕ L(1) = (0, 16) ⊕ (0, 1) = (0, 17)

Wait, let me re-read the basis construction. The first `log2(numCols)` basis vectors are for columns, then the row bases come after. So:

Basis order:
```
bases2D[0..4]:  column bases (col=1, 2, 4, 8, 16)
bases2D[5..11]: row bases (row=1, 2, 4, 8, 16, 32, 64)
```

For offset=17 = 0b10001:
- Bit 0 set → use bases2D[0] = (0, 1)
- Bit 4 set → use bases2D[4] = (0, 16)
- L(17) = (0,1) ⊕ (0,16) = (0, 17)

For offset with row component, e.g., offset = numCols×4 + 1 = 32×4 + 1 = 129:
- This is (row=4, col=1)
- Bit 0 set → bases2D[0] = (0, 1)
- Bit 7 set → bases2D[7] which corresponds to row=4
  - row=4, phase=(4/4)%8=1, swizzle=8×1=8
  - bases2D[7] = (4, 8)
- L(129) = (0,1) ⊕ (4,8) = (4, 9)

**Verification code:**
{% raw %}
```cpp
// offset 129 = row 4, col 1
auto result = swizzled.apply({{S("offset"), 129}});
// Expected: (row=4, actualCol = 1 ⊕ 8 = 9)
assert(result == SmallVector{{S("dim0"), 4}, {S("dim1"), 9}});
```
{% endraw %}

**Tip:** Print a few more offsets if the swizzle pattern feels off; mismatched phases usually show up immediately. You can dump the full basis with `swizzled.toString()` to inspect each power-of-two contribution.

**Arithmetic view:** Closed-form for row/column and swizzle.

```cpp
int row   = offset / numCols;
int col   = offset % numCols;
int phase = (row / perPhase) % maxPhase;
int col2  = col ^ ((vec * phase) % numCols);  // XOR-based swizzle

int dim0 = row;
int dim1 = col2;
```

### 6.4 Example 4 — Register → Shared Conversion Pipeline

This is the **complete workflow** you'll eventually wire into Triton's lowering passes when converting data from register layouts to shared memory layouts.

**Scenario:** Store tensor data from distributed register layout to swizzled shared memory.

**Step-by-step breakdown:**

{% raw %}
```cpp
// Step 1: Define the source layout (distributed in registers)
// (register, lane, warp, block) → (dim0, dim1)
auto regLayout = BlockedEncodingAttr::get(
  ctx,
  /*sizePerThread=*/{4, 2},
  /*threadsPerWarp=*/{8, 4},
  /*warpsPerCTA=*/{2, 2},
  /*order=*/{1, 0},
  CTALayoutAttr::get(/*...*/)
).toLinearLayout(shape);

// Step 2: Define the destination layout (shared memory with swizzle)
// (offset, block) → (dim0, dim1)
auto sharedLayout = swizzledSharedToLinearLayout(shape, sharedAttr);

// Step 3: Compute the conversion layout
// (register, lane, warp, block) → (offset, block)
auto cvtLayout = regLayout.invertAndCompose(sharedLayout);

// What does cvtLayout tell us?
// For each (register, lane, warp, block) tuple, it gives the corresponding
// (offset, block) in shared memory that should receive that value.
```
{% endraw %}

**Conceptual verification:**

Before lowering, you can verify the conversion makes sense:

{% raw %}
```cpp
// Pick a specific register value
int regId = 3;
int laneId = 5;
int warpId = 1;

// Where does this register map in the tensor?
auto tensorCoords = regLayout.apply({
  {S("register"), regId},
  {S("lane"), laneId},
  {S("warp"), warpId}
});
// tensorCoords = {{S("dim0"), X}, {S("dim1"), Y}}

// Where should we write this in shared memory?
auto smemCoords = cvtLayout.apply({
  {S("register"), regId},
  {S("lane"), laneId},
  {S("warp"), warpId}
});
// smemCoords = {{S("offset"), offsetVal}}

// Verify: does sharedLayout map offsetVal to the same (X, Y)?
auto check = sharedLayout.apply({{S("offset"), smemCoords[0].second}});
assert(check == tensorCoords);  // Should match!
```
{% endraw %}

**Arithmetic view:** For a swizzled shared layout, compute the write offset directly.

```cpp
// Given (X,Y) = regLayout(register, lane, warp).  Then:
int row   = X;
int col2  = Y;                                // swizzled column
int phase = (row / perPhase) % maxPhase;
int col   = col2 ^ ((vec * phase) % numCols); // invert the swizzle
int offset = row * numCols + col;             // destination in shared
```

**Step 4: Use during LLVM lowering**

{% raw %}
```cpp
// Inside LocalStoreOpConversion or similar
auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
Value blockId = getBlockId(rewriter, loc);
Value smemBase = getSharedMemoryBase(dst);

// Unpack the register values (LLVM struct → individual Values)
auto registerValues = unpackLLElements(loc, src, rewriter);
int numRegs = registerValues.size();

for (int regId = 0; regId < numRegs; ++regId) {
  // Compute the shared memory offset for this register
  auto offsetPairs = applyLinearLayout(loc, rewriter, cvtLayout, {
    {S("register"), b.i32_val(regId)},
    {S("lane"), laneId},
    {S("warp"), warpId},
    {S("block"), blockId}
  });

  Value sharedOffset = offsetPairs[0].second;

  // Compute pointer and store
  Value ptr = gep(ptr_ty(ctx, 3), smemBase, sharedOffset);
  store(registerValues[regId], ptr);
}
```
{% endraw %}

**Implementation tips:**

1. **Dimension name consistency:** Keep the dimension names immutable across calls. Typos or mismatches (`"lane"` vs `"lanes"`) lead to silent wrong-code that's hard to debug.

2. **Bounds checking:** Guard the loop with the actual number of registers you hold. Layouts happily accept out-of-range values and will produce garbage offsets.

3. **Debugging with `toString()`:** Consider dumping `cvtLayout.toString()` during development. It prints the basis in a compact form that lets you spot swizzle errors quickly:
   ```cpp
   llvm::errs() << "Conversion layout:\n" << cvtLayout.toString() << "\n";
   ```

4. **Test incrementally:**
   - First verify `regLayout` with a few spot checks
   - Then verify `sharedLayout` independently
   - Finally verify `cvtLayout` by checking round-trip consistency

5. **Watch for block dimensions:** If your layout includes a `block` dimension but you're lowering a single-CTA kernel, you can often ignore it (set to 0). But for multi-CTA kernels, make sure you pass the correct `blockId`.

**Common pitfalls:**

- **Forgetting to unpack:** `src` is often an LLVM struct; you must unpack it into individual register values before storing.
- **Wrong pointer type:** Make sure the GEP uses the correct address space (3 for shared memory on NVIDIA).
- **Swizzle parameter mismatches:** If `cvtLayout` looks wrong, double-check that `sharedAttr`'s `vec`, `perPhase`, and `maxPhase` match the actual shared memory allocation.

---

## Conclusion

Armed with the patterns above, `LinearLayout` stops being a black box and becomes a practical tool you can reach for daily.
