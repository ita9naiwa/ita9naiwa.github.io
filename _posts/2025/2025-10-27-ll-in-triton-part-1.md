---
layout: article
title: "Linear Layout in Triton (1)"
category: "mlsys"
tag: "mlsys"
comment: true
key: 20251025
mathjax: true
---

### Introduction
As I study Triton, I realized that one of the most difficult concepts in Triton is `Linear Layout` (LL for short), but there are only a few resources to learn about it (see references). Therefore, I'm writing down my understanding and struggles with this interesting and important concept to help others who want to study Triton and Linear Layout.

**Warning**: This post is not meant to be very theoretical, but rather intuitive to help you understand what LL is and how it's used in the Triton codebase. Please refer to the official paper and actual implementation for deeper understanding.

#### Motivation
In Triton DSL, memory layout is presented at a very high level—rows and columns in a `torch.Tensor`-like way. But for actual hardware to run this, we somehow need to spread this data into an **appropriate** layout across the hardware. This is where LL comes into play.

Previously, Triton had several attributes to represent such transformations from logical representation to hardware representation. But as more hardware variants emerged and new instructions kept coming out, a need for a unified representation arose. Linear Layout (LL) provides a way to represent this transformation in a (1) **unified** and (2) **composable** manner (these two characteristics will be discussed in detail in the next parts).

## 1. The Basic Idea: Linear Layout (LL)

### 1.1 What is a Linear Layout?

At its core, a Linear Layout is a mapping from hardware location (e.g., warp, lane, register) to logical index (e.g., row, col). For example:

```
L(thread=4, warp=0) = (row=8, col=0)
```

> The key fact about LLs is, the mapping from (t,w) to (x,y) is not arbitrary. We only need to specify the value of L(t,w) at certain special points (namely, the values L(t,0) and L(0,w) where t and w are powers of 2), and from those we can compute all the other values of L.
> - LinearLayout.h:33-36

This might sound mysterious at first, but it's actually quite elegant. Let's build up the intuition step by step.

### 1.2 Setup: A Simple Example

Consider this scenario:

- A hardware with 4 warps and 4 threads in each warp
    - We have a total of 16 hardware locations: `4 × 4`
- A tensor $T$ with shape `4 by 4`

We want to define a mapping from thread (=t) and warp (=w) to column (col) and row
```
    L(t, w) = (col, row)
```

This mapping table can be presented as below:
```
               t/w    0     1     2    3
               0      ? (0,1) (0,2)    ?
    L(t,w) =   1  (1,1)     ?     ?    ?
               2  (2,2)     ?     ?    ?
               3      ?     ?     ?    ?
```

### 1.3 The Magic: XOR Rule (Linearity Rule)

Here's the key insight: you only need to specify these four values to define the whole linear layout! These special values are called the **"basis vectors"** or **"bases"** of the layout.

We complete the table by XOR-ing together the bases, according to the following rule (I write "⊕" for XOR):

**Linearity Rule:**
```
L(t1 ⊕ t2, w1 ⊕ w2) = L(t1, w1) ⊕ L(t2, w2)
```

The linearity rule plus our four basis choices allows us to fill in the whole table. Here's how we might compute some of the values:
```
   L(0,0) = L(1 ⊕ 1, 0 ⊕ 0) = L(1,0) ⊕ L(1,0) = (1,1) ⊕ (1,1) = (0,0)
   L(0,3) = L(0 ⊕ 0, 2 ⊕ 1) = L(0,2) ⊕ L(0,1) = (0,2) ⊕ (0,1) = (0,3)
   L(3,0) = L(2 ⊕ 1, 0 ⊕ 0) = L(2,0) ⊕ L(1,0) = (2,2) ⊕ (1,1) = (3,3)
   L(3,3) = L(3 ⊕ 0, 0 ⊕ 3) = L(3,0) ⊕ L(0,3) = (3,3) ⊕ (0,3) = (3,0).
```

Notice it's a consequence of the linearity rule that L(0,0) = (0,0), no
matter what values we chose for the table.

The whole table looks like this.
```
              t/w   0     1     2     3
              0  (0,0) (0,1) (0,2) (0,3)
    L(t,w) =  1  (1,1) (1,0) (1,3) (1,2)
              2  (2,2) (2,3) (2,0) (2,1)
              3  (3,3) (3,2) (3,1) (3,0).
```

Careful readers will recognize this as a classic **"swizzled" layout** where `(t, w) -> (t, w ⊕ t)`. To go from this formula to an LL, you only need to compute the results at input points `(0,1), (0,2), (1,0), and (2,0)`.

### 1.4 Why XOR? A Preview

You might be wondering: why XOR specifically? Why not regular addition? The answer lies in the mathematical structure called GF(2), which we'll explore in the next section. For now, just note that XOR has nice properties that make it perfect for representing hardware layouts efficiently.

---

## 2. Mathematical Foundations: GF(2) Linear Functions

Now that we've seen Linear Layout in action, let's understand the mathematical structure that makes it work. Don't worry—we'll keep it intuitive!

### 2.1 What is the GF(2) field?

**GF(2) is the mathematical structure behind bit operations:**
- **Elements**: `{0, 1}`
- **Addition**: XOR (`⊕`)
  - `0 ⊕ 0 = 0`, `0 ⊕ 1 = 1`, `1 ⊕ 0 = 1`, `1 ⊕ 1 = 0`
- **Multiplication**: AND (`×`)
  - `0 × 0 = 0`, `0 × 1 = 0`, `1 × 0 = 0`, `1 × 1 = 1`

### 2.2 Linear Functions in GF(2)

In standard linear algebra, a linear function can be written as:

```
L(a) = a₁·B₁ + a₂·B₂ + ... + aₘ·Bₘ
```

Where:
- `a = [a₁, a₂, ..., aₘ]` is the input vector
- `Bᵢ` is a basis vector
- We use real addition (+) and multiplication (·)

**In GF(2)**, we replace these operations with XOR and AND:

```
L(a) = (a₁ × B₁) ⊕ (a₂ × B₂) ⊕ ... ⊕ (aₘ × Bₘ)
```

This is the key connection: Linear Layout is just linear algebra over GF(2)!

### 2.3 Concrete example: a 4×4 matrix

**Matrix–vector product example (GF(2)):**

```
    | 1 0 0 0 |   | 0 |
    | 0 1 1 0 | × | 1 |
    | 0 0 1 1 |   | 1 |
    | 0 0 1 1 |   | 0 |
```

**Step-by-step calculation:**
```
:= (col1 × 0) ⊕ (col2 × 1) ⊕ (col3 × 1) ⊕ (col4 × 0)

:= |0|     |1|     |0|     |0|
  |0|  ⊕  |1|  ⊕  |1|  ⊕  |0|
  |0|     |0|     |1|     |0|
  |0|     |0|     |1|     |0|

:= |1|     |0|
  |1|  ⊕  |1|
  |0|     |1|
  |0|     |1|

:= |0|
  |0|
  |1|
  |1|
```

### 2.4 Representing bitwise integer

Compressing bit vectors into integers makes it more concise:

```
Matrix:  | 0001  0010  1110  1100 |  (each column interpreted as integer bits)
Input:   0110

Calculation:
  (first bit)   (second bit) (third bit) (fourth bit)
:= (0001 × 0) ⊕ (0010 × 1) ⊕ (1110 × 1) ⊕ (1100× 0)
:= 1100
```
- Input `a = 6 = 0b0110` has bits set at positions 2 and 3
- XOR-ing the 2nd basis (`2`) and 3rd basis (`14`) gives the result
- This is exactly how Linear Layout works!

---

## 3. Concrete Examples of Linear Layouts

Now that we understand the theory, let's look at some real examples of linear layouts you might encounter in practice.

### 3.1 Identity Layout

The simplest possible layout is the identity:
$$
L(x) = x
$$

Here the layout simply mirrors its input: `L(0) = 0`, `L(1) = 1`, and so on.

**Example with 8 elements (0-7):**

With eight elements, the indices act like 3-bit vectors, and a natural basis is:
- `L(1) = 1` → binary `001`
- `L(2) = 2` → binary `010`
- `L(4) = 4` → binary `100`

These basis vectors are independent over GF(2), so any index can be expressed as an XOR combination of them.

**How it works:**
- `L(001) = 001`
- `L(010) = 010`
- `L(100) = 100`

An index like `011` (decimal 3) is the XOR of the first two basis vectors:
$$
L(011) = L(010 \oplus 001) = L(010) \oplus L(001) = 010 \oplus 001 = 011
$$

The identity layout behaves exactly like a linear map on a GF(2) vector space: decompose any index as an XOR of basis indices, and the same combination is returned. Simple, but it forms the foundation for understanding more complex layouts.

### 3.2 Swizzled Layout

Now let's look at something more interesting—a swizzled layout that's commonly used in GPU memory access patterns.


**Setup:**
- 4 threads (thread 0~3), 4 warps (warp 0~3)
- 4×4 tensor
- Swizzle pattern: `L(t,w) = (t, w⊕t)`

**Basis definition:**
```
L(0, 1) = (0, 1)  ← warp basis 1
L(0, 2) = (0, 2)  ← warp basis 2
L(1, 0) = (1, 1)  ← thread basis 1 (swizzled!)
L(2, 0) = (2, 2)  ← thread basis 2 (swizzled!)
```

**Computing the full table:**
```
L(0,0) = L(1⊕1, 0⊕0) = L(1,0) ⊕ L(1,0) = (1,1) ⊕ (1,1) = (0,0)
L(1,1) = L(1, 0⊕1) = L(1,0) ⊕ L(0,1) = (1,1) ⊕ (0,1) = (1,0)
L(2,1) = L(2, 0⊕1) = L(2,0) ⊕ L(0,1) = (2,2) ⊕ (0,1) = (2,3)
L(3,3) = L(1⊕2, 1⊕2) = L(1,1) ⊕ L(2,2) = ...
```

**Resulting mapping table:**
```
         warp→  0     1     2     3
thread↓
  0          (0,0) (0,1) (0,2) (0,3)
  1          (1,1) (1,0) (1,3) (1,2)
  2          (2,2) (2,3) (2,0) (2,1)
  3          (3,3) (3,2) (3,1) (3,0)
```

This is a classic **swizzled pattern**: `(t, w⊕t)`. This kind of swizzling is important for avoiding bank conflicts in shared memory on GPUs!

### 3.3 Implementing Linear Layouts in Code

Now let's see how to actually implement these layouts in Triton's codebase. The API is quite elegant:

{% raw %}
```cpp
MLIRContext *ctx = ...;
StringAttr kThread = StringAttr::get(ctx, "thread");
StringAttr kWarp = StringAttr::get(ctx, "warp");
StringAttr kDim0 = StringAttr::get(ctx, "dim0");
StringAttr kDim1 = StringAttr::get(ctx, "dim1");

// Define bases - we only specify the power-of-2 basis vectors!
LinearLayout swizzled({
  {kThread, {
    {1, 1},  // L(thread=1, warp=0) = (dim0=1, dim1=1)
    {2, 2}   // L(thread=2, warp=0) = (dim0=2, dim1=2)
  }},
  {kWarp, {
    {0, 1},  // L(thread=0, warp=1) = (dim0=0, dim1=1)
    {0, 2}   // L(thread=0, warp=2) = (dim0=0, dim1=2)
  }}
}, {kDim0, kDim1});

// Usage: apply the layout to get the logical position
auto result = swizzled.apply({{kThread, 3}, {kWarp, 2}});
// result = {{kDim0, 3}, {kDim1, 1}}
// This means thread=3, warp=2 maps to position (3, 1) in the tensor
```
{% endraw %}

Notice how we only specify the basis vectors (powers of 2), and the `LinearLayout` class automatically computes all other values using the linearity rule!

---

## Conclusion
The key insight is that by specifying just a few basis vectors (powers of 2), we can compactly represent complex hardware-to-logical mappings. This makes Linear Layout both **memory-efficient** and **composable**—we can combine multiple layouts together.

In the next parts of this series, we'll explore:
- **Part 2**: Dimension naming conventions, core API, and practical examples
- **Part 3**: Layout conversions, lowering, and advanced topics