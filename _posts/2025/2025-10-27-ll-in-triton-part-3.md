---
layout: article
title: "Linear Layout in Triton (3)"
category: "mlsys"
tag: "mlsys"
comment: true
key: 20251027_3
mathjax: true
---

## 7. Triton Layout Conversions

### 7.1 Conversion Entry Points

```cpp
// include/triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h

LinearLayout toLinearLayout(RankedTensorType type);
LinearLayout toLinearLayout(MemDescType type);
LinearLayout toLinearLayout(TensorOrMemDesc type);
LinearLayout toLinearLayout(ArrayRef<int64_t> shape, Attribute layout);
```

**Usage Example:**
```cpp
RankedTensorType tensorTy = ...;
auto layout = toLinearLayout(tensorTy);
```

### 7.2 BlockedEncodingAttr Conversion

**Implementation Location:** `lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp:926`

```cpp
LinearLayout BlockedEncodingAttr::toLinearLayout(ArrayRef<int64_t> shape) const {
  MLIRContext *ctx = getContext();
  auto order = getOrder();

  LinearLayout ctaLayout =
    identityStandardND(S("register"), getSizePerThread(), order) *
    identityStandardND(S("lane"), getThreadsPerWarp(), order) *
    identityStandardND(S("warp"), getWarpsPerCTA(), order);

  return combineCtaCgaWithShape(ctaLayout, getCTALayout(), shape);
}
```

**`identityStandardND` Function:**
```cpp
// Construct N-dimensional identity according to order
LinearLayout identityStandardND(
  StringAttr inDim,
  ArrayRef<unsigned> shape,
  ArrayRef<unsigned> order
) {
  // order[0] is most minor
  LinearLayout result = LinearLayout::empty();
  for (int i = 0; i < shape.size(); i++) {
    int dim = order[i];
    result *= LinearLayout::identity1D(
      shape[dim], inDim, S("dim" + std::to_string(dim))
    );
  }
  return result;
}
```

**Example:**
```cpp
// sizePerThread = [4, 2], order = [1, 0] (row-major)
identityStandardND(S("register"), {4, 2}, {1, 0})
// = identity1D(2, "register", "dim1") * identity1D(4, "register", "dim0")
// register 0~1 → dim1 variation
// register 2~3,4~5,6~7 → dim1 repetition with dim0 variation
```

### 7.3 SwizzledSharedEncodingAttr Conversion

**Implementation Location:** `lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp:85`

```cpp
LinearLayout swizzledSharedToLinearLayout(
  ArrayRef<int64_t> shape,
  SwizzledSharedEncodingAttr shared
) {
  // ... rank=1 special case ...

  // 2D+ case: apply swizzle to the lowest 2 dimensions
  int colDim = shared.getOrder()[0];
  int rowDim = shared.getOrder()[1];
  int numCols = shapePerCTA[colDim];
  int numRows = shapePerCTA[rowDim];

  std::vector<std::vector<int>> bases2D;

  // Column bases: simple identity
  for (int col = 1; col < numCols; col *= 2) {
    bases2D.push_back({0, col});
  }

  // Row bases: apply swizzle
  for (int row = 1; row < numRows; row *= 2) {
    int vec = shared.getVec();
    int perPhase = shared.getPerPhase();
    int maxPhase = shared.getMaxPhase();
    int swizzle = (vec * ((row / perPhase) % maxPhase)) % numCols;
    bases2D.push_back({row, swizzle});
  }

  LinearLayout ctaLayout = LinearLayout({
    {S("offset"), bases2D}
  }, {rowDimName, colDimName});

  // Higher dimension expansion
  for (int i = 2; i < rank; i++) {
    int dim = shared.getOrder()[i];
    ctaLayout *= LinearLayout::identity1D(
      shapePerCTA[dim], S("offset"), outDimNames[dim]
    );
  }

  return combineCtaCgaWithShape(ctaLayout, shared.getCTALayout(), shape);
}
```

**Swizzle Formula Explanation:**
```
phase = (row / perPhase) % maxPhase
colSwizzle = (vec * phase) % numCols
actualCol = baseCol ⊕ colSwizzle
```

**Bank Conflict Avoidance Principle:**
- Shared memory has 32 banks (NVIDIA) or 64 banks (AMD)
- Swizzle distributes consecutive elements in the same row across different banks
- Groups of `vec` size rotate per phase

### 7.4 NVMMASharedEncodingAttr Conversion (Hopper)

**Core Function:** `getCoreMatrixLinearLayout`

**Implementation Location:** `lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp:189`

```cpp
LinearLayout getCoreMatrixLinearLayout(
  NVMMASharedEncodingAttr shared,
  bool disableSwizzle
) {
  int elemBitWidth = shared.getElementBitWidth();
  int tileWidthBytes = shared.getSwizzlingByteWidth();
  int vec = shared.getVec();
  int perPhase = shared.getPerPhase();
  int maxPhase = shared.getMaxPhase();

  int tileRows = 8;
  int tileCols = 8 * std::max(16, tileWidthBytes) / elemBitWidth;
  bool isFp4Padded = shared.getFp4Padded();

  std::vector<std::vector<int>> bases2D;

  // Column bases
  for (int col = 1; col < tileCols; col *= 2) {
    if (isFp4Padded) {
      // FP4: only 8 out of 16 offsets are "real", the rest are padding
      // Packed representation: 16 → 8
      int colPacked = col / 16 * 8 + col % 8;
      bases2D.push_back({0, colPacked});
    } else {
      bases2D.push_back({0, col});
    }
  }

  // Row bases
  for (int row = 1; row < tileRows; row *= 2) {
    if (disableSwizzle) {
      bases2D.push_back({row, 0});
    } else if (isFp4Padded) {
      int colPadded = vec * ((row / perPhase) % maxPhase);
      int colPacked = colPadded / 16 * 8 + colPadded % 8;
      bases2D.push_back({row, colPacked});
    } else {
      bases2D.push_back({row, vec * ((row / perPhase) % maxPhase)});
    }
  }

  return LinearLayout({{S("offset"), bases2D}}, {S("dim0"), S("dim1")});
}
```

**FP4 Padding Explanation:**
- FP4 (4-bit float): 2 values per 1 byte
- NVMMA processes in 8-byte units
- Out of 16 FP4 values (8 bytes), only 8 are actually used, the rest are padding
- LinearLayout "folds" 16 offsets into 8 actual positions
- During `invertAndCompose`, smaller offsets are automatically selected

### 7.5 Combining CTA and CGA

**CTA (Cooperative Thread Array):** Layout within a single block
**CGA (Cooperative Grid Array):** Distribution across multiple blocks

```cpp
LinearLayout combineCtaCgaWithShape(
  LinearLayout ctaLayout,
  CTALayoutAttr cgaLayoutAttr,
  ArrayRef<int64_t> shape
) {
  // 1) Construct CGA layout
  auto cgaLayout = makeCgaLayout(cgaLayoutAttr);

  // 2) Multiply CTA and CGA
  auto layout = ctaLayout * cgaLayout;

  // 3) Expand to match shape
  layout = ensureLayoutNotSmallerThan(layout, outDims, shape);

  return layout;
}
```

**`makeCgaLayout` Implementation:**
```cpp
LinearLayout makeCgaLayout(CTALayoutAttr layout) {
  int rank = layout.getCTAOrder().size();
  LinearLayout ret = LinearLayout::empty();

  for (int i = 0; i < rank; i++) {
    int dim = layout.getCTAOrder()[i];
    int split = layout.getCTASplitNum()[dim];
    int ctas = layout.getCTAsPerCGA()[dim];

    // 'split' blocks are actually distributed, the rest are replicated (zeros)
    ret *= LinearLayout::identity1D(split, S("block"), S("dim"+dim)) *
           LinearLayout::zeros1D(ctas/split, S("block"), S("dim"+dim));
  }

  return ret.transposeOuts(standardOutDimNames);
}
```

**Example:**
```cpp
// CTAsPerCGA = [2, 4], CTASplitNum = [2, 2], CTAOrder = [1, 0]
//
// dim0: split=2, total=2 → fully distributed
//   block 0 → dim0 position 0
//   block 1 → dim0 position 1
//
// dim1: split=2, total=4 → partially distributed + replicated
//   block 0,2 → dim1 position 0
//   block 1,3 → dim1 position 1
```

---

## 8. Usage in Lowering

### 8.1 Typical Conversion Patterns

**Scenario:** `LocalStoreOp` - Storing from registers to shared memory

```cpp
LogicalResult LocalStoreOpConversion::matchAndRewrite(
  triton::gpu::LocalStoreOp op,
  OpAdaptor adaptor,
  ConversionPatternRewriter &rewriter
) const {
  auto loc = op.getLoc();
  Value src = adaptor.getSrc();  // register values
  Value dst = adaptor.getDst();  // Shared memory descriptor

  // 1) Get layouts
  auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
  auto dstTy = cast<MemDescType>(op.getDst().getType());

  auto srcLayout = toLinearLayout(srcTy);  // (register,lane,warp,block)→(dim0,dim1)
  auto dstLayout = toLinearLayout(dstTy);  // (offset,block)→(dim0,dim1)

  // 2) Compute conversion layout
  auto cvtLayout = srcLayout.invertAndCompose(dstLayout);
  // (register,lane,warp,block) → (offset,block)

  // 3) Get hardware IDs
  auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
  Value blockId = getBlockId(rewriter, loc);

  // 4) Shared memory base address
  Value smemBase = getSharedMemoryBase(dst);

  // 5) Store each register
  auto srcVals = unpackLLElements(loc, src, rewriter);
  int numRegs = srcVals.size();

  for (int regId = 0; regId < numRegs; regId++) {
    // 5a) Compute offset
    auto offsetPairs = applyLinearLayout(loc, rewriter, cvtLayout, {
      {S("register"), b.i32_val(regId)},
      {S("lane"), laneId},
      {S("warp"), warpId},
      {S("block"), blockId}
    });

    Value offset = offsetPairs[0].second;

    // 5b) Compute pointer and store
    Value ptr = gep(ptr_ty(ctx, 3), smemBase, offset);
    store(srcVals[regId], ptr);
  }

  rewriter.eraseOp(op);
  return success();
}
```

### 8.2 `applyLinearLayout` Implementation Principles

**Location:** `lib/Conversion/TritonGPUToLLVM/Utility.cpp:237`

```cpp
SmallVector<std::pair<StringAttr, Value>>
applyLinearLayout(
  Location loc,
  RewriterBase &rewriter,
  const LinearLayout &layout,
  ArrayRef<std::pair<StringAttr, Value>> indices
) {
  // 1) Initialize results to 0 for each output dimension
  SmallVector<Value> results(layout.getNumOutDims());
  for (auto &r : results) r = i32_val(0);

  // 2) Process each input dimension
  for (auto [inDim, inVal] : indices) {
    int inDimSizeLog2 = layout.getInDimSizeLog2(inDim);

    // 3) For each basis
    for (int bit = 0; bit < inDimSizeLog2; bit++) {
      // 3a) Extract bit-th bit of inVal
      Value bitVal = and_(inVal, i32_val(1 << bit));
      Value bitSet = icmp_ne(bitVal, i32_val(0));

      // 3b) Contribution from this basis
      auto basis = layout.getBasis(inDim, bit);

      // 3c) Conditional XOR for each output dimension
      for (int outIdx = 0; outIdx < basis.size(); outIdx++) {
        if (basis[outIdx] != 0) {
          Value contribution = select(
            bitSet,
            i32_val(basis[outIdx]),
            i32_val(0)
          );
          results[outIdx] = xor_(results[outIdx], contribution);
        }
      }
    }
  }

  // 4) Return with output dimension names
  SmallVector<std::pair<StringAttr, Value>> output;
  int idx = 0;
  for (auto outDim : layout.getOutDimNames()) {
    output.push_back({outDim, results[idx++]});
  }
  return output;
}
```

**Optimized Version:** The actual implementation includes optimizations such as additive stride detection and pre-computation.

### 8.3 MMA Operand Conversion Example

**Problem:** Convert dot product operand to MMA-friendly shared layout

```cpp
// AccelerateMatmul.cpp
Value getSharedMemoryMMAOperand(
  Value tensor,
  const SharedEncodingAttr &sharedLayout,
  Location loc,
  RewriterBase &rewriter
) {
  // 1) Source layout
  auto srcTy = cast<RankedTensorType>(tensor.getType());
  auto srcLayout = toLinearLayout(srcTy);

  // 2) Target shared layout
  auto dstTy = MemDescType::get(..., sharedLayout);
  auto dstLayout = toLinearLayout(dstTy);

  // 3) Verify conversion has no bank conflicts
  if (!isSwizzleOptimal(srcLayout, dstLayout)) {
    // Adjust swizzle parameters...
  }

  // 4) Allocate shared memory
  Value smem = allocSharedMemory(dstTy);

  // 5) Register → Shared store
  auto cvt = srcLayout.invertAndCompose(dstLayout);
  auto srcVals = unpack(tensor);

  for (int i = 0; i < srcVals.size(); i++) {
    auto offsets = applyLinearLayout(loc, rewriter, cvt, {
      {S("register"), i32_val(i)},
      {S("lane"), getLaneId()},
      {S("warp"), getWarpId()}
    });
    Value ptr = gep(smem, offsets[0].second);
    store(srcVals[i], ptr);
  }

  return smem;
}
```

---

## 9. Advanced Topics

### 9.1 Bank Conflict Analysis

**Shared Memory Banks:**
- NVIDIA: 32 banks, 4-byte granularity
- AMD: 64 banks, 4-byte granularity

**Conflict Condition:** Multiple threads in the same warp access different addresses in the same bank

**Verification with LinearLayout:**
```cpp
bool hasNoBankConflicts(const LinearLayout &layout) {
  // 1) Restrict layout to (lane, offset) → (dim0, dim1)
  auto laneToTensor = layout.sublayout({S("lane")}, {...});

  // 2) Inverse: (dim0, dim1) → lane
  auto tensorToLane = laneToTensor.invert();

  // 3) Shared offset → bank mapping
  // bank = (offset % numElements) % numBanks
  auto offsetToBank = LinearLayout::identity1D(
    numElements, S("offset"), S("bank")
  ).reshapeOuts({{S("bank"), numBanks}});

  // 4) (dim0,dim1) → offset (shared layout)
  auto tensorToOffset = sharedLayout.invert();

  // 5) Compose: (dim0,dim1) → lane, (dim0,dim1) → offset → bank
  auto tensorToLaneBank = tensorToLane * (tensorToOffset.compose(offsetToBank));

  // 6) Check if lane → bank mapping is injective
  // i.e., different lanes access different banks
  return tensorToLaneBank.sublayout({S("lane")}, {S("bank")}).isInjective();
}
```

### 9.2 `divideLeft` and `divideRight`

**Purpose:** Layout decomposition - finding C in `A = B * C`

```cpp
// divideLeft: compute C satisfying A = B * C
std::optional<LinearLayout> C = divideLeft(A, B);

// divideRight: compute C satisfying A = C * B
std::optional<LinearLayout> C = divideRight(A, B);
```

**Usage Example: Vectorization Extraction**
```cpp
// fullLayout: (register, lane) → (dim0)
// Goal: find consecutive vectors in register dimension

// 1) Pattern for 4 consecutive elements
auto vec4Pattern = LinearLayout::identity1D(4, S("register"), S("dim0"));

// 2) Divide fullLayout by vec4Pattern
auto quotient = divideLeft(fullLayout, vec4Pattern);

if (quotient.has_value()) {
  // fullLayout = vec4Pattern * quotient
  // i.e., can vectorize by grouping registers in sets of 4
  int numVec4Groups = quotient->getTotalInDimSize();
  // ...
}
```

### 9.3 `getFreeVariableMasks`

**Purpose:** Find input bits that don't affect the output

```cpp
auto masks = layout.getFreeVariableMasks();
// masks[inDim] = mask of bits irrelevant to output
```

**Usage Example: Detecting Broadcast Dimensions**
```cpp
auto layout = LinearLayout::zeros1D(8, S("lane"), S("dim0")) *
              LinearLayout::identity1D(4, S("register"), S("dim0"));

auto masks = layout.getFreeVariableMasks();
// masks[S("lane")] = 0b111  (all bits are free)
// masks[S("register")] = 0b000  (all bits are significant)

// We can tell that lane is a broadcast dimension
```

### 9.4 `ColumnAction`: Basis Reordering

**Purpose:** Rearrange basis columns of a specific input dimension

```cpp
// Reorder register values to optimize memory access patterns
ColumnAction action({2, 0, 1}, S("register"), /*inSizeLog2=*/3);

// Rearrange layout bases
auto newLayout = action.apply(oldLayout);

// Rearrange values accordingly
auto newValues = action.apply(oldValues);
```

**Example: Transpose**
```cpp
// Original: register [0,1,2,3,4,5,6,7] → (row,col)
// Goal: rearrange into (col,row) order

// 1) Compute required reordering pattern
auto action = computeTransposeAction(layout);

// 2) Rearrange both layout and values
auto transposedLayout = action.apply(layout);
auto transposedValues = action.apply(registerValues);
```

---

## 10. Debugging and Troubleshooting

### 10.1 Checking Layout with `toString()`

```cpp
auto layout = toLinearLayout(tensorTy);
llvm::errs() << "Layout: " << layout.toString() << "\n";
```

**Output Example:**
```
LinearLayout(
  ins={register:8, lane:32, warp:4},
  outs={dim0:16, dim1:16},
  bases={
    register: [[1,0], [2,0], [4,0], [0,1], [0,2], [0,4]],
    lane: [[8,0], [16,0], [0,8], [0,16], [0,32]],
    warp: [[0,64], [0,128]]
  }
)
```

### 10.2 Spot Checking with Apply

```cpp
// Check output for specific input
auto result = layout.apply({
  {S("register"), 5},
  {S("lane"), 10},
  {S("warp"), 1}
});

for (auto [dim, val] : result) {
  llvm::errs() << dim.str() << " = " << val << "\n";
}
```

### 10.3 Common Errors and Solutions

#### Error 1: "Layout is not surjective"

**Cause:** Bases do not fully cover the output space

**Solution:**
```cpp
// Bad example
LinearLayout bad({
  {S("lane"), {{1}, {2}}}  // bases: [1, 2]
}, {S("dim0")});
// Infers output size as 4, but cannot actually generate all of 0,1,2,3

// Good example 1: Explicitly specify output size
LinearLayout good1({
  {S("lane"), {{1}, {2}}}
}, {{S("dim0"), 4}}, /*requireSurjective=*/false);

// Good example 2: Provide complete bases
LinearLayout good2({
  {S("lane"), {{1}, {2}, {4}}}
}, {S("dim0")});  // [1,2,4] can generate all 0~7
```

#### Error 2: "Shape must be a power of 2"

**Cause:** Input shape is not a power of 2

**Solution:**
```cpp
// Bad example
auto layout = toLinearLayout({12, 20}, blockedAttr);  // 12, 20 are not P2

// Good example: Pad shape to P2
auto paddedShape = {16, 32};
auto layout = toLinearLayout(paddedShape, blockedAttr);

// Track actual shape separately
auto actualShape = {12, 20};
```

#### Error 3: "Dimension mismatch in compose"

**Cause:** Output/input dimension mismatch in `compose`

**Solution:**
```cpp
// L1: (register) → (offset)
// L2: (addr) → (dim0)  // Input is "addr" but L1 output is "offset"

// Solution 1: Change L1 output dimension name
auto L1fixed = L1.reshapeOuts({{S("addr"), L1.getTotalOutDimSize()}});
auto composed = L1fixed.compose(L2);  // OK

// Solution 2: Change L2 input dimension name
auto L2fixed = L2.reshapeIns({{S("offset"), L2.getTotalInDimSize()}});
auto composed = L1.compose(L2fixed);  // OK
```

#### Error 4: `invertAndCompose` failure

**Cause:** `outer` is non-surjective or codomain is too small

**Debugging:**
```cpp
auto outer = sharedLayout;

// 1) Check surjectivity
if (!outer.isSurjective()) {
  llvm::errs() << "Outer layout is not surjective!\n";
  llvm::errs() << "Total in size: " << outer.getTotalInDimSize() << "\n";
  llvm::errs() << "Total out size: " << outer.getTotalOutDimSize() << "\n";
}

// 2) Compare codomain sizes
for (auto outDim : outer.getOutDimNames()) {
  int innerSize = innerLayout.getOutDimSize(outDim);
  int outerSize = outer.getOutDimSize(outDim);
  if (outerSize < innerSize) {
    llvm::errs() << "Outer dimension " << outDim
                 << " too small: " << outerSize << " < " << innerSize << "\n";
  }
}
```

### 10.4 Writing Unit Tests

```cpp
// See unittest/Dialect/TritonGPU/LinearLayoutConversionsTest.cpp
TEST_F(LinearLayoutTest, MyCustomLayout) {
  auto layout = /* construct layout */;

  // 1) Validate sizes
  EXPECT_EQ(layout.getTotalInDimSize(), 64);
  EXPECT_EQ(layout.getTotalOutDimSize(), 256);

  // 2) Validate surjectivity
  EXPECT_TRUE(layout.isSurjective());

  // 3) Validate specific inputs
  auto result = layout.apply({{S("register"), 5}, {S("lane"), 3}});
  EXPECT_EQ(result[0].second, 23);  // dim0 = 23
  EXPECT_EQ(result[1].second, 7);   // dim1 = 7

  // 4) Validate inverse
  auto inv = layout.invert();
  auto roundTrip = layout.compose(inv);
  EXPECT_TRUE(roundTrip.isTrivialOver({S("dim0"), S("dim1")}));
}
```

---

## 11. FAQ

### Q1: Differences between LinearLayout and CuTe Layout?

**Commonalities:**
- Both are general-purpose programmable layouts
- Both unify special-case layouts

**Differences:**

| Aspect | LinearLayout | CuTe |
|------|--------------|------|
| Dimension names | Named dimensions (e.g., "register") | Numeric indices |
| Nesting | Not allowed | Allowed (can flatten) |
| Non-P2 sizes | Unsupported | Supported |
| Swizzle | Included in layout itself | Applied separately |
| Compile time | MLIR IR level (slowness OK) | C++ template (very fast) |
| Auto exploration | Possible (optimal layout search) | Manual selection |

### Q2: Why Hardware→Tensor direction?

**Reasons:**
1. **Functionality**: Tensor→Hardware can be one-to-many (during broadcast), Hardware→Tensor is always one-to-one
2. **Composition**: Only functions can compose cleanly
3. **Inversion**: `invertAndCompose` works safely as a pseudoinverse

**Intuition:**
- "What data does this thread hold?" (Hardware→Tensor) ✓
- "Who holds this data?" (Tensor→Hardware) - multiple answers possible ✗

### Q3: How to handle non-P2 shapes?

**Current constraint:** Most conversions require P2 shapes

**Solutions:**
1. **Padding**: Round shape up to P2
```cpp
int paddedM = nextPowerOf2(actualM);
int paddedN = nextPowerOf2(actualN);
auto layout = toLinearLayout({paddedM, paddedN}, encoding);
```

2. **Zeros bases**: Set some positions to 0
```cpp
// e.g., 64×64 tile but only use 60×60
// Set some bases to 0 to constrain range
```

3. **Mask**: Check valid range at runtime
```cpp
Value valid = and_(
  icmp_slt(row, i32_val(actualM)),
  icmp_slt(col, i32_val(actualN))
);
Value data = select(valid, loadedValue, zero);
```

### Q4: Differences between `compose` and `invertAndCompose`?

**`compose`:**
- `A.compose(B)` = B∘A = "A first, B later"
- A's output = B's input (names and sizes must match)
- Both are known forward functions

**`invertAndCompose`:**
- `A.invertAndCompose(B)` ≈ B⁻¹∘A
- A and B's **outputs** are in the same space (both tensor indices)
- B must be invertible (surjective)
- "Convert A's inputs to B's inputs"

**Analogy:**
```
compose:
  Temperature(Celsius) --A--> Temperature(Fahrenheit) --B--> Temperature(Kelvin)
  compose = direct conversion from Celsius to Kelvin

invertAndCompose:
  Address --A--> City name
  Zip code --B--> City name
  invertAndCompose = convert Address to Zip code
  (using City name as intermediary)
```

### Q5: When to use non-injective layouts?

**Example: Broadcast**
```cpp
auto layout =
  LinearLayout::identity1D(8, S("register"), S("dim0")) *
  LinearLayout::zeros1D(32, S("lane"), S("dim0"));

// All lanes access the same dim0 value
// lane=0, reg=3: dim0=3
// lane=5, reg=3: dim0=3
// lane=31, reg=3: dim0=3
```

**Example: Data Replication**
```cpp
// Two blocks hold the same data
auto cgaLayout = makeCgaLayout(CTALayoutAttr::get(
  /*CTAsPerCGA=*/{2, 1},
  /*CTASplitNum=*/{1, 1},  // dim0 has no split = replication
  ...
));

// Both block=0 and block=1 cover the same dim0 range
```

### Q6: Differences between `reshapeIns` and `transposeIns`?

**`transposeIns`:** Only changes dimension order (data unchanged)
```cpp
// Original: (register:4, lane:8)
auto t = layout.transposeIns({S("lane"), S("register")});
// Result: (lane:8, register:4)
// Data mapping is identical, only order changes
```

**`reshapeIns`:** Flatten dimensions then re-split
```cpp
// Original: (register:4, lane:8) - total 32 elements
auto r = layout.reshapeIns({{S("thread"), 32}});
// Result: (thread:32)
// Merge register and lane into one

auto r2 = r.reshapeIns({{S("x"), 8}, {S("y"), 4}});
// Result: (x:8, y:4)
// Split back into 2D
```

**When to transpose first?**
- reshape flattens in minor-to-major order
- If desired flatten order differs, transpose first

```cpp
// To flatten (register:4, lane:8) in lane-major order:
auto t = layout.transposeIns({S("lane"), S("register")});
auto f = t.flattenIns();
// lane varies faster
```

---

## 12. References

### 12.1 Source Files

**Core Headers:**
- `include/triton/Tools/LinearLayout.h` - Class definition, API, mathematical background
- `include/triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h` - Conversion API
- `include/triton/Conversion/TritonGPUToLLVM/Utility.h` - `applyLinearLayout`

**Implementation:**
- `lib/Tools/LinearLayout.cpp` - Core logic
- `lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp` - Layout conversion implementation
- `lib/Conversion/TritonGPUToLLVM/Utility.cpp` - Lowering utilities

**Tests:**
- `unittest/Dialect/TritonGPU/LinearLayoutConversionsTest.cpp` - Unit tests
- `python/test/unit/language/test_core.py` - E2E tests

### 12.2 Related Concepts

- **GF(2) Galois Field**: Finite field theory
- **Linear Algebra over GF(2)**: Binary matrix operations
- **Tensor Core Programming**: NVIDIA WMMA/MMA, AMD MFMA
- **Shared Memory Banking**: GPU memory system
- **CuTe Layout**: NVIDIA CUTLASS layout system

### 12.3 Example Code Locations

**Blocked → Linear:**
- `lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp:926`

**Swizzled Shared → Linear:**
- `lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp:85`

**NVMMA Shared → Linear:**
- `lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp:189`

**Store operation lowering:**
- `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/MemoryOpToLLVM.cpp`
- `third_party/amd/lib/TritonAMDGPUToLLVM/LoadStoreOpToLLVM.cpp`

**Layout conversion (register ↔ shared):**
- `lib/Conversion/TritonGPUToLLVM/ConvertLayoutOpToLLVM.cpp`

---

## Summary

**Linear Layout:**
1. Uniformly represents GPU tensor layouts as GF(2) linear functions
2. Defines complete mappings with a small number of basis vectors (sparse representation)
3. Computes all values using XOR linearity (efficient)
4. Standardizes composition, inversion, rearrangement, etc. as general operations
5. Unifies Triton's diverse hardware/layouts into a single framework

---

### References
- [LinearLayout.h](https://github.com/triton-lang/triton/blob/main/include/triton/Tools/LinearLayout.h)
- [Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using 𝔽2](https://arxiv.org/abs/2505.23819)
- [Triton Linear Layout: Concept](https://www.lei.chat/posts/triton-linear-layout-concept/#what-we-need)