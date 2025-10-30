# Linear Layout 완전 가이드 (Triton)

## 목차
1. [Linear Layout이란 무엇인가?](#1-linear-layout이란-무엇인가)
2. [수학적 기초: GF(2) 선형 함수](#2-수학적-기초-gf2-선형-함수)
3. [기저(Basis)의 개념과 선형성 규칙](#3-기저basis의-개념과-선형성-규칙)
4. [차원 명명 체계와 의미](#4-차원-명명-체계와-의미)
5. [핵심 API 상세 가이드](#5-핵심-api-상세-가이드)
6. [실전 예제: 단계별 구성](#6-실전-예제-단계별-구성)
7. [Triton 레이아웃 변환](#7-triton-레이아웃-변환)
8. [Lowering에서의 활용](#8-lowering에서의-활용)
9. [고급 주제](#9-고급-주제)
10. [디버깅과 문제 해결](#10-디버깅과-문제-해결)
11. [FAQ](#11-faq)

---

## 1. Linear Layout이란 무엇인가?

### 1.1 핵심 아이디어

**Linear Layout(LL)**은 Triton 컴파일러에서 텐서 데이터가 GPU 하드웨어에 어떻게 분산되어 있는지를 표현하는 통합된 수학적 프레임워크입니다.

**기본 개념:**
- **함수 관점**: LL은 "하드웨어 위치" → "논리 텐서 인덱스" 매핑 함수입니다
- **예시**: `L(thread=5, warp=2) = (row=7, col=3)`은 "워프 2의 쓰레드 5가 텐서의 [7,3] 위치 값을 보유"를 의미합니다

**왜 하드웨어→텐서 방향인가?**
- 하나의 텐서 요소가 여러 하드웨어 위치에 복제될 수 있습니다 (예: 브로드캐스트)
- 텐서→하드웨어는 함수가 아닐 수 있지만, 하드웨어→텐서는 항상 함수입니다
- 함수로 표현하면 합성(composition), 역변환(inversion) 등의 수학적 조작이 깔끔해집니다

### 1.2 왜 Linear Layout이 필요한가?

**Triton의 문제:**
- 기존에는 `BlockedEncodingAttr`, `SliceEncodingAttr`, `MmaEncodingAttr` 등 다양한 특화된 레이아웃 클래스들이 존재
- 각 레이아웃 간 변환, 주소 계산 등이 각각 별도의 특수 케이스 로직으로 구현됨
- 새로운 하드웨어(Hopper, AMD 등)나 레이아웃 추가 시 조합 폭발

**Linear Layout의 해결책:**
- 모든 레이아웃을 하나의 통일된 표현(기저 벡터들)으로 변환
- 레이아웃 합성, 역변환, 재배치 등을 일반적인 선형대수 연산으로 표준화
- 새로운 레이아웃은 기저만 정의하면 기존 인프라 재사용 가능

**실제 효과:**
```cpp
// 기존: 레이아웃 타입별 특수 케이스
if (auto blocked = ...) { /* blocked 특화 로직 */ }
else if (auto mma = ...) { /* mma 특화 로직 */ }
// ... 수십 개의 케이스

// Linear Layout 사용
auto layout = toLinearLayout(anyLayoutType);  // 통일된 표현으로 변환
auto result = layout.compose(otherLayout);     // 일반적인 연산
```

### 1.3 핵심 특징

1. **희소 표현 (Basis vectors)**: 2^N개 입력 중 log(N)개 기저만 저장
2. **선형성 (XOR rule)**: 모든 값은 기저들의 XOR 조합으로 계산
3. **합성 가능 (Composable)**: 레이아웃끼리 곱셈, 합성, 역변환 가능
4. **차원 명명 (Named dimensions)**: 입력/출력 차원에 의미 있는 이름 부여

---

## 2. 수학적 기초: GF(2) 선형 함수

### 2.1 GF(2) 필드란?

**GF(2)는 비트 연산의 수학적 구조입니다:**
- **원소**: `{0, 1}` 두 개
- **덧셈**: XOR (`⊕`)
  - `0 ⊕ 0 = 0`, `0 ⊕ 1 = 1`, `1 ⊕ 0 = 1`, `1 ⊕ 1 = 0`
- **곱셈**: AND (`×`)
  - `0 × 0 = 0`, `0 × 1 = 0`, `1 × 0 = 0`, `1 × 1 = 1`

### 2.2 선형 함수의 정의

일반적인 선형 함수는 다음과 같이 표현됩니다:

```
L(a) = a₁·B₁ + a₂·B₂ + ... + aₘ·Bₘ
```

여기서:
- `a = [a₁, a₂, ..., aₘ]`는 입력 벡터
- `Bᵢ`는 기저 벡터 (basis vector)
- 일반 선형대수에서는 실수 덧셈/곱셈 사용

**GF(2) 선형 함수는 XOR와 AND를 사용합니다:**

```
L(a) = (a₁ × B₁) ⊕ (a₂ × B₂) ⊕ ... ⊕ (aₘ × Bₘ)
```

### 2.3 구체적 예시: 4×4 행렬

**행렬-벡터 곱셈 예시 (GF(2)):**

```
    | 1 0 0 0 |   | 0 |
    | 0 1 1 0 | × | 1 |
    | 0 0 1 1 |   | 1 |
    | 0 0 1 1 |   | 0 |
```

**단계별 계산:**
```
= (열1 × 0) ⊕ (열2 × 1) ⊕ (열3 × 1) ⊕ (열4 × 0)

= |0|     |1|     |0|     |0|
  |0|  ⊕  |1|  ⊕  |1|  ⊕  |0|
  |0|     |0|     |1|     |0|
  |0|     |0|     |1|     |0|

= |1|     |0|
  |1|  ⊕  |1|
  |0|     |1|
  |0|     |1|

= |0|
  |0|
  |1|
  |1|
```

### 2.4 정수로 표현하기

비트 벡터를 정수로 압축하면 더 간결합니다:

```
행렬:  | 1  2  14  12 |  (각 열을 비트로 해석한 정수)
입력:  6  (= 0b0110)

계산:
= (1 × 0) ⊕ (2 × 1) ⊕ (14 × 1) ⊕ (12 × 0)
= 0 ⊕ 2 ⊕ 14 ⊕ 0
= 2 ⊕ 14
= 0b0010 ⊕ 0b1110
= 0b1100
= 12
```

**핵심 통찰:**
- 입력 `a = 6 = 0b0110`에서 설정된 비트는 2번, 3번 위치
- 2번 기저(`2`)와 3번 기저(`14`)를 XOR하면 결과가 나옵니다
- 이것이 바로 Linear Layout의 작동 원리입니다!

---

## 3. 기저(Basis)의 개념과 선형성 규칙

### 3.1 기저 벡터란?

**기저는 2의 거듭제곱 입력에서의 함수 값입니다.**

**1D 예시:** 8-요소 레이아웃 `L(x)`
- 기저 = `[L(1), L(2), L(4)]`
- 이 3개 값만으로 `L(0)` ~ `L(7)` 모두 계산 가능

**2D 예시:** 4×4 레이아웃 `L(x,y)`
- x 방향 기저 = `[L(1,0), L(2,0)]`
- y 방향 기저 = `[L(0,1), L(0,2)]`
- 총 4개 기저로 16개 출력 모두 계산

### 3.2 선형성 규칙 (Linearity Rule)

**핵심 공식:**
```
L(a ⊕ b) = L(a) ⊕ L(b)
```

**다차원 확장:**
```
L(x₁ ⊕ x₂, y₁ ⊕ y₂) = L(x₁, y₁) ⊕ L(x₂, y₂)
```

### 3.3 구체적 예제: Swizzled Layout

**설정:**
- 4개 쓰레드 (thread 0~3), 4개 워프 (warp 0~3)
- 4×4 텐서
- 스위즐 패턴: `L(t,w) = (t, w⊕t)`

**기저 정의:**
```
L(0, 1) = (0, 1)  ← warp 기저 1
L(0, 2) = (0, 2)  ← warp 기저 2
L(1, 0) = (1, 1)  ← thread 기저 1 (스위즐!)
L(2, 0) = (2, 2)  ← thread 기저 2 (스위즐!)
```

**전체 테이블 계산:**
```
L(0,0) = L(1⊕1, 0⊕0) = L(1,0) ⊕ L(1,0) = (1,1) ⊕ (1,1) = (0,0)
L(1,1) = L(1, 0⊕1) = L(1,0) ⊕ L(0,1) = (1,1) ⊕ (0,1) = (1,0)
L(2,1) = L(2, 0⊕1) = L(2,0) ⊕ L(0,1) = (2,2) ⊕ (0,1) = (2,3)
L(3,3) = L(1⊕2, 1⊕2) = L(1,1) ⊕ L(2,2) = ...
```

**결과 매핑 테이블:**
```
         warp→  0     1     2     3
thread↓
  0          (0,0) (0,1) (0,2) (0,3)
  1          (1,1) (1,0) (1,3) (1,2)
  2          (2,2) (2,3) (2,0) (2,1)
  3          (3,3) (3,2) (3,1) (3,0)
```

이것이 전형적인 **swizzled pattern**: `(t, w⊕t)`

### 3.4 코드로 표현하기

```cpp
MLIRContext *ctx = ...;
StringAttr kThread = StringAttr::get(ctx, "thread");
StringAttr kWarp = StringAttr::get(ctx, "warp");
StringAttr kDim0 = StringAttr::get(ctx, "dim0");
StringAttr kDim1 = StringAttr::get(ctx, "dim1");

// 기저 정의
LinearLayout swizzled({
  {kThread, {
    {1, 1},  // L(thread=1) = (dim0=1, dim1=1)
    {2, 2}   // L(thread=2) = (dim0=2, dim1=2)
  }},
  {kWarp, {
    {0, 1},  // L(warp=1) = (dim0=0, dim1=1)
    {0, 2}   // L(warp=2) = (dim0=0, dim1=2)
  }}
}, {kDim0, kDim1});

// 사용 예
auto result = swizzled.apply({{kThread, 3}, {kWarp, 2}});
// result = {{kDim0, 3}, {kDim1, 1}}
```

---

## 4. 차원 명명 체계와 의미

### 4.1 입력 차원 (Input Dimensions)

입력 차원은 **하드웨어 계층 구조**를 나타냅니다.

#### 레지스터 분산 레이아웃 (Distributed Layout)
```
"register" → 한 쓰레드가 보유한 여러 레지스터 요소의 인덱스
"lane"     → 워프 내 쓰레드 ID (0~31 for NVIDIA, 0~63 for AMD)
"warp"     → CTA 내 워프 ID
"block"    → 클러스터 내 블록(CTA) ID
```

**예시: BlockedEncodingAttr 변환**
```cpp
// sizePerThread=[4,2], threadsPerWarp=[8,4], warpsPerCTA=[2,2]
auto layout = BlockedEncodingAttr::get(...);
auto ll = layout.toLinearLayout(shape);

// ll은 다음 입력 차원을 가짐:
// - register: 0~7 (4×2-1)
// - lane: 0~31 (8×4-1)
// - warp: 0~3 (2×2-1)
// - block: ...
```

#### Shared 메모리 레이아웃
```
"offset" → Shared 메모리 내 선형 오프셋 (요소 단위)
"block"  → 블록 ID (multi-CTA 경우)
```

**예시: SwizzledSharedEncodingAttr**
```cpp
auto shared = SwizzledSharedEncodingAttr::get(
  ctx, vec=8, perPhase=4, maxPhase=8, order={1,0}, ...);
auto ll = swizzledSharedToLinearLayout(shape, shared);

// ll의 입력: offset, block
// ll의 출력: dim0, dim1
// offset → (dim0, dim1) 매핑이 swizzle pattern을 표현
```

### 4.2 출력 차원 (Output Dimensions)

출력 차원은 **논리 텐서의 축**을 나타냅니다.

```
"dim0"  → 텐서의 첫 번째 차원 (order와 무관하게 원본 shape 기준)
"dim1"  → 텐서의 두 번째 차원
"dim2"  → 텐서의 세 번째 차원
...
```

**중요:** 출력 차원은 레이아웃의 `order` 필드와 독립적입니다. `order`는 메모리 배치 순서를 결정하지만, LinearLayout의 출력 차원 이름은 항상 표준 순서(`dim0, dim1, ...`)를 사용합니다.

### 4.3 차원 순서와 Reshape

차원들은 **minor-to-major 순서**를 가집니다:
- 첫 번째 차원이 가장 minor (메모리에서 가장 빠르게 변함)
- 마지막 차원이 가장 major

**reshape 시 중요:**
```cpp
// 입력 차원: ["register", "lane"] (이 순서)
// flatten 시: register가 minor, lane이 major
auto flattened = layout.flattenIns();
// 결과: 단일 차원, 크기 = register_size × lane_size
```

---

## 5. 핵심 API 상세 가이드

### 5.1 기본 생성자들

#### 5.1.1 `identity1D` - 항등 함수

```cpp
static LinearLayout identity1D(
  int32_t size,        // 입력 크기 (2의 거듭제곱)
  StringAttr inDim,    // 입력 차원 이름
  StringAttr outDim    // 출력 차원 이름
);
```

**의미:** `L(x) = x` for `x ∈ [0, size)`

**예시:**
```cpp
auto L = LinearLayout::identity1D(8, S("lane"), S("dim0"));
// L(0) = 0, L(1) = 1, ..., L(7) = 7
// 기저: [L(1)=1, L(2)=2, L(4)=4]
```

**언제 사용?**
- 연속적인 인덱스 매핑
- 다른 레이아웃의 기본 빌딩 블록

#### 5.1.2 `strided1D` - 등간격 stride

```cpp
static LinearLayout strided1D(
  int32_t size,
  int32_t stride,
  StringAttr inDim,
  StringAttr outDim
);
```

**의미:** `L(x) = stride × x` for `x ∈ [0, size)`

**예시:**
```cpp
auto L = LinearLayout::strided1D(4, 2, S("lane"), S("dim0"));
// L(0) = 0, L(1) = 2, L(2) = 4, L(3) = 6
// 기저: [L(1)=2, L(2)=4]
```

**주의:** stride는 2의 거듭제곱이어야 합니다 (GF(2) 연산 특성).

#### 5.1.3 `zeros1D` - 브로드캐스트

```cpp
static LinearLayout zeros1D(
  int32_t size,
  StringAttr inDim,
  StringAttr outDim,
  int32_t outDimSize = 1
);
```

**의미:** `L(x) = 0` for all `x` (브로드캐스팅)

**예시:**
```cpp
auto L = LinearLayout::zeros1D(8, S("lane"), S("dim1"));
// L(0) = L(1) = ... = L(7) = 0
// 기저: [0, 0, 0]  (모든 기저가 0)
```

**사용 예:** 한 축은 모든 쓰레드가 같은 값을 읽는 브로드캐스트 패턴

### 5.2 기저로부터 직접 생성

```cpp
LinearLayout(
  BasesT bases,                           // 입력 차원 → 기저 벡터 맵
  ArrayRef<StringAttr> outDimNames        // 출력 차원 이름들
);
```

**예시: 2D Swizzle**
```cpp
LinearLayout swizzle({
  {S("offset"), {
    {0, 1},  {0, 2},  {0, 4},  {0, 8},   // col 기저
    {1, 0},  {2, 0},  {4, 4},  {8, 8}    // row 기저 (swizzled)
  }}
}, {S("dim0"), S("dim1")});
```

**기저 형식:**
- `bases[inDim][i]`는 `L(inDim=2^i, other=0)`의 값
- `bases[inDim][i][j]`는 `j`번째 출력 차원의 값

### 5.3 레이아웃 조합: `operator*`

```cpp
friend LinearLayout operator*(LinearLayout inner, LinearLayout outer);
```

**의미:** Direct sum (직합) - 두 레이아웃을 독립적으로 결합

**규칙:**
1. 입력 차원이 겹치지 않으면: 두 입력 공간의 직합
2. 출력 차원이 겹치면: 동일 출력에 함께 기여
3. 순서: `inner` 차원이 더 minor

**예시 1: 2D 항등 만들기**
```cpp
auto L = LinearLayout::identity1D(4, S("lane"), S("dim1"))
       * LinearLayout::identity1D(8, S("register"), S("dim0"));

// 입력: (register, lane)
// 출력: (dim0, dim1)
// L(reg=3, lane=2) = (dim0=3, dim1=2)
```

**예시 2: 출력 차원 공유**
```cpp
auto L1 = LinearLayout::identity1D(4, S("lane"), S("dim0"));
auto L2 = LinearLayout::identity1D(8, S("register"), S("dim0"));
auto L = L1 * L2;

// L(lane=2, register=3) = (dim0 = 2 ⊕ 3 = 1)
// 두 입력이 같은 출력 차원에 XOR로 결합됨
```

**예시 3: 브로드캐스트와 결합**
```cpp
auto L = LinearLayout::zeros1D(4, S("lane"), S("dim1"))
       * LinearLayout::identity1D(8, S("register"), S("dim0"));

// L(lane=?, register=5) = (dim0=5, dim1=0)
// lane 값과 무관하게 dim1은 항상 0
```

### 5.4 함수 합성: `compose`

```cpp
LinearLayout compose(const LinearLayout &outer) const;
```

**의미:** `(outer ∘ this)(x) = outer(this(x))`

**요구사항:**
- `this`의 출력 차원 = `outer`의 입력 차원
- `this->getOutDimSize(d) ≤ outer.getInDimSize(d)`

**예시:**
```cpp
// L1: (register) → (offset)
auto L1 = LinearLayout::identity1D(32, S("register"), S("offset"));

// L2: (offset) → (dim0, dim1)
auto L2 = /* some swizzled layout */;

// L3: (register) → (dim0, dim1)
auto L3 = L1.compose(L2);
```

### 5.5 역변환과 합성: `invertAndCompose`

```cpp
LinearLayout invertAndCompose(const LinearLayout &outer) const;
```

**의미:** `C(x)`를 계산하여 `this(x) = outer(C(x))` 만족

**핵심 사용 사례: 레지스터 → Shared 메모리 저장**
```cpp
// regLayout: (register,lane,warp) → (dim0,dim1)
// memLayout: (offset,block) → (dim0,dim1)

auto cvt = regLayout.invertAndCompose(memLayout);
// cvt: (register,lane,warp) → (offset,block)

// 의미: 레지스터[r,l,w]의 값을 어느 shared offset에 저장해야 하는가?
```

**요구사항:**
- `outer`는 surjective (모든 출력 커버)
- `outer`의 codomain ≥ `this`의 codomain
- `outer`가 non-injective여도 OK (가장 작은 해 선택)

**예시 시나리오:**
```cpp
// 1) 레지스터 분산: thread 0의 reg 0이 tensor[2,3]을 보유
auto regLayout = toLinearLayout(blockedEncoding);
assert(regLayout.apply({{S("register"),0}, {S("lane"),0}, {S("warp"),0}})
       == {{S("dim0"),2}, {S("dim1"),3}});

// 2) Shared 메모리: offset 10이 tensor[2,3] 위치
auto memLayout = toLinearLayout(sharedEncoding);
assert(memLayout.apply({{S("offset"),10}})
       == {{S("dim0"),2}, {S("dim1"),3}});

// 3) 변환: thread 0, reg 0은 offset 10에 저장해야 함
auto cvt = regLayout.invertAndCompose(memLayout);
assert(cvt.apply({{S("register"),0}, {S("lane"),0}, {S("warp"),0}})
       == {{S("offset"),10}});
```

### 5.6 형태 변환

#### `flattenIns/Outs`
```cpp
LinearLayout flattenIns() const;
LinearLayout flattenOuts() const;
```

모든 입력/출력 차원을 단일 차원으로 병합.

```cpp
// 입력: (register:4, lane:8, warp:2)
auto flat = layout.flattenIns();
// 출력: (register:64)  // 4×8×2, minor-to-major 순서
```

#### `reshapeIns/Outs`
```cpp
LinearLayout reshapeIns(
  ArrayRef<std::pair<StringAttr, int32_t>> newInDims
) const;
```

차원을 재구성. 먼저 flatten 후 unstack.

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

차원 순서를 재정렬 (주로 reshape 전에 사용).

```cpp
auto transposed = layout.transposeOuts({S("dim1"), S("dim0")});
```

### 5.7 적용 (Apply)

#### 정수 적용
```cpp
SmallVector<std::pair<StringAttr, int32_t>>
apply(ArrayRef<std::pair<StringAttr, int32_t>> ins) const;
```

```cpp
auto result = layout.apply({
  {S("register"), 3},
  {S("lane"), 5},
  {S("warp"), 1}
});
// result: {{S("dim0"), ...}, {S("dim1"), ...}}
```

#### MLIR Value 적용
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

Lowering 시 실제 주소 계산에 사용:
```cpp
Value regId = b.i32_val(0);
Value laneId = getLaneId(rewriter, loc);
Value warpId = getWarpId(rewriter, loc);

auto offsets = applyLinearLayout(loc, rewriter, cvt, {
  {S("register"), regId},
  {S("lane"), laneId},
  {S("warp"), warpId}
});
// offsets[0].second는 shared memory offset (Value)
```

---

## 6. 실전 예제: 단계별 구성

### 6.1 예제 1: 간단한 1D 분산

**시나리오:** 32개 요소를 4개 쓰레드에 분산, 각 쓰레드가 8개 보유

```cpp
// 각 쓰레드: 8개 레지스터
// 쓰레드 0: 요소 [0,4,8,12,16,20,24,28]
// 쓰레드 1: 요소 [1,5,9,13,17,21,25,29]
// 쓰레드 2: 요소 [2,6,10,14,18,22,26,30]
// 쓰레드 3: 요소 [3,7,11,15,19,23,27,31]

auto layout =
  LinearLayout::identity1D(8, S("register"), S("dim0")) *
  LinearLayout::strided1D(4, 1, S("lane"), S("dim0"));

// 검증
assert(layout.apply({{S("register"),0}, {S("lane"),0}}) == {{S("dim0"),0}});
assert(layout.apply({{S("register"),1}, {S("lane"),0}}) == {{S("dim0"),4}});
assert(layout.apply({{S("register"),0}, {S("lane"),1}}) == {{S("dim0"),1}});
assert(layout.apply({{S("register"),2}, {S("lane"),3}}) == {{S("dim0"),11}});
```

**설명:**
- `register`: stride=4 (기저: [4,8,16,...])
- `lane`: stride=1 (기저: [1,2])
- XOR 결합: `dim0 = register_contribution ⊕ lane_contribution`

### 6.2 예제 2: 2D Blocked Layout

**시나리오:**
- 16×16 텐서
- sizePerThread = [2, 2]: 각 쓰레드 4개 요소
- threadsPerWarp = [4, 4]: 워프당 16 쓰레드
- warpsPerCTA = [2, 2]: CTA당 4 워프

```cpp
// 1) Register 레이아웃: 2×2 블록
auto regLayout =
  LinearLayout::identity1D(2, S("register"), S("dim0")) *
  LinearLayout::identity1D(2, S("register"), S("dim1"));

// 2) Lane 레이아웃: 4×4 블록
auto laneLayout =
  LinearLayout::strided1D(4, 2, S("lane"), S("dim0")) *  // stride=2
  LinearLayout::strided1D(4, 2, S("lane"), S("dim1"));   // stride=2

// 3) Warp 레이아웃: 2×2 블록
auto warpLayout =
  LinearLayout::strided1D(2, 8, S("warp"), S("dim0")) *  // stride=8
  LinearLayout::strided1D(2, 8, S("warp"), S("dim1"));   // stride=8

// 4) 결합
auto layout = regLayout * laneLayout * warpLayout;

// 검증: lane=5(=0b0101), register=2(=0b10)
// dim0: reg_contrib=0, lane_contrib=2×1=2, warp_contrib=0 → 2
// dim1: reg_contrib=2, lane_contrib=2×0=0, warp_contrib=0 → 2
assert(layout.apply({
  {S("register"), 2},
  {S("lane"), 5},
  {S("warp"), 0}
}) == {{S("dim0"), 2}, {S("dim1"), 2}});
```

**BlockedEncodingAttr 자동 변환:**
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
// 위의 수동 구성과 동일한 결과
```

### 6.3 예제 3: Shared Memory Swizzle

**시나리오:** 128×32 shared 메모리, 128-byte swizzle

```cpp
int numRows = 128;
int numCols = 32;  // FP32 기준
int vec = 8;       // 8 요소 = 32 bytes
int perPhase = 4;  // 4 rows per phase
int maxPhase = 8;  // 8 phases

// 기저 구성
std::vector<std::vector<int>> bases2D;

// Col 기저 (swizzle 없음)
for (int col = 1; col < numCols; col *= 2) {
  bases2D.push_back({0, col});
}

// Row 기저 (swizzle 적용)
for (int row = 1; row < numRows; row *= 2) {
  int phase = (row / perPhase) % maxPhase;
  int colSwizzle = vec * phase;
  bases2D.push_back({row, colSwizzle % numCols});
}

LinearLayout swizzled({
  {S("offset"), bases2D}
}, {S("dim0"), S("dim1")});

// 예: offset=17 (row=4, col=1)
// row 기저: row=4 → phase=4/4%8=1, swizzle=8×1=8
// L(offset=17) = L(16) ⊕ L(1)
//              = (4, 8) ⊕ (0, 1)
//              = (4, 9)
```

### 6.4 예제 4: Register → Shared 변환

**완전한 워크플로우:**

```cpp
// 1) 레지스터 레이아웃 정의
auto regLayout = BlockedEncodingAttr::get(...).toLinearLayout(shape);
// 입력: (register, lane, warp, block)
// 출력: (dim0, dim1)

// 2) Shared 메모리 레이아웃 정의
auto memLayout = swizzledSharedToLinearLayout(shape, sharedAttr);
// 입력: (offset, block)
// 출력: (dim0, dim1)

// 3) 변환 레이아웃 계산
auto cvtLayout = regLayout.invertAndCompose(memLayout);
// 입력: (register, lane, warp, block)
// 출력: (offset, block)

// 4) LLVM lowering에서 주소 계산
auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
Value blockId = getBlockId(rewriter, loc);

for (int regId = 0; regId < numRegs; regId++) {
  auto offsetPairs = applyLinearLayout(loc, rewriter, cvtLayout, {
    {S("register"), b.i32_val(regId)},
    {S("lane"), laneId},
    {S("warp"), warpId},
    {S("block"), blockId}
  });

  Value offset = offsetPairs[0].second;  // offset 값
  Value ptr = gep(smemBase, offset);
  store(registerValues[regId], ptr);
}
```

---

## 7. Triton 레이아웃 변환

### 7.1 변환 엔트리포인트

```cpp
// include/triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h

LinearLayout toLinearLayout(RankedTensorType type);
LinearLayout toLinearLayout(MemDescType type);
LinearLayout toLinearLayout(TensorOrMemDesc type);
LinearLayout toLinearLayout(ArrayRef<int64_t> shape, Attribute layout);
```

**사용 예:**
```cpp
RankedTensorType tensorTy = ...;
auto layout = toLinearLayout(tensorTy);
```

### 7.2 BlockedEncodingAttr 변환

**구현 위치:** `lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp:926`

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

**`identityStandardND` 함수:**
```cpp
// N차원 항등을 order에 따라 구성
LinearLayout identityStandardND(
  StringAttr inDim,
  ArrayRef<unsigned> shape,
  ArrayRef<unsigned> order
) {
  // order[0]이 가장 minor
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

**예시:**
```cpp
// sizePerThread = [4, 2], order = [1, 0] (row-major)
identityStandardND(S("register"), {4, 2}, {1, 0})
// = identity1D(2, "register", "dim1") * identity1D(4, "register", "dim0")
// register 0~1 → dim1 variation
// register 2~3,4~5,6~7 → dim1 repetition with dim0 변화
```

### 7.3 SwizzledSharedEncodingAttr 변환

**구현 위치:** `lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp:85`

```cpp
LinearLayout swizzledSharedToLinearLayout(
  ArrayRef<int64_t> shape,
  SwizzledSharedEncodingAttr shared
) {
  // ... rank=1 특수 케이스 ...

  // 2D+ 케이스: 최하위 2개 차원에 swizzle 적용
  int colDim = shared.getOrder()[0];
  int rowDim = shared.getOrder()[1];
  int numCols = shapePerCTA[colDim];
  int numRows = shapePerCTA[rowDim];

  std::vector<std::vector<int>> bases2D;

  // Column 기저: 단순 항등
  for (int col = 1; col < numCols; col *= 2) {
    bases2D.push_back({0, col});
  }

  // Row 기저: swizzle 적용
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

  // 고차원 확장
  for (int i = 2; i < rank; i++) {
    int dim = shared.getOrder()[i];
    ctaLayout *= LinearLayout::identity1D(
      shapePerCTA[dim], S("offset"), outDimNames[dim]
    );
  }

  return combineCtaCgaWithShape(ctaLayout, shared.getCTALayout(), shape);
}
```

**Swizzle 공식 설명:**
```
phase = (row / perPhase) % maxPhase
colSwizzle = (vec * phase) % numCols
actualCol = baseCol ⊕ colSwizzle
```

**뱅크 충돌 회피 원리:**
- Shared 메모리는 32개 뱅크 (NVIDIA) 또는 64개 뱅크 (AMD)
- 같은 row의 연속된 요소들이 다른 뱅크에 분산되도록 swizzle
- `vec` 크기의 그룹이 phase마다 회전

### 7.4 NVMMASharedEncodingAttr 변환 (Hopper)

**핵심 함수:** `getCoreMatrixLinearLayout`

**구현 위치:** `lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp:189`

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

  // Column 기저
  for (int col = 1; col < tileCols; col *= 2) {
    if (isFp4Padded) {
      // FP4: 16개 offset 중 8개만 "real", 나머지는 padding
      // Packed representation: 16 → 8
      int colPacked = col / 16 * 8 + col % 8;
      bases2D.push_back({0, colPacked});
    } else {
      bases2D.push_back({0, col});
    }
  }

  // Row 기저
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

**FP4 Padding 설명:**
- FP4 (4-bit float): 2개 값이 1 byte
- NVMMA는 8-byte 단위로 처리
- 16개 FP4 값 (8 bytes) 중 실제로는 8개만 사용, 나머지는 padding
- LinearLayout은 16개 offset을 8개 실제 위치로 "folding"
- `invertAndCompose` 시 작은 offset이 자동 선택됨

### 7.5 CTA와 CGA 결합

**CTA (Cooperative Thread Array):** 단일 블록 내 레이아웃
**CGA (Cooperative Grid Array):** 여러 블록 간 분산

```cpp
LinearLayout combineCtaCgaWithShape(
  LinearLayout ctaLayout,
  CTALayoutAttr cgaLayoutAttr,
  ArrayRef<int64_t> shape
) {
  // 1) CGA 레이아웃 구성
  auto cgaLayout = makeCgaLayout(cgaLayoutAttr);

  // 2) CTA와 CGA 곱셈
  auto layout = ctaLayout * cgaLayout;

  // 3) Shape에 맞춰 확장
  layout = ensureLayoutNotSmallerThan(layout, outDims, shape);

  return layout;
}
```

**`makeCgaLayout` 구현:**
```cpp
LinearLayout makeCgaLayout(CTALayoutAttr layout) {
  int rank = layout.getCTAOrder().size();
  LinearLayout ret = LinearLayout::empty();

  for (int i = 0; i < rank; i++) {
    int dim = layout.getCTAOrder()[i];
    int split = layout.getCTASplitNum()[dim];
    int ctas = layout.getCTAsPerCGA()[dim];

    // split 개는 실제 분산, 나머지는 복제 (zeros)
    ret *= LinearLayout::identity1D(split, S("block"), S("dim"+dim)) *
           LinearLayout::zeros1D(ctas/split, S("block"), S("dim"+dim));
  }

  return ret.transposeOuts(standardOutDimNames);
}
```

**예시:**
```cpp
// CTAsPerCGA = [2, 4], CTASplitNum = [2, 2], CTAOrder = [1, 0]
//
// dim0: split=2, total=2 → 완전 분산
//   block 0 → dim0 위치 0
//   block 1 → dim0 위치 1
//
// dim1: split=2, total=4 → 부분 분산 + 복제
//   block 0,2 → dim1 위치 0
//   block 1,3 → dim1 위치 1
```

---

## 8. Lowering에서의 활용

### 8.1 전형적인 변환 패턴

**시나리오:** `LocalStoreOp` - 레지스터에서 Shared 메모리로 저장

```cpp
LogicalResult LocalStoreOpConversion::matchAndRewrite(
  triton::gpu::LocalStoreOp op,
  OpAdaptor adaptor,
  ConversionPatternRewriter &rewriter
) const {
  auto loc = op.getLoc();
  Value src = adaptor.getSrc();  // 레지스터 값들
  Value dst = adaptor.getDst();  // Shared 메모리 descriptor

  // 1) 레이아웃 가져오기
  auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
  auto dstTy = cast<MemDescType>(op.getDst().getType());

  auto srcLayout = toLinearLayout(srcTy);  // (register,lane,warp,block)→(dim0,dim1)
  auto dstLayout = toLinearLayout(dstTy);  // (offset,block)→(dim0,dim1)

  // 2) 변환 레이아웃 계산
  auto cvtLayout = srcLayout.invertAndCompose(dstLayout);
  // (register,lane,warp,block) → (offset,block)

  // 3) 하드웨어 ID 가져오기
  auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
  Value blockId = getBlockId(rewriter, loc);

  // 4) Shared 메모리 베이스 주소
  Value smemBase = getSharedMemoryBase(dst);

  // 5) 각 레지스터에 대해 저장
  auto srcVals = unpackLLElements(loc, src, rewriter);
  int numRegs = srcVals.size();

  for (int regId = 0; regId < numRegs; regId++) {
    // 5a) 오프셋 계산
    auto offsetPairs = applyLinearLayout(loc, rewriter, cvtLayout, {
      {S("register"), b.i32_val(regId)},
      {S("lane"), laneId},
      {S("warp"), warpId},
      {S("block"), blockId}
    });

    Value offset = offsetPairs[0].second;

    // 5b) 포인터 계산 및 저장
    Value ptr = gep(ptr_ty(ctx, 3), smemBase, offset);
    store(srcVals[regId], ptr);
  }

  rewriter.eraseOp(op);
  return success();
}
```

### 8.2 `applyLinearLayout` 구현 원리

**위치:** `lib/Conversion/TritonGPUToLLVM/Utility.cpp:237`

```cpp
SmallVector<std::pair<StringAttr, Value>>
applyLinearLayout(
  Location loc,
  RewriterBase &rewriter,
  const LinearLayout &layout,
  ArrayRef<std::pair<StringAttr, Value>> indices
) {
  // 1) 각 출력 차원에 대해 결과 초기화 (0으로)
  SmallVector<Value> results(layout.getNumOutDims());
  for (auto &r : results) r = i32_val(0);

  // 2) 각 입력 차원 처리
  for (auto [inDim, inVal] : indices) {
    int inDimSizeLog2 = layout.getInDimSizeLog2(inDim);

    // 3) 각 기저에 대해
    for (int bit = 0; bit < inDimSizeLog2; bit++) {
      // 3a) inVal의 bit번째 비트 추출
      Value bitVal = and_(inVal, i32_val(1 << bit));
      Value bitSet = icmp_ne(bitVal, i32_val(0));

      // 3b) 이 기저의 기여도
      auto basis = layout.getBasis(inDim, bit);

      // 3c) 각 출력 차원에 조건부 XOR
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

  // 4) 출력 차원 이름과 함께 반환
  SmallVector<std::pair<StringAttr, Value>> output;
  int idx = 0;
  for (auto outDim : layout.getOutDimNames()) {
    output.push_back({outDim, results[idx++]});
  }
  return output;
}
```

**최적화 버전:** 실제 구현은 additive stride 검출, 미리 계산 등 최적화 포함.

### 8.3 MMA Operand 변환 예시

**문제:** Dot product operand를 MMA-friendly shared layout으로 변환

```cpp
// AccelerateMatmul.cpp
Value getSharedMemoryMMAOperand(
  Value tensor,
  const SharedEncodingAttr &sharedLayout,
  Location loc,
  RewriterBase &rewriter
) {
  // 1) 소스 레이아웃
  auto srcTy = cast<RankedTensorType>(tensor.getType());
  auto srcLayout = toLinearLayout(srcTy);

  // 2) 목표 shared 레이아웃
  auto dstTy = MemDescType::get(..., sharedLayout);
  auto dstLayout = toLinearLayout(dstTy);

  // 3) 변환이 bank conflict 없는지 검증
  if (!isSwizzleOptimal(srcLayout, dstLayout)) {
    // Swizzle 파라미터 조정...
  }

  // 4) Shared 메모리 할당
  Value smem = allocSharedMemory(dstTy);

  // 5) 레지스터 → Shared 저장
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

## 9. 고급 주제

### 9.1 Bank Conflict 분석

**Shared 메모리 뱅크:**
- NVIDIA: 32개 뱅크, 4-byte 단위
- AMD: 64개 뱅크, 4-byte 단위

**충돌 조건:** 동일 워프의 여러 쓰레드가 같은 뱅크의 다른 주소 접근

**LinearLayout으로 검증:**
```cpp
bool hasNoBankConflicts(const LinearLayout &layout) {
  // 1) Layout을 (lane, offset) → (dim0, dim1)으로 제한
  auto laneToTensor = layout.sublayout({S("lane")}, {...});

  // 2) 역변환: (dim0, dim1) → lane
  auto tensorToLane = laneToTensor.invert();

  // 3) Shared offset → bank 매핑
  // bank = (offset % numElements) % numBanks
  auto offsetToBank = LinearLayout::identity1D(
    numElements, S("offset"), S("bank")
  ).reshapeOuts({{S("bank"), numBanks}});

  // 4) (dim0,dim1) → offset (shared layout)
  auto tensorToOffset = sharedLayout.invert();

  // 5) 합성: (dim0,dim1) → lane, (dim0,dim1) → offset → bank
  auto tensorToLaneBank = tensorToLane * (tensorToOffset.compose(offsetToBank));

  // 6) Lane → bank 매핑이 단사(injective)인지 확인
  // 즉, 다른 lane은 다른 bank에 접근하는지
  return tensorToLaneBank.sublayout({S("lane")}, {S("bank")}).isInjective();
}
```

### 9.2 `divideLeft`와 `divideRight`

**목적:** 레이아웃 분해 - `A = B * C`에서 C 찾기

```cpp
// divideLeft: A = B * C를 만족하는 C 계산
std::optional<LinearLayout> C = divideLeft(A, B);

// divideRight: A = C * B를 만족하는 C 계산
std::optional<LinearLayout> C = divideRight(A, B);
```

**사용 예: Vectorization 추출**
```cpp
// fullLayout: (register, lane) → (dim0)
// 목표: register 차원에서 연속 벡터 찾기

// 1) 연속 4개 요소 패턴
auto vec4Pattern = LinearLayout::identity1D(4, S("register"), S("dim0"));

// 2) fullLayout을 vec4Pattern으로 나누기
auto quotient = divideLeft(fullLayout, vec4Pattern);

if (quotient.has_value()) {
  // fullLayout = vec4Pattern * quotient
  // 즉, register 4개씩 묶어서 vectorize 가능
  int numVec4Groups = quotient->getTotalInDimSize();
  // ...
}
```

### 9.3 `getFreeVariableMasks`

**목적:** 출력에 영향을 주지 않는 입력 비트 찾기

```cpp
auto masks = layout.getFreeVariableMasks();
// masks[inDim] = 출력에 무관한 비트들의 마스크
```

**사용 예: Broadcast 차원 검출**
```cpp
auto layout = LinearLayout::zeros1D(8, S("lane"), S("dim0")) *
              LinearLayout::identity1D(4, S("register"), S("dim0"));

auto masks = layout.getFreeVariableMasks();
// masks[S("lane")] = 0b111  (모든 비트가 free)
// masks[S("register")] = 0b000  (모든 비트가 중요)

// lane은 브로드캐스트 차원임을 알 수 있음
```

### 9.4 `ColumnAction`: 기저 재배열

**목적:** 특정 입력 차원의 기저 열을 재배치

```cpp
// 레지스터 값 순서를 재배치하여 메모리 접근 패턴 최적화
ColumnAction action({2, 0, 1}, S("register"), /*inSizeLog2=*/3);

// Layout의 기저 재배치
auto newLayout = action.apply(oldLayout);

// Value들도 동일하게 재배치
auto newValues = action.apply(oldValues);
```

**예시: Transpose**
```cpp
// 원본: register [0,1,2,3,4,5,6,7] → (row,col)
// 목표: (col,row) 순서로 재배치

// 1) 필요한 재배치 패턴 계산
auto action = computeTransposeAction(layout);

// 2) 레이아웃과 값 모두 재배치
auto transposedLayout = action.apply(layout);
auto transposedValues = action.apply(registerValues);
```

---

## 10. 디버깅과 문제 해결

### 10.1 `toString()`으로 레이아웃 확인

```cpp
auto layout = toLinearLayout(tensorTy);
llvm::errs() << "Layout: " << layout.toString() << "\n";
```

**출력 예:**
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

### 10.2 Apply로 스팟 체크

```cpp
// 특정 입력의 출력 확인
auto result = layout.apply({
  {S("register"), 5},
  {S("lane"), 10},
  {S("warp"), 1}
});

for (auto [dim, val] : result) {
  llvm::errs() << dim.str() << " = " << val << "\n";
}
```

### 10.3 자주 발생하는 에러와 해결

#### 에러 1: "Layout is not surjective"

**원인:** 기저가 출력 공간을 완전히 커버하지 못함

**해결:**
```cpp
// 잘못된 예
LinearLayout bad({
  {S("lane"), {{1}, {2}}}  // 기저: [1, 2]
}, {S("dim0")});
// 출력 크기를 4로 추론하지만, 실제로는 0,1,2,3 모두 생성 불가

// 올바른 예 1: 출력 크기 명시
LinearLayout good1({
  {S("lane"), {{1}, {2}}}
}, {{S("dim0"), 4}}, /*requireSurjective=*/false);

// 올바른 예 2: 완전한 기저 제공
LinearLayout good2({
  {S("lane"), {{1}, {2}, {4}}}
}, {S("dim0")});  // [1,2,4]로 0~7 모두 생성 가능
```

#### 에러 2: "Shape must be a power of 2"

**원인:** 입력 shape가 2의 거듭제곱이 아님

**해결:**
```cpp
// 잘못된 예
auto layout = toLinearLayout({12, 20}, blockedAttr);  // 12, 20은 비-P2

// 올바른 예: Shape을 P2로 패딩
auto paddedShape = {16, 32};
auto layout = toLinearLayout(paddedShape, blockedAttr);

// 실제 shape는 별도로 추적
auto actualShape = {12, 20};
```

#### 에러 3: "Dimension mismatch in compose"

**원인:** `compose` 시 출력/입력 차원 불일치

**해결:**
```cpp
// L1: (register) → (offset)
// L2: (addr) → (dim0)  // 입력이 "addr"인데 L1 출력은 "offset"

// 해결 1: L1 출력 차원 이름 변경
auto L1fixed = L1.reshapeOuts({{S("addr"), L1.getTotalOutDimSize()}});
auto composed = L1fixed.compose(L2);  // OK

// 해결 2: L2 입력 차원 이름 변경
auto L2fixed = L2.reshapeIns({{S("offset"), L2.getTotalInDimSize()}});
auto composed = L1.compose(L2fixed);  // OK
```

#### 에러 4: `invertAndCompose` 실패

**원인:** `outer`가 non-surjective이거나 codomain이 작음

**디버깅:**
```cpp
auto outer = sharedLayout;

// 1) Surjective 확인
if (!outer.isSurjective()) {
  llvm::errs() << "Outer layout is not surjective!\n";
  llvm::errs() << "Total in size: " << outer.getTotalInDimSize() << "\n";
  llvm::errs() << "Total out size: " << outer.getTotalOutDimSize() << "\n";
}

// 2) Codomain 크기 비교
for (auto outDim : outer.getOutDimNames()) {
  int innerSize = innerLayout.getOutDimSize(outDim);
  int outerSize = outer.getOutDimSize(outDim);
  if (outerSize < innerSize) {
    llvm::errs() << "Outer dimension " << outDim
                 << " too small: " << outerSize << " < " << innerSize << "\n";
  }
}
```

### 10.4 단위 테스트 작성

```cpp
// unittest/Dialect/TritonGPU/LinearLayoutConversionsTest.cpp 참고
TEST_F(LinearLayoutTest, MyCustomLayout) {
  auto layout = /* construct layout */;

  // 1) 크기 검증
  EXPECT_EQ(layout.getTotalInDimSize(), 64);
  EXPECT_EQ(layout.getTotalOutDimSize(), 256);

  // 2) Surjectivity 검증
  EXPECT_TRUE(layout.isSurjective());

  // 3) 특정 입력 검증
  auto result = layout.apply({{S("register"), 5}, {S("lane"), 3}});
  EXPECT_EQ(result[0].second, 23);  // dim0 = 23
  EXPECT_EQ(result[1].second, 7);   // dim1 = 7

  // 4) 역변환 검증
  auto inv = layout.invert();
  auto roundTrip = layout.compose(inv);
  EXPECT_TRUE(roundTrip.isTrivialOver({S("dim0"), S("dim1")}));
}
```

---

## 11. FAQ

### Q1: LinearLayout vs CuTe Layout의 차이?

**공통점:**
- 둘 다 범용 프로그래머블 레이아웃
- 특수 케이스 레이아웃들을 통합

**차이점:**

| 항목 | LinearLayout | CuTe |
|------|--------------|------|
| 차원 이름 | 명명된 차원 (예: "register") | 숫자 인덱스 |
| 중첩 | 불가 | 가능 (flatten 가능) |
| 비-P2 크기 | 미지원 | 지원 |
| Swizzle | 레이아웃 자체에 포함 | 별도 적용 |
| 컴파일 타임 | MLIR IR 수준 (느려도 OK) | C++ template (매우 빠름) |
| 자동 탐색 | 가능 (최적 레이아웃 탐색) | 수동 선택 |

### Q2: 왜 하드웨어→텐서 방향인가?

**이유:**
1. **함수성**: 텐서→하드웨어는 일대다(브로드캐스트 시), 하드웨어→텐서는 항상 일대일
2. **합성**: 함수끼리만 깔끔하게 합성 가능
3. **역변환**: `invertAndCompose`가 의사역행렬로 안전하게 동작

**직관:**
- "이 쓰레드가 어떤 데이터를 가지고 있는가?" (하드웨어→텐서) ✓
- "이 데이터를 누가 가지고 있는가?" (텐서→하드웨어) - 여러 답 가능 ✗

### Q3: 비-P2 shape은 어떻게 처리하나?

**현재 제약:** 대부분의 변환은 P2 shape 요구

**해결 방법:**
1. **Padding**: Shape을 P2로 올림
```cpp
int paddedM = nextPowerOf2(actualM);
int paddedN = nextPowerOf2(actualN);
auto layout = toLinearLayout({paddedM, paddedN}, encoding);
```

2. **Zeros 기저**: 일부 위치를 0으로 설정
```cpp
// 예: 64×64 타일이지만 실제로는 60×60만 사용
// 기저 일부를 0으로 설정하여 범위 제한
```

3. **Mask**: 런타임에 유효 범위 체크
```cpp
Value valid = and_(
  icmp_slt(row, i32_val(actualM)),
  icmp_slt(col, i32_val(actualN))
);
Value data = select(valid, loadedValue, zero);
```

### Q4: `compose`와 `invertAndCompose`의 차이?

**`compose`:**
- `A.compose(B)` = B∘A = "A 먼저, B 나중"
- A의 출력 = B의 입력 (이름과 크기 일치 필요)
- 둘 다 알려진 forward 함수

**`invertAndCompose`:**
- `A.invertAndCompose(B)` ≈ B⁻¹∘A
- A와 B의 **출력**이 같은 공간 (둘 다 텐서 인덱스)
- B는 역변환 가능해야 (surjective)
- "A의 입력을 B의 입력으로 변환"

**비유:**
```
compose:
  온도(섭씨) --A--> 온도(화씨) --B--> 온도(켈빈)
  compose = 섭씨를 켈빈으로 직접 변환

invertAndCompose:
  주소A --A--> 도시 이름
  우편번호 --B--> 도시 이름
  invertAndCompose = 주소A를 우편번호로 변환
  (도시 이름을 매개로)
```

### Q5: 비-단사(non-injective) 레이아웃은 언제 쓰나?

**예시: 브로드캐스트**
```cpp
auto layout =
  LinearLayout::identity1D(8, S("register"), S("dim0")) *
  LinearLayout::zeros1D(32, S("lane"), S("dim0"));

// 모든 lane이 같은 dim0 값 접근
// lane=0, reg=3: dim0=3
// lane=5, reg=3: dim0=3
// lane=31, reg=3: dim0=3
```

**예시: 데이터 복제 (Replication)**
```cpp
// 2개 블록이 같은 데이터 보유
auto cgaLayout = makeCgaLayout(CTALayoutAttr::get(
  /*CTAsPerCGA=*/{2, 1},
  /*CTASplitNum=*/{1, 1},  // dim0은 split 없음 = 복제
  ...
));

// block=0과 block=1 모두 동일한 dim0 범위 커버
```

### Q6: `reshapeIns`와 `transposeIns`의 차이?

**`transposeIns`:** 차원 순서만 변경 (데이터 불변)
```cpp
// 원본: (register:4, lane:8)
auto t = layout.transposeIns({S("lane"), S("register")});
// 결과: (lane:8, register:4)
// 데이터 매핑은 동일, 순서만 바뀜
```

**`reshapeIns`:** 차원을 flatten 후 재분할
```cpp
// 원본: (register:4, lane:8) - 총 32 요소
auto r = layout.reshapeIns({{S("thread"), 32}});
// 결과: (thread:32)
// register와 lane을 하나로 병합

auto r2 = r.reshapeIns({{S("x"), 8}, {S("y"), 4}});
// 결과: (x:8, y:4)
// 다시 2차원으로 분할
```

**언제 transpose 먼저?**
- reshape는 minor-to-major 순서로 flatten
- 원하는 flatten 순서가 아니면 먼저 transpose

```cpp
// (register:4, lane:8)을 lane-major로 flatten하려면:
auto t = layout.transposeIns({S("lane"), S("register")});
auto f = t.flattenIns();
// lane이 더 빠르게 변함
```

### Q7: LinearLayout을 직접 수정할 수 있나?

**답: 불가능 (Immutable)**

모든 LinearLayout 변환 함수는 새 객체를 반환합니다 (`[[nodiscard]]`).

```cpp
// 잘못된 예
layout.transposeOuts({S("dim1"), S("dim0")});  // 반환값 무시 - 경고!

// 올바른 예
layout = layout.transposeOuts({S("dim1"), S("dim0")});
// 또는
auto newLayout = layout.transposeOuts({S("dim1"), S("dim0")});
```

### Q8: 성능 고려사항은?

**컴파일 타임:**
- LinearLayout 연산은 컴파일 타임 (MLIR pass)에서 수행
- 런타임 성능과 무관
- 복잡한 탐색/분석도 OK

**런타임:**
- `applyLinearLayout`로 생성된 코드만 런타임에 실행
- 기저가 sparse하면 생성 코드도 간결
- 예: 항등 레이아웃 → 단순 복사 코드

**최적화 팁:**
- Bank conflict 없는 레이아웃 선택 (컴파일 타임에 검증)
- Vectorization 가능한 레이아웃 (`divideLeft`로 분석)
- Coalesced memory access (연속 레인 → 연속 주소)

---

## 12. 참고 자료

### 12.1 소스 파일

**핵심 헤더:**
- `include/triton/Tools/LinearLayout.h` - 클래스 정의, API, 수학적 배경
- `include/triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h` - 변환 API
- `include/triton/Conversion/TritonGPUToLLVM/Utility.h` - `applyLinearLayout`

**구현:**
- `lib/Tools/LinearLayout.cpp` - 핵심 로직
- `lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp` - 레이아웃 변환 구현
- `lib/Conversion/TritonGPUToLLVM/Utility.cpp` - Lowering 유틸

**테스트:**
- `unittest/Dialect/TritonGPU/LinearLayoutConversionsTest.cpp` - 단위 테스트
- `python/test/unit/language/test_core.py` - E2E 테스트

### 12.2 관련 개념

- **GF(2) Galois Field**: 유한 필드 이론
- **Linear Algebra over GF(2)**: 이진 행렬 연산
- **Tensor Core Programming**: NVIDIA WMMA/MMA, AMD MFMA
- **Shared Memory Banking**: GPU 메모리 시스템
- **CuTe Layout**: NVIDIA CUTLASS의 레이아웃 시스템

### 12.3 예제 코드 위치

**Blocked → Linear:**
- `lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp:926`

**Swizzled Shared → Linear:**
- `lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp:85`

**NVMMA Shared → Linear:**
- `lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp:189`

**Store 연산 lowering:**
- `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/MemoryOpToLLVM.cpp`
- `third_party/amd/lib/TritonAMDGPUToLLVM/LoadStoreOpToLLVM.cpp`

**Layout 변환 (register ↔ shared):**
- `lib/Conversion/TritonGPUToLLVM/ConvertLayoutOpToLLVM.cpp`

---

## 요약

**Linear Layout은:**
1. GPU 텐서 레이아웃을 GF(2) 선형 함수로 통일되게 표현
2. 소수의 기저 벡터로 전체 매핑을 정의 (희소 표현)
3. XOR 선형성으로 모든 값을 계산 (효율적)
4. 합성, 역변환, 재배치 등을 일반적인 연산으로 표준화
5. Triton의 다양한 하드웨어/레이아웃을 하나의 프레임워크로 통합

**핵심 워크플로우:**
```
Triton Layout Attr
  ↓ toLinearLayout()
LinearLayout (register,lane,warp,block) → (dim0,dim1,...)
  ↓ invertAndCompose(sharedLayout)
LinearLayout (register,lane,warp,block) → (offset,block)
  ↓ applyLinearLayout()
LLVM IR: 실제 주소 계산 코드
```

**실전 사용:**
1. 레이아웃 변환: `toLinearLayout(type)`
2. 조합/변환: `operator*`, `compose`, `invertAndCompose`
3. 주소 계산: `applyLinearLayout(loc, rewriter, layout, indices)`
4. 검증: `toString()`, `apply()`, 단위 테스트

이제 Triton의 LinearLayout을 완전히 이해하고 활용할 수 있습니다!
