---
layout: article
title: "PTX Mental Model"
category: "mlsys"
tag: "mlsys"
comment: true
key: 20251005
mathjax: true
---

### Simplest matmul using PTX

During the Chuseok (Mid-Autumn Festival) break I decided to build a clear mental model for how `mma` works at the PTX level.

The example below is close to the simplest matrix multiply that mixes CUDA and PTX.

- The problem size is m16n8k16
- A, B, and accumulator operands all use `f16`.

```c++
    __shared__ __align__(128) half As[16 * 16]; // 16x16 row-major, 512 bytes
    __shared__ __align__(128) half Bs[16 * 8];  // 16x8 row-major, 256 bytes
```
`__align__(128)` aligns smem by 128 bytes which is later required by ldmatrix.

#### Stage 1: Global Memory to Shared Memory using `cp.async`

`cp.async` performs fast, asynchronous copies from global memory (gmem) into shared memory (smem). The instruction accepts two cache modifiers: `.ca`, which lets the copy use both L1 and L2 caches, and `.cg`, which bypasses L1 while still leveraging L2.

In this kernel the copy is a streaming, one-and-done transfer between global memory and shared memory tiles. L1 lives inside each streaming multiprocessor, is only a few dozen kilobytes, and is private to the warp that fills it. That makes it perfect for tiny datasets with tight temporal locality, but a poor fit for bulk transfers that will not be reused. By contrast, L2 is several megabytes, is shared by the whole GPU, and coalesces large sequential accesses before they hit DRAM. Bypassing L1 avoids thrashing its small capacity and keeps the streamed data from polluting caches that could be used by other warps. Consequently `.cg` is the usual choice for tiled matrix copies because it keeps the transaction in L2 and maximizes bandwidth between global memory and shared memory.

`cp.async.cg` therefore skips L1 and reaches shared memory more efficiently for this workload.

```cpp
// ========================================================================
    // Stage 1: Global Memory -> Shared Memory using cp.async
    // ========================================================================
    // cp.async provides hardware-accelerated asynchronous copy
    // Benefits: Non-blocking, bypasses L1 cache, higher bandwidth

    // Convert generic pointers to address space-specific pointers
    // Required by PTX instructions that need explicit address space
    unsigned long long a_smem_ptr = __cvta_generic_to_shared(As);
    unsigned long long b_smem_ptr = __cvta_generic_to_shared(Bs);
    unsigned long long a_gmem_ptr = (unsigned long long)A;
    unsigned long long b_gmem_ptr = (unsigned long long)B;

    // Copy A: 16x16 halves = 512 bytes = 32 chunks of 16 bytes
    // Each thread copies 16-byte chunks (8 halves)
    // cp.async.cg: commit group, ensures 16-byte aligned transfers
    for (int chunk = tid; chunk < 32; chunk += blockDim.x) {
        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], 16;\n"
            :: "l"(a_smem_ptr + chunk * 16),   // Destination: SMEM address
               "l"(a_gmem_ptr + chunk * 16)    // Source: GMEM address
        );
    }

    // Copy B: 16x8 halves = 256 bytes, non-contiguous in GMEM
    // Need to copy row by row since B has stride 16 but we only use 8 columns
    // Each row: 8 halves = 16 bytes
    for (int row = tid; row < 16; row += blockDim.x) {
        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], 16;\n"
            :: "l"(b_smem_ptr + row * 16),     // Destination: SMEM (stride 16 bytes)
               "l"(b_gmem_ptr + row * 32)      // Source: GMEM (32 = 16 halves * 2 bytes)
        );
    }

    // Synchronization for cp.async
    // commit_group: Commits all preceding cp.async operations
    // wait_group 0: Waits for all committed groups to complete
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();  // Ensure all threads see the copied data
```

#### ldmatrix

`ldmatrix` - short for "load matrix" - moves a tile from shared memory into registers. Conceptually the data flows gmem -> smem -> registers, but the interesting part is how the warp distributes pieces of the tile into its lanes so that `mma.sync` receives the layout Tensor Cores expect.

`ldmatrix` is a warp-wide instruction: every lane participates, the source must be in shared memory, and the hardware reshuffles data as it lands in registers. The PTX mnemonic

```
ldmatrix[.sync][.aligned].m8n8[.x1|.x2|.x4][.trans].shared.b16
```

highlights the available modifiers:

- `ldmatrix` indicates a cooperative warp load of a 2D tile from shared memory into registers.
- `.sync` means the instruction is warp-synchronous; all 32 lanes must execute it together.
- `.aligned` promises that the shared memory address is 16-byte aligned so the hardware can issue 128-bit transactions efficiently.
- `.m8n8` selects the tile shape. Tensor Cores consume 8x8 fragments, so a warp always loads that size.
- `.x1`, `.x2`, or `.x4` specify how many tiles each lane receives. For m16n8k16 MMA, operand A is usually loaded with `.x4` while operand B often uses `.x2`.
- `.trans` asks the hardware to interpret the tile as transposed when it is written into registers, which is how we feed row-major data into the layout expected by the Tensor Cores.
- `.shared` declares that the source memory space is shared.
- `.b16` states that each element is 16 bits (half precision), so an 8x8 tile contains 64 elements or 128 bytes.

Putting that together, the variant we rely on is `ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16`.

#### Loading A, B fragments

```cpp
    unsigned a_reg[4];              // A fragment: 4x 32-bit registers (8 halves)
    int a_quad = lane >> 3;                  // Thread group ID: 0..3 (8 threads per group)
    int a_row  = lane & 7;                   // Row within 8x8 tile: 0..7
    int a_col_block = (a_quad & 1) * 8;      // Column block: 0 or 8
    int a_row_block = (a_quad >> 1) * 8;     // Row block: 0 or 8

    // Calculate SMEM address for this thread's starting position
    // Row-major address: &As[(row_block + row) * 16 + col_block]
    unsigned long long a_addr =
        a_ptr + (unsigned long long)((a_row_block + a_row) * 16 + a_col_block) * sizeof(half);

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
        "{%0,%1,%2,%3}, [%4];\n"
        : "=r"(a_reg[0]), "=r"(a_reg[1]), "=r"(a_reg[2]), "=r"(a_reg[3])  // Output: 4 registers
        : "l"(a_addr)                                                     // Input: SMEM address
    );

    // Thread-to-data mapping:
    // - Two 8x8 tiles along K dimension (rows 0-7 and 8-15)
    // - Upper 16 lanes (16-31) reuse addresses of lower 16 lanes for safety
    unsigned b_reg[2];              // B fragment: 2x 32-bit registers (4 halves)
    int b_quad = lane >> 3;                  // Thread group ID: 0..3 (8 threads per group)
    int b_row  = lane & 7;                   // Row within 8x8 tile: 0..7
    int b_k_block = (b_quad & 1) * 8;        // K block: 0 or 8 (along rows)

    // Calculate SMEM address: &Bs[(k_block + row) * 8 + 0]
    unsigned long long b_addr =
        b_ptr + (unsigned long long)((b_k_block + b_row) * 8) * sizeof(half);

    // Upper 16 lanes reuse lower addresses for .x2 safety
    // This is a common pattern to avoid addressing issues
    if (b_quad > 1) {
        int lower = lane & 15;               // Map to lower 16 lanes: 0..15
        int lg = lower >> 3;                 // Group in lower half: 0..1
        int lr = lower & 7;                  // Row in lower half: 0..7
        int lrBlk = (lg & 1) * 8;           // K block for lower half
        b_addr = b_ptr + (unsigned long long)((lrBlk + lr) * 8) * sizeof(half);
    }

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
        "{%0,%1}, [%2];\n"
        : "=r"(b_reg[0]), "=r"(b_reg[1])    // Output: 2 registers
        : "l"(b_addr)                        // Input: SMEM address
    );
```

#### Stage 3: Matrix Multiply-Accumulate (MMA)

The PTX mnemonic looks dense, so here is a quick legend:

- `mma`: matrix multiply–accumulate; computes `D = A × B + C` at warp scope via Tensor Cores.
- `.sync`: warp-synchronous execution; all 32 lanes must participate so the warp behaves like a single vector unit.
- `.aligned`: promises the operand fragments are 128-bit aligned, enabling wide loads and stores.
- `.m16n8k16`: tile shape with `M = 16`, `N = 8`, `K = 16`; the warp multiplies a 16×16 tile of `A` with a 16×8 tile of `B` to produce a 16×8 tile of `C`.
- `.row.col`: layout tags for `A` and `B`; `A` is row-major, `B` is column-major, matching NVIDIA’s MMA conventions.
- `.f16.f16.f16.f16`: data types for `D`, `A`, `B`, and input `C`; this variant keeps everything in half precision.

In short, `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16` asks the warp to multiply those tiles, accumulate into the existing registers, and leave the result in place.


With both operand fragments staged in registers, the multiply-accumulate step drops to a single Tensor Core instruction. `mma.sync` consumes the warp-scoped fragments, performs the fused multiply-accumulate in hardware, and updates the accumulator registers in place.
```cpp
    unsigned c_reg[2] = {0u, 0u};   // C accumulator: 2x 32-bit registers (4 halves), init to 0
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(c_reg[0]), "+r"(c_reg[1])                                // Output: C/D registers (in-place)
        : "r"(a_reg[0]), "r"(a_reg[1]), "r"(a_reg[2]), "r"(a_reg[3]),   // Input: A registers
          "r"(b_reg[0]), "r"(b_reg[1])                                  // Input: B registers
    );
```

The accumulator now holds the 16x8 tile of products, still spread across the warp. Each lane owns a 2x2 slice, and the layout follows the interleaving baked into `mma.m16n8k16`. The final step scatters those slices back to global memory.

#### Stage 4: Store Results to Global Memory (Scatter)

`mma.m16n8k16` leaves the accumulator in a warp-distributed layout, so the
final step is a scatter. A few anchors help keep the pattern straight:

- **Lane ownership:** every lane holds a 2x2 sub-tile; the lower pair of halves
  belong to row `r` and the upper pair to row `r + 8`.
- **Quad grouping:** bundling the 32 threads into eight "quads" (four lanes)
  lines up with the hardware layout—each quad handles one pair of rows while
  its lanes march across columns in steps of two.
- **Register view:** the accumulator arrives as `uint32` registers; casting to
  `half2` lets us extract the four halves without bit twiddling.

With those rules in hand, we compute the base row and column for the lane and
store the two rows separated by eight positions to rebuild the 16x8 tile in
global memory.

```cpp
// ========================================================================
    // Stage 4: Store Results to Global Memory (Scatter)
    // ========================================================================
    // Each lane stores a 2x2 sub-tile. mma.m16n8k16 interleaves rows in
    // groups of eight, so every lane writes to two output rows separated
    // by eight positions.

    // Reinterpret c_reg as half2 for easier access to packed halves
    half2* c_as_h2 = reinterpret_cast<half2*>(c_reg);
    auto a = __low2half(c_as_h2[0]), b = __high2half(c_as_h2[0]),
         c = __low2half(c_as_h2[1]), d = __high2half(c_as_h2[1]);

    // Calculate output position for this lane
    int quad = lane / 4;           // Quad ID: 0..7 (4 threads per quad)
    int col_in_quad = lane % 4;    // Column position within quad: 0..3

    int r0 = quad;                 // Base row: 0..7
    int c0 = col_in_quad * 2;      // Base column: 0,2,4,6 (each thread handles 2 columns)

    // Calculate pointers to two rows (r0 and r0+8)
    half* Cbase0 = C + r0 * 8 + c0;       // Row r0
    half* Cbase1 = C + (r0 + 8) * 8 + c0; // Row r0+8 (8 rows later)

    // Write 2x2 tile: lower pair to row r0, upper pair to row r0+8
    Cbase0[0] = a;     // [r0, c0]
    Cbase0[1] = b;    // [r0, c0+1]
    Cbase1[0] = c;     // [r0+8, c0]
    Cbase1[1] = d;    // [r0+8, c0+1]
```


Full code is available at
https://gist.github.com/ita9naiwa/22814334591e1a3b31b0200a88d9ec89

The gist also holds the full kernel and a minimal harness so you can step through the sequence yourself. Walking through the movement of data from global memory to Tensor Cores and back made the PTX instructions far less mysterious for me, and I hope this breakdown helps you build the same intuition.
