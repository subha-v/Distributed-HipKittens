# exp_18 blocker 1 — remote packed-bf16 atomics over xGMI: **PASS**

**Result: `global_atomic_pk_add_bf16` works on a peer address over Infinity
Fabric and is atomic across devices.** The protocol review flagged this as the
measurement that could close A11 on hardware grounds. It does not.

## What was run

Standalone HIP, no mori, no megakernel, seconds to run. Memory is
coarse-grained `hipMalloc` on device 0, which satisfies HIP's documented
precondition for `unsafeAtomicAdd`
(`amd_hip_unsafe_atomics.h:50-54`: fine-grained global memory is UB).

| phase | setup | expectation if correct |
|---|---|---|
| **A** | 256 cells, one thread on **dev1** per cell, 128 sequential remote RMWs of `{1.0, 1.0}` | every cell `== {128, 128}` |
| **B** | same 256 cells, **dev1 and dev2 both** run 256 threads doing 64 RMWs each | every cell `== {128, 128}` |

Phase A asks only whether a remote packed-bf16 read-modify-write is *performed*.
Phase B asks whether it is *atomic* when two devices contend at the remote
coherence point.

## Result

```
devices=8  dev1->dev0 peer=1
A remote RMW (1 writer):     exact=256 zero=0 other=0  cell0=0x43004300 (want 0x43004300)
B cross-device contention:   exact=256 zero=0 other=0  cell0=0x43004300 (want 0x43004300)
VERDICT: PASS
```

**ISA confirms the hardware instruction, not a CAS emulation:**

```
global_atomic_pk_add_bf16 v[0:1], v2, off        // DD488000 007F0200
atomic_pk_add_bf16 count: 1     atomic_cmpswap count: 0
```

## A wrong first attempt, recorded because the trap is easy to fall into

The review's suggested test used `K = 1024` increments of 1.0, reasoning that
1024 is exactly representable in bf16. It ran, and every cell read **256.0**,
which looks exactly like "75% of updates were lost".

It is not. **bf16 has 8 significant bits, so `256 + 1` rounds back to 256** —
the accumulator saturates at 256 regardless of how correct the atomic is. The
final value being representable is not enough; every *partial* sum has to be.
`K` was lowered to 128 so that all sums 1..128 are exact, after which the only
way to miss the target is a genuinely dropped update.

A second trap in the same run: `llvm-objdump` on the host executable
disassembles **host** code and reported zero atomics of any kind. The device
object has to be produced with `--genco` and unbundled first.

## What this does and does not establish

**Established:** the fabric implements a packed-fp remote RMW; it is atomic
under cross-device contention; the compiler emits the hardware instruction; and
notably it works despite the builtin carrying **no memory-scope argument**
(`__builtin_amdgcn_flat_atomic_fadd_v2bf16`, agent scope by construction). The
review's concern that agent scope on a peer address is "the wrong cache domain
by construction" is not borne out in practice on this part.

**Not established — the remaining half of blocker 1:** this used coarse-grained
`hipMalloc`. Our real target is mori's symmetric heap, which is HIP-VMM
allocated with `HeapType::Uncached`, and `Uncached` is not `coarse grain`. The
question is now much narrower than "does the mechanism exist": it is **"does
mori's heap satisfy `unsafeAtomicAdd`'s coarse-grain precondition?"**

One in-tree fact reduces even that risk: the kernel already performs remote
*integer* system-scope RMWs on the mori heap and all gates pass
(`_mps.hip:705-708`, `reserve_rows_relaxed<system>` on
`peer_ptr(dest_counter + cur, dest, symmetric)`), so the heap supports remote
atomics as a class. And per the review, each `slots[p][pos]` plane is written by
exactly one producer rank, so **cross-rank atomicity is not even required by the
design** — only remote atomicity from a single rank plus visibility to the
owner, which is the weaker of the two properties Phase B just demonstrated.

## Status of A11 after this

| blocker | status |
|---|---|
| 1a. does remote packed-bf16 atomic exist and work? | **CLEARED — PASS** |
| 1b. does it hold on mori's VMM `Uncached` heap? | **OPEN** — narrow, needs a heap-grain check or an in-situ test |
| 2. `slots` zeroing / epoch reuse with no cross-rank ordering point | **OPEN — the real design problem** |
| 3. bit-exactness gate | **NOT A BLOCKER** — the gate compares two standalone combine kernels and never touches the megakernel |
| 4. bf16 non-associativity | **NOT A BLOCKER** — the epilogue already does an unordered multi-CTA bf16 atomic fan-in of the same degree |

Blocker 2 is now the critical path, and the review already names the fix:
epoch-parity double-buffering of `slots` using the in-tree exp_56 `dest_counter`
idiom, rather than end-of-epoch zeroing (which would move 448 MiB into the
combine tail and eat most of the prize).

## Artifact

`~/overnight-scratch/exp_18/bf16_peer_atomic.log`, source at `/tmp/bf16pa.cpp`
inside `subha_k1`.
