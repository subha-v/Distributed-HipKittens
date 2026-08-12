# exp_22 — resource saturation vs CTA count (NanoFlow Fig 7 analog)

Status: **planned, unbuilt**. Ubench only — no megakernel edits, no campaign arms.

## Goal

Produce the CDNA4 analog of NanoFlow v1 Figure 7: per-resource throughput as a
function of CTA count, for the three resource classes this kernel actually
uses, **plus a concurrent overlay NanoFlow does not have** (each curve measured
again while the M7-shaped GEMM runs on the remaining CTAs).

Three deliverable numbers, all currently argued from citation or sweep:

1. **Knee of the xGMI push curve** → the topology-correct service pool size C,
   derived from measurement instead of the (C, g, mode) sweep.
2. **Concurrent/isolated ratio at that C** → how much of isolated push
   bandwidth survives under M7 traffic (the exp_20 interference finding as a
   reusable curve, not a one-off).
3. **MFMA capacity curve at 256−C** → confirms the fitted 256/(256−C) tax law.

Reference ceilings for the plot: 76.8 GB/s per xGMI link per direction,
537.6 GB/s aggregate egress, 8 TB/s HBM3E, and the cost model's 148 GB/s
service-rate requirement (27.5% of egress).

## Hypotheses (pre-registered)

- H1: xGMI single-link push saturates by 8 CTAs at MLP≥4 (competition
  analysis: winners provision ~1 CTA per XCD per link; gemm-rs rank02 holds
  full rate with 8–16 CTAs via load depth).
- H2: aggregate-egress push saturates by 16–32 CTAs.
- H3: the concurrent xGMI curve is depressed well below the isolated one at
  equal C, and the depression grows with the *protocol* knobs (probe atomics),
  not payload bytes — consistent with exp_20's protocol-not-payload verdict.
- H4: MFMA TFLOPS vs CTAs is linear (one block per CU; no oversubscription
  regime exists on CDNA4).

Falsifier: if the isolated single-link curve needs ≥32 CTAs to reach ~75% of
76.8 GB/s at any MLP depth, the "tiny topology-sized pool" story is wrong for
our pusher shape and A1's C=64 optimum is *bandwidth-limited*, not
protocol-limited — that inverts the M4-first priority.

## Sweep

| axis | points |
|---|---|
| mode | a=GEMM, b=HBM reduce, c=xGMI push |
| CTAs (mode a) | 32, 64, 96, 128, 160, 192, 224, 256 |
| CTAs (mode b) | 8, 16, 32, 64, 128, 256 |
| CTAs (mode c) | 1, 2, 4, 8, 16, 32, 64 |
| MLP depth (mode c only) | 1, 4, 8 |
| peer fanout (mode c only) | single-peer, round-robin-7 |
| overlay | isolated; concurrent (role-split with mode-a tasks on 256−C CTAs) |

Concurrent overlay runs modes b and c only, against a **C-matched reserve-only
control** (exp_20 lesson: 192-with-traffic vs 192-without, never vs 256).

## Protocol

- 5 rotations per point, median; each point sized to 10–50 ms of kernel time
  (mode c: ≥4 GB pushed per launch).
- Host hipEvent wall time + per-CTA `s_memrealtime` role start/end (concurrent
  arm roles finish at different times; wall time alone attributes nothing).
- Checksum read-back once per config (a silently failing store must not fake
  bandwidth).
- Standing traps: bump `K0P6_MPS_SRC_REV`-equivalent rev macro in the ubench
  `.hip`, `git reset --hard` on the node checkout, one GPU job, `setsid` +
  `timeout`, never SIGKILL.
- Label all output `valid_diagnostic`; no correctness ladder (ubench), no
  campaign arms touched.

## Validation command

One smoke point per mode on the node before the sweep:

```bash
setsid timeout 600 ./run_ubench.sh --mode c --ctas 8 --mlp 4 --fanout 7 --check
```

(expected: checksum PASS + a GB/s line within [10, 540])

## Effort

1–2 days. The role-split concurrent mode is ~80% of the work; modes a/b are
existing kernel bodies behind a task-list driver.
