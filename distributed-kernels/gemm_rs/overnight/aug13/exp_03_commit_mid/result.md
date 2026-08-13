# exp_03 result — COMMIT_MID ladder run 1

**Status: LADDER ABORTED AT M9 (GPU memory access fault); attribution
running.** M1–M5 + placement + lds_race were green; M7 (the paired speed
campaign) never ran, so there is NO speed verdict on row B yet. No
production artifact was touched (`build/gemm_rs_mi300x.so` untouched by
`build_arms.sh` by construction).

## Ladder record (2026-08-13 22:44–23:33Z, lease aug13_exp03)

| gate | verdict | detail |
|---|---|---|
| M1 build | OK | cmid / cmid0 410256 B; cmid0b 410264 B |
| M2 census | PASS (placement) | base: vmcnt0→barrier **0** MFMA, barrier→vmcnt0 **64**; cmid: **32/32** — exactly the designed mid-commit split, rotation-invariant profile. VGPR 246/248 on the 256-family, 0 spills, 0 scratch, both arms. |
| M2 ratchet | **DIFF (advisory)** | flag-off ISA vs exp_27 archive: 18 diff lines. Known caveat written in census.sh: exp_27 censused the PRE-REVERT source (exp_26's PERSHAPE was shipped then reverted between the census and tonight). The binding default-off guarantee remains build-level: `build_arms.sh` never writes the production .so. |
| lds_race | clean | 0 hazards across all symbols on the cmid ISA |
| M3 (cmid) | green | 17/17 shapes, 1e-2 AND 2e-3 |
| M4 controls | green | all three negative controls failed exactly as designed |
| M5 soak (cmid) | green | 600 epochs, worst \|diff\| 4.883e-04 vs 2e-3 |
| M9 (cmid) | **GPU FAULT** | see below |

VGPR note: cmid shifts allocation on the small instantiations
(<32,64,128> 98→108, <64,128,64> 104→112, <128,192,32> 136→131); the
256-family — where M7 would measure — is bit-for-bit at 246/248 either way.

## The M9 event

`m9_stale_slot.py` on `HK_KERNEL_MODULE=gemm_rs_mi300x_cmid` hit

    Memory access fault by GPU node-7 (Agent handle: 0xda49c40) on address
    0x7fa306053000. Reason: Unknown.

then wedged in the GPU coredump handler (`execvp failed` → broken pipe →
handler status 1). `timeout`'s TERM never landed — the tree sat 45 min past
its 30-min budget until an external `kill -9` (container pids) recovered it;
`run_ab.sh` then completed its die path (ABORT 23:33:07Z) and the trap
released the lease. KFD came back clean; no node reset was needed.

Evidence quality caveats, recorded before attribution:

1. **The run's stdout was lost** — python buffers stdout when redirected and
   the tree died by SIGKILL, so every sweep/epoch marker vanished. The fault
   CANNOT be placed within the run: each M9 sweep epoch interleaves the
   GOLDEN module (`gemm_rs_mi300x_e3base`), the candidate (cmid), and later
   the CTRL_PUBLISH_EARLY control. The faulting kernel is unknown.
2. M9's own docstring documents this exact signature ("Memory access fault
   by GPU node-7", wedge in driver teardown) as the STALE-GOLDEN failure
   mode. The table sidecar guard passed (the run reached GPU work), but the
   guard only compares bm/bn/bk per case — it is not proof the golden is
   sound.
3. Two defunct `[timeout]` zombies from Aug 11/12 sit in the container —
   wedges of this general kind predate tonight.
4. M9 had not been run on ANY module in the aug13 sessions before tonight;
   its last green run predates this branch state. The fault is therefore NOT
   yet attributable to the cmid edit.

## Attribution (m9_attrib.sh, pre-registered, running)

Canary M3(prod) for node health, then M9 full-scale with `python3 -u` (so
the faulting case survives this time) on:

- run 1 `cmid` — reproduce?
- run 2 `cmid0` — the base twin from the same fresh build family.

| run 1 | run 2 | verdict |
|---|---|---|
| fault | green | arm-specific: cmid REJECTED on correctness grounds; row B closed regardless of speed |
| fault | fault | environment/instrument (golden or harness): cmid exonerated pending M9 infra fix |
| green | green | nonreproducible: rerun 1 before concluding anything |

Wedge-proofing added: `timeout -k` (KILL after TERM), post-fault container
pkill + kfd wait between runs.

## What this means for the queue

M7 is blocked behind an M9 verdict either way: the ship rule requires the
full ladder, and a candidate that faults a null-perturbation gate does not
get timed. If attribution lands on the environment, the M9 infrastructure
(golden rebuild per its own doc, or the harness fill path) must be fixed and
M9 re-run green on cmid before M7 may proceed.
