# exp_01 — root-cause the deterministic `address (nil)` fault in `mps_mega`

## Status
IN PROGRESS — started 2026-08-11 07:02Z.

## The evidence we start from (from `../../MPS_OVERNIGHT_HANDOFF.md`)

- Deterministic `Memory access fault by GPU node-2..9 on address (nil)` on all 8
  GPUs at `mps_mega`'s FIRST launch, `K0_MPS_CFG="C=8,g=2,mode=2,flush_rows=16"`.
- `K0_MPS_DEBUG_STOP` bisect: stops 1..6 (M0 through M7 incl. the stream-mode
  enqueue hooks) are all CLEAN. Only the full run faults.
- Fault domain is therefore `{M7.6 service, M8 dynamic combine, M9}`.
- Host assertions that PASS today: `desc_mps.numel()==63`; words 0..54 shared
  with the working `pf6gm_mega` arm; word 55 → 9-word snapshot
  `{local_heap_base, peer_base[0..7]}` readback-validated nonzero; config word
  passes `config_is_valid`; **slots 60/61 alignment checks pass**.

## The hypothesis this experiment tests (H1)

**`desc[61]` (the ~448 MiB slots arena from mori `shmem_malloc`) is NULL.**

Reasoning that promotes this above the handoff's "pointer flavor" framing:

1. **`(nil)` means the faulting VA is literally 0.** A wrong heap-relative
   translation (`local - local_heap_base + peer_base`) of a *valid* pointer
   yields a garbage **nonzero** address, which the fault handler would print as
   a real number. Only a genuinely null pointer prints `(nil)`.
2. **`peer_ptr(nullptr, self)` evaluates to exactly 0.** The translation is
   `local - local_heap_base + peer_base[r]`. For the self slot,
   `peer_base[self] == local_heap_base`, so a null `local` gives
   `0 - local_heap_base + local_heap_base == 0`. Null slots ⇒ address (nil),
   exactly, on every rank. This is a much tighter fit than "wrong flavor".
3. **Every stated host assertion still passes for `nullptr`.** An alignment
   check on 0 succeeds trivially (0 is aligned to everything). So a failed
   allocation would sail through the entire existing guard set silently — which
   is precisely why this survived the previous session's checks.
4. **Only the tail dereferences slots.** M0..M7 never touch the slots arena,
   which is why the debug-stop bisect is clean through stop=6 and the fault is
   universal and instant at the full run.

H1 is falsifiable in one cheap run: dump the descriptor and look at word 61.

## Method

`K0_MPS_DESC_DUMP=1` writes `/out/mps_desc_rank*.txt` with the exact 63 words at
first launch. Run it at `K0_MPS_DEBUG_STOP=6` — a configuration proven CLEAN —
so we obtain the descriptor **without** taking the fault, and the run also
completes so the harness writes its artifacts. `K0_MPS_TRACE=1` additionally
writes `/out/progress_rank*.log` markers.

Launcher: `../../tools/run_descdump.sh` (output root
`$HOME/k0-mok-mps-descdump`, log `$HOME/exp01_descdump.log`).

## Decision rule (pre-registered, before seeing the numbers)

- **`desc[61] == 0`** ⇒ H1 CONFIRMED. The bug is host-side: a failed/absent
  allocation. Fix = make the allocation succeed (heap sizing) **and** add a
  fail-fast host null check so a null pointer can never again pass the
  descriptor gate. Then re-run the ladder.
- **`desc[61] != 0` but OUTSIDE `[local_heap_base, local_heap_base+heapSize)`**
  ⇒ the handoff's original suspect #1 (wrong pointer flavor / not heap-relative)
  is live; translated addresses are wrong-but-nonzero, and we must explain the
  `(nil)` print some other way (e.g. a *different* null pointer in the tail).
- **`desc[61]` valid and in range** ⇒ H1 dead. Move to suspect #2 (M8
  dynamic-claim/slot path) via the `pull_fallback=1` discriminator described in
  the handoff: mode 2 + `pull_fallback=1` keeps streamed flags but switches M8
  to remote `part` pulls. Clean ⇒ fault is in slot addressing; still faults ⇒
  fault is in the service/stream protocol.

Two read-only subagents run in parallel with this GPU job: a host-side
descriptor-build audit (where word 61 is assigned; is the allocation return
ever null-checked; what is the mori symmetric heap size) and a device-side
tail audit (every pointer dereference reachable only in M7.6/M8/M9, ranked).

## Node lease

Preflight clean at launch: `pgrep -af 'torchrun|mpirun'` empty, `rocm-smi
--showpids` shows only `gpuagent` (pid 44579). No preemption of other tenants
was required.
