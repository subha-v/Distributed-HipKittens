#!/usr/bin/env bash
# exp_34 confound fix: make mode 14's per-task VMEM drain deletion
# INDEPENDENTLY SELECTABLE via g bit 0x80 (kCoarseKeepDrainBit), so the
# waterfall can price granularity and drain deletion separately in ONE binary.
# Also dumps the m8_batch acquire site for the ISA-marker build. CPU ONLY.
set -uo pipefail
D=$HOME/e34/DHK/distributed-kernels/fused_moe
# Idempotent: snapshot once, then always start from that snapshot.
[ -f "$HOME/e34/out/pre_keepdrain.hip" ] || \
  cp "$D/k0pf6gm_device_tile_mps.hip" "$HOME/e34/out/pre_keepdrain.hip"
[ -f "$HOME/e34/out/pre_keepdrain.cuh" ] || \
  cp "$D/moe_mps_adapter.cuh" "$HOME/e34/out/pre_keepdrain.cuh"
cp "$HOME/e34/out/pre_keepdrain.hip" "$D/k0pf6gm_device_tile_mps.hip"
cp "$HOME/e34/out/pre_keepdrain.cuh" "$D/moe_mps_adapter.cuh"

python3 - <<'PY'
import io, sys
D = "/home/subvadla/e34/DHK/distributed-kernels/fused_moe/"
ADP = D + "moe_mps_adapter.cuh"
KRN = D + "k0pf6gm_device_tile_mps.hip"

def edit(path, subs):
    s = io.open(path, encoding="utf-8").read()
    for old, new in subs:
        if s.count(old) != 1:
            print("ANCHOR FAIL count=%d in %s: %r" % (s.count(old), path, old[:90]))
            sys.exit(1)
        s = s.replace(old, new)
    io.open(path, "w", encoding="utf-8", newline="\n").write(s)
    print("patched", path)

edit(ADP, [
 # (1) the g-bit table: 0x0080 is no longer reserved.
 ("//   0x0080  reserved (rejected)\n",
  "//   0x0080  exp_34 C: mode 14 KEEPS mode 12's per-task VMEM drain\n"),

 # (2) the constant + the word-bit alias, and admit it to the legal set.
 ("""inline constexpr std::uint32_t kRemoteAccumGLegalBits =
        kRemoteAccumGMask | kRemoteAccumDetectBit | kRemoteAccumThrottleBit |
        kRemoteAccumSkipPartZeroBit | kRemoteAccumThrottleDepthMask;
""",
  """// ---- exp_34 C: the drain-deletion CONFOUND CONTROL bit -----------------------
// Mode 14 changes two things at once against mode 12: the readiness signal's
// granularity, and -- because the per-task event publication disappears with the
// per-row protocol -- the per-task VMEM drain that only ever ordered that
// publication (~2,840 vmcnt(0) + 2,840 __syncthreads() per CTA). A single
// mode 12 -> mode 14 waterfall rung would therefore carry two variables. This
// bit restores the deferred drain under mode 14 so the two are priced
// separately, in ONE binary, from the config word the per-task hook already
// loads: rung (i) mode 14 + 0x80 = granularity alone, rung (ii) mode 14 = plus
// the drain deletion. Legal ONLY on mode 14 (rejected below elsewhere), because
// on modes 12/13 the drain is not deleted and the bit would mean nothing.
inline constexpr std::uint32_t kCoarseKeepDrainBit = 0x80u;
// Same bit as seen in the PACKED descriptor word: encode_config puts the low
// byte of `group_slices` at bit 8, so 0x80 lands at bit 15. The per-task hooks
// test the packed word directly (they already load it for the mode field) rather
// than decoding a config struct on the hot path.
inline constexpr std::uint64_t kCoarseKeepDrainWordBit = 0x8000ull;
inline constexpr std::uint32_t kRemoteAccumGLegalBits =
        kRemoteAccumGMask | kRemoteAccumDetectBit | kRemoteAccumThrottleBit |
        kRemoteAccumSkipPartZeroBit | kRemoteAccumThrottleDepthMask |
        kCoarseKeepDrainBit;
"""),

 # (3) the predicate, next to mode_is_coarse.
 ("""__host__ __device__ __forceinline__ bool mode_is_coarse(config c) {
    return c.mode == kModeCoarseReady;
}
""",
  """__host__ __device__ __forceinline__ bool mode_is_coarse(config c) {
    return c.mode == kModeCoarseReady;
}

// exp_34 C: true when mode 14 is asked to keep mode 12's per-task drain. Host
// side only (reporting / validation); the device hook reads the packed word.
__host__ __device__ __forceinline__ bool coarse_keeps_drain(config c) {
    return mode_is_coarse(c) && (c.group_slices & kCoarseKeepDrainBit) != 0u;
}
"""),

 # (4) validator: the bit is meaningless off mode 14, so reject it there.
 ("""        if (mode_is_coarse(c) &&
            (c.group_slices & kRemoteAccumDetectBit) != 0u) return false;
""",
  """        if (mode_is_coarse(c) &&
            (c.group_slices & kRemoteAccumDetectBit) != 0u) return false;
        // exp_34 C: the drain-retention control has no meaning where the drain
        // was never deleted. Reject on 12/13 instead of silently ignoring it --
        // otherwise a mistyped waterfall arm would LOOK like the control arm.
        if ((c.group_slices & kCoarseKeepDrainBit) != 0u &&
            !mode_is_coarse(c)) return false;
"""),
])

edit(KRN, [
 ("#define K0P6_MPS_SRC_REV 27", "#define K0P6_MPS_SRC_REV 28"),

 # task_drain: record that the deletion is selectable.
 ("""  // verbatim. Noted as a rung deletion, not a free ride: it removes ~2,840
  // vmcnt(0) drains and ~2,840 __syncthreads() per CTA.
""",
  """  // verbatim. Noted as a rung deletion, not a free ride: it removes ~2,840
  // vmcnt(0) drains and ~2,840 __syncthreads() per CTA. exp_34 C makes that
  // deletion INDEPENDENTLY SELECTABLE so the waterfall can price it apart from
  // granularity: with `g |= kCoarseKeepDrainBit` (0x80) the deferred drain comes
  // back through k0p6_mps_task_done_maybe_defer. This hook is unconditional for
  // mode 14 either way -- the drain, when kept, is paid at the next task head,
  // which is exactly where mode 12 pays it.
"""),

 # task_done_maybe_defer: the selectable arm.
 ("""  // exp_34 mode 14 buffers nothing, so `p.b0` stays -1 for the whole task loop
  // and k0p6_mps_task_flush_defer returns on its first line at every task head:
  // no deferred vmcnt(0), no deferred __syncthreads(), no publication.
  if (m == 14ull) return;
""",
  """  // exp_34 mode 14 buffers nothing, so `p.b0` stays -1 for the whole task loop
  // and k0p6_mps_task_flush_defer returns on its first line at every task head:
  // no deferred vmcnt(0), no deferred __syncthreads(), no publication.
  //
  // exp_34 C (confound control): with kCoarseKeepDrainBit set, buffer the task
  // identity with gcount == 0. k0p6_mps_task_flush_defer then still pays its
  // vmcnt(0) + __syncthreads() at the next task head, and its publication loop
  // runs ZERO iterations -- mode 14 with mode 12's drain and with no publication
  // (mode 14 has no consumer for one: the queue is dead and an event would make
  // K0P6_MPS_ERR_SERVICE ambiguous). That is the arm that prices signal
  // granularity alone; the default arm adds the drain deletion on top.
  if (m == 14ull) {
    if ((w & hk_moe::mps::kCoarseKeepDrainWordBit) == 0ull) return;
    if (p.b0 < 0) {
      p.b0 = b;
      p.nc = nc;
    }
    p.gcount = 0;
    return;
  }
"""),
])

# the mode field in task_done_maybe_defer must come from a word we keep, so the
# bit test costs no second descriptor read.
s = io.open(KRN, encoding="utf-8").read()
old = """__device__ __forceinline__ void k0p6_mps_task_done_maybe_defer(
    int b, int nc, int tid, const long long* k0p6_desc, k0p6_defer& p) {
  const unsigned long long m =
      ((unsigned long long)k0p6_dread(k0p6_desc, K0P6_D_MPS_CFG) >> 16 &
       0xFFull);
"""
new = """__device__ __forceinline__ void k0p6_mps_task_done_maybe_defer(
    int b, int nc, int tid, const long long* k0p6_desc, k0p6_defer& p) {
  // One load of the packed config word serves both the mode field and exp_34
  // C's drain-retention bit (bit 15 of the word = 0x80 of `g`).
  const unsigned long long w =
      (unsigned long long)k0p6_dread(k0p6_desc, K0P6_D_MPS_CFG);
  const unsigned long long m = (w >> 16 & 0xFFull);
"""
assert s.count(old) == 1, s.count(old)
io.open(KRN, "w", encoding="utf-8", newline="\n").write(s.replace(old, new))
print("patched maybe_defer word load")
PY
rc=$?
[ $rc -eq 0 ] || { echo "PATCH FAILED rc=$rc"; exit 1; }

echo "### diff summary vs pre-keepdrain"
diff -u "$HOME/e34/out/pre_keepdrain.cuh" "$D/moe_mps_adapter.cuh" | head -80
diff -u "$HOME/e34/out/pre_keepdrain.hip" "$D/k0pf6gm_device_tile_mps.hip" | head -80

echo "### m8_batch acquire site (for the ISA marker build)"
grep -n -B6 -A10 "acquire_payload_system" "$D/k0pf6gm_device_tile_mps.hip" | head -60
echo "===DONE==="
