#!/usr/bin/env bash
# exp_34: does mode 14 inherit the exp_24 injection throttle at all? The throttle
# is gated on mode_is_direct_accum(c) && (g & kRemoteAccumThrottleBit). If mode 14
# is not in that predicate, mode 14 is unthrottled BY CONSTRUCTION and the
# waterfall rung (d) is not "mode 12 ratchet + coarse signals".
set -uo pipefail
cd "$HOME/Distributed-HipKittens"
A=distributed-kernels/fused_moe/moe_mps_adapter.cuh
echo "===mode predicates at the pin==="
git show "291dfa08:$A" | grep -n -B3 -A8 -E "mode_is_direct_accum|mode_is_coarse\(config|mode_is_parity_publish|mode_is_stream\(config" | head -70
echo
echo "===the throttle bit constants and depth mask==="
git show "291dfa08:$A" | grep -n -E "kRemoteAccumThrottle|kRemoteAccumSkipPartZero|kRemoteAccumDetectBit|kModeCoarse|kModeRemoteAccum" | head -20
echo "===DONE==="
