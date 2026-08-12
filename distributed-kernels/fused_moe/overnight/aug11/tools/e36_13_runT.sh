#!/usr/bin/env bash
# exp_36: run the standard screen ladder at one T.
#   usage: runT.sh <T> <TAG> [cfgset]
# cfgset: std (default) | ratchet | skewprobe
#
# The ladder holds one variable per point against the ratchet config:
#   1 C=16 g=353  depth 4, throttle on   <- the ratchet
#   2 C=16 g=97   depth 8                <- throttle depth sensitivity
#   3 C=16 g=65   throttle OFF           <- is the injection bound still worth it
#   4 C=8  g=353  smaller service pool   <- placement axis down
#   5 C=32 g=353  bigger service pool    <- placement axis up
#   6 C=64 g=1 mode=2                    <- the dedicated-CTA placement arm
set -u
E36="$HOME/e36"
T="${1:?T}"
TAG="${2:?TAG}"
SET="${3:-std}"
CF="$E36/${TAG}.cfgs"
mkdir -p "$E36"
case "$SET" in
  std)
    cat > "$CF" <<'EOF'
C=16,g=353,mode=12,flush_rows=16,timestamps=1
C=16,g=97,mode=12,flush_rows=16,timestamps=1
C=16,g=65,mode=12,flush_rows=16,timestamps=1
C=8,g=353,mode=12,flush_rows=16,timestamps=1
C=32,g=353,mode=12,flush_rows=16,timestamps=1
C=64,g=1,mode=2,flush_rows=16,timestamps=1
EOF
    ;;
  ratchet)
    cat > "$CF" <<'EOF'
C=16,g=353,mode=12,flush_rows=16,timestamps=1
EOF
    ;;
  rank3)
    # the three points that span the observed ranking: best (throttled mode 12),
    # the dedicated-CTA placement arm, and the unthrottled neighbour.
    cat > "$CF" <<'EOF'
C=16,g=353,mode=12,flush_rows=16,timestamps=1
C=64,g=1,mode=2,flush_rows=16,timestamps=1
C=16,g=65,mode=12,flush_rows=16,timestamps=1
EOF
    ;;
  c816)
    # third paired replicate of the only sub-5% delta in the sweep.
    cat > "$CF" <<'EOF'
C=16,g=353,mode=12,flush_rows=16,timestamps=1
C=8,g=353,mode=12,flush_rows=16,timestamps=1
EOF
    ;;
  cmin)
    # the placement axis driven down: C=4 is the smallest pool the mode-12
    # validator accepts (C=0 needs mode 14), plus a C=8 replicate for pairing.
    cat > "$CF" <<'EOF'
C=4,g=353,mode=12,flush_rows=16,timestamps=1
C=8,g=353,mode=12,flush_rows=16,timestamps=1
EOF
    ;;
  place4)
    # the placement axis (C) plus the throttle-off neighbour, at campaign res.
    cat > "$CF" <<'EOF'
C=8,g=353,mode=12,flush_rows=16,timestamps=1
C=32,g=353,mode=12,flush_rows=16,timestamps=1
C=64,g=1,mode=2,flush_rows=16,timestamps=1
C=16,g=65,mode=12,flush_rows=16,timestamps=1
EOF
    ;;
  depth)
    cat > "$CF" <<'EOF'
C=16,g=353,mode=12,flush_rows=16,timestamps=1
C=16,g=97,mode=12,flush_rows=16,timestamps=1
EOF
    ;;
  *) echo "unknown cfgset $SET" >&2; exit 2;;
esac
E36_T="$T" E36_TAG="$TAG" E36_CAPMODE="${E36_CAPMODE:-pinned}" \
  E36_PROCS="${E36_PROCS:-1}" E36_WARMUP="${E36_WARMUP:-1}" E36_TIMED="${E36_TIMED:-1}" \
  bash "$E36/launch.sh"
exit 0
