#!/usr/bin/env bash
# ===========================================================================
# screen.sh -- reusable MoK synthetic-prefill screening / campaign driver
# for the 8x MI350X node (gbt350-odcdh2-c05-1, gfx950).
#
# Lives in the repo (overnight/aug11/tools/screen.sh) so it is version
# controlled; push it to the node with tools/push.ps1 and run it there.
#
# USAGE
#   bash screen.sh TAG /path/to/cfgs.txt      # one K0_MPS_CFG per line
#   bash screen.sh TAG -  <<EOF               # or on stdin
#   C=16,g=33,mode=12,flush_rows=16
#   EOF
#
# Blank lines and #-comments in the config list are ignored. Whitespace is
# stripped, so "C=16, g=33" and "C=16,g=33" are the same point.
#
# ENV KNOBS (all optional)
#   SCREEN_ARMS        production,pf6gm_mega,mps_mega   -> K0_MOK_ARMS
#   SCREEN_WARMUP      1     -> K0_MOK_WARMUP_ITERS  (campaign: 500)
#   SCREEN_TIMED       1     -> K0_MOK_TIMED_ITERS   (campaign: 100)
#   SCREEN_PROCS       1     -> run_campaign.sh arg 2, the rotation count
#                              (campaign: 5). Arms rotate once per process.
#   SCREEN_RUN_TIMEOUT 1500  -> K0_MOK_RUN_TIMEOUT, per-run docker timeout
#   SCREEN_JOB_TIMEOUT 1800  -> outer `timeout` around the whole campaign
#                              (raise to ~4500 for a 5-process campaign)
#   SCREEN_TRACE       1     -> K0_MPS_TRACE (host-side progress markers)
#   SCREEN_ROUTE       ''    -> K0_SYNTH_ROUTE   (skew family; '' = default)
#   SCREEN_SEED        0     -> K0_SYNTH_SEED
#   SCREEN_SOAK_ITERS  ''    -> K0_MPS_SOAK_ITERS. LEAVE EMPTY. The harness
#                              raises unless it is exactly 600.
#   SCREEN_BRANCH      codex/distributed-hipkittens-scaffold
#   SCREEN_SYNC        1     -> git fetch+reset the node checkout ONCE, before
#                              the first run of the batch and never between
#                              runs (mid-batch resync would silently change
#                              the kernel under a half-finished sweep)
#   SCREEN_IDLE_WAIT   300   -> seconds to wait for a busy node before
#                              recording BUSY for that point and moving on
#
# DISCIPLINE THIS SCRIPT ENFORCES
#   * refuses to start unless `pgrep -af 'torchrun|mpirun'` is empty AND
#     rocm-smi --showpids lists nothing but gpuagent
#   * re-checks idleness before every single run
#   * setsid -w + timeout on every run (-w so the driver actually waits)
#   * every run gets its OWN output root. run_campaign.sh exits 7
#     ("refusing to overwrite completed campaign") if summary.json already
#     exists, and the old driver then re-read the STALE summary and logged
#     the previous run's microseconds against a VOID gate. Unique dirs make
#     re-running the identical config a first-class operation.
#   * records the TRUE mps_mega build identity before and after each run --
#     see the note below, this is subtler than it looks.
#   * a failed point logs a FAIL row and the batch continues.
#
# HSACO EVIDENCE -- READ THIS BEFORE TRUSTING A "REBUILD" CLAIM
#   .../mori/jit/gfx950_mlx5/latest/ is a directory of SYMLINKS into the
#   content-hashed build dirs, and every run re-creates those symlinks. So:
#     * `stat` WITHOUT -L reports the symlink's own mtime, which advances on
#       every run whether or not anything was compiled -- it is worthless as
#       rebuild evidence, and `ls -t` sorts by it, so `ls -t */..hsaco | head`
#       always returns latest/.
#     * `stat -L` (and readlink) report the real hashed build. THAT is the
#       evidence. hsaco_before == hsaco_after means a cache hit.
#   Separately, mori only emits its `[mori-jit]  Cached: <path>` lines on some
#   runs; a full cache hit can produce no jit line at all. So the log-derived
#   hash (jit_log) is frequently empty and must never be the only check.
# ===========================================================================
set -uo pipefail

TAG="${1:-}"
CFGSRC="${2:--}"
if [ -z "$TAG" ]; then
  echo "usage: screen.sh TAG [cfgfile|-]" >&2
  exit 64
fi

ARMS="${SCREEN_ARMS:-production,pf6gm_mega,mps_mega}"
WARMUP="${SCREEN_WARMUP:-1}"
TIMED="${SCREEN_TIMED:-1}"
PROCS="${SCREEN_PROCS:-1}"
RUN_TIMEOUT="${SCREEN_RUN_TIMEOUT:-1500}"
JOB_TIMEOUT="${SCREEN_JOB_TIMEOUT:-1800}"
TRACE="${SCREEN_TRACE:-1}"
ROUTE="${SCREEN_ROUTE:-}"
SEED="${SCREEN_SEED:-0}"
SOAK="${SCREEN_SOAK_ITERS:-}"
BRANCH="${SCREEN_BRANCH:-codex/distributed-hipkittens-scaffold}"
SYNC="${SCREEN_SYNC:-1}"
IDLE_WAIT="${SCREEN_IDLE_WAIT:-300}"

DHK="$HOME/Distributed-HipKittens"
K0="$HOME/amd-master/auto-gpu-kernel/k0_fused_moe"
CAMPAIGN="$K0/benchmarks/mok_synthetic_prefill/run_campaign.sh"
MPSSRC="$DHK/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip"
JITDIR="$HOME/.cache/k0-mok-synthetic-prefill/mori/jit/gfx950_mlx5"
SCRATCH="$HOME/overnight-scratch"
CSV="$SCRATCH/screen_${TAG}.csv"
LOCK="/tmp/k0_mok_synthetic_gpu_lock"
ITERS="w${WARMUP}t${TIMED}p${PROCS}"

mkdir -p "$SCRATCH" "$HOME/k0-mok-${TAG}"

# ---------------------------------------------------------------- config list
CFGS=()
if [ "$CFGSRC" = "-" ]; then CFGSRC=/dev/stdin; fi
if [ ! -r "$CFGSRC" ]; then
  echo "cannot read config list: $CFGSRC" >&2
  exit 65
fi
while IFS= read -r raw || [ -n "$raw" ]; do
  raw="${raw%%#*}"
  raw="$(printf '%s' "$raw" | tr -d '[:space:]')"
  [ -n "$raw" ] && CFGS+=("$raw")
done < "$CFGSRC"
if [ "${#CFGS[@]}" -eq 0 ]; then
  echo "config list is empty" >&2
  exit 66
fi

# ------------------------------------------------------------------- helpers
foreign_procs () { pgrep -af 'torchrun|mpirun' 2>/dev/null; }

kfd_procs () {
  /opt/rocm/bin/rocm-smi --showpids 2>/dev/null |
    awk '/^[0-9]+[ \t]/ { if ($2 != "gpuagent") print $1" "$2 }'
}

node_idle () {
  local a b
  a="$(foreign_procs)"
  b="$(kfd_procs)"
  if [ -n "$a" ] || [ -n "$b" ]; then
    printf '%s\n%s\n' "$a" "$b" | sed '/^$/d'
    return 1
  fi
  return 0
}

# "<true mtime>|<content hash dir>" for the mps_mega build the next run will
# load. Resolves latest/ through the symlink on purpose -- see the header.
hsaco_stamp () {
  local link f
  link="$JITDIR/latest/k0pf6gm_mps_mega.hsaco"
  if [ -e "$link" ]; then
    f="$(readlink -f "$link")"
  else
    f="$(ls -tL "$JITDIR"/*/k0pf6gm_mps_mega.hsaco 2>/dev/null | head -1)"
  fi
  if [ -z "$f" ] || [ ! -e "$f" ]; then printf 'none|none'; return; fi
  printf '%s|%s' "$(stat -L -c %y "$f" | cut -c1-19)" "$(basename "$(dirname "$f")")"
}

say () { echo "[$(date -u +%H:%M:%SZ)] $*"; }

# --------------------------------------------------------------- batch header
say "SCREEN $TAG start -- ${#CFGS[@]} config(s), arms=$ARMS, iters=$ITERS"
echo "host: $(hostname)"

if [ -d "$LOCK" ]; then
  if node_idle >/dev/null; then
    say "WARN stale GPU lease dir $LOCK on an idle node (mtime $(stat -c %y "$LOCK" | cut -c1-19)) -- removing"
    rmdir "$LOCK" 2>/dev/null || say "WARN could not remove $LOCK"
  else
    say "FATAL $LOCK held and the node is NOT idle:"
    node_idle
    exit 6
  fi
fi

if ! node_idle > /tmp/screen_busy.$$ 2>&1 ; then
  say "REFUSING TO START -- node is not idle:"
  cat /tmp/screen_busy.$$
  rm -f /tmp/screen_busy.$$
  exit 20
fi
rm -f /tmp/screen_busy.$$
say "node idle: no torchrun/mpirun, no non-gpuagent KFD process"

# The campaign containers bind-mount ~/Distributed-HipKittens READ-ONLY, so the
# node checkout is the source of truth for every arm. Sync ONCE per batch.
if [ "$SYNC" = "1" ]; then
  say "syncing node checkout to origin/$BRANCH (once, before the first run)"
  git -C "$DHK" fetch --all -q
  git -C "$DHK" reset -q --hard "origin/$BRANCH"
else
  say "SCREEN_SYNC=0 -- leaving the node checkout as-is"
fi
HEADSHA="$(git -C "$DHK" rev-parse --short HEAD)"
SRCREV="$(grep -oE 'K0P6_MPS_SRC_REV [0-9]+' "$MPSSRC" | tail -1 | awk '{print $2}')"
echo "node source : $(git -C "$DHK" log --oneline -1)"
echo "node HEAD   : $(git -C "$DHK" rev-parse HEAD)"
echo "node branch : $(git -C "$DHK" rev-parse --abbrev-ref HEAD)"
echo "SRC_REV     : ${SRCREV}"
echo "mps hsaco @batch start: $(hsaco_stamp)"

HDR='idx,utc,cfg,status,iters,head,src_rev,prod_us,pf6gm_us,mps_us,ratio_vs_prod,ratio_vs_pf6gm,pf6gm_vs_prod,gate_pass,gate_max_abs,gate_rel,control_fails,pperr,soak,soak_epochs,ts_planM6_us,ts_M7_us,ts_combine_us,ts_servicedrain_us,ts_m2_to_end_us,ts_planM3toM5_us,ts_M6_us,spin_ok_max,spin_fail_max,eager_status,eager_gate_ok,mps_hsaco,jit_log,hsaco_before,hsaco_after,rc,outdir,note'
[ -f "$CSV" ] || echo "$HDR" > "$CSV"
NCOL=$(echo "$HDR" | tr ',' '\n' | wc -l)

# A row we can write without the python parser (BUSY / LOCKED / SKIPPED).
stub_row () {
  local idx="$1" cfg="$2" status="$3" note="$4" i
  printf '%s,%s,"%s",%s,%s,%s,%s' \
    "$idx" "$(date -u +%FT%TZ)" "$cfg" "$status" "$ITERS" "$HEADSHA" "$SRCREV" >> "$CSV"
  for ((i = 8; i < NCOL; ++i)); do printf ',' >> "$CSV"; done
  printf ',"%s"\n' "${note//,/;}" >> "$CSV"
}

# ------------------------------------------------------------------- run_one
run_one () {
  local IDX="$1" CFG="$2"
  local SLUG STAMP OUT LOG RC HB HA waited

  SLUG="$(printf '%s' "$CFG" | tr -d ' =,')"
  STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
  # UNIQUE output root per run: run_campaign.sh exits 7 if summary.json is
  # already there, and a shared root makes a repeat measurement silently
  # re-read the previous run's summary.
  OUT="$HOME/k0-mok-${TAG}/${IDX}_${SLUG}_${STAMP}"
  LOG="$SCRATCH/${TAG}_${IDX}_${SLUG}_${STAMP}.log"

  say "=== [$IDX/${#CFGS[@]}] $CFG"

  waited=0
  while ! node_idle >/dev/null 2>&1; do
    if [ "$waited" -ge "$IDLE_WAIT" ]; then
      say "BUSY for ${waited}s -- skipping this point"
      node_idle
      stub_row "$IDX" "$CFG" "BUSY" "node busy ${waited}s: $(node_idle | tr '\n' ' ')"
      return 0
    fi
    [ "$waited" -eq 0 ] && say "node busy, waiting up to ${IDLE_WAIT}s ..."
    sleep 10; waited=$((waited + 10))
  done

  HB="$(hsaco_stamp)"
  say "mps hsaco BEFORE: $HB"
  mkdir -p "$OUT"

  cd "$K0" || return 1
  setsid -w timeout "$JOB_TIMEOUT" env \
    K0_MOK_ARMS="$ARMS" \
    K0_MOK_WARMUP_ITERS="$WARMUP" \
    K0_MOK_TIMED_ITERS="$TIMED" \
    K0_MOK_OUTPUT_ROOT="$OUT" \
    K0_MOK_RUN_TIMEOUT="$RUN_TIMEOUT" \
    K0_MPS_CFG="$CFG" \
    K0_MPS_TRACE="$TRACE" \
    K0_MPS_SOAK_ITERS="$SOAK" \
    K0_SYNTH_ROUTE="$ROUTE" \
    K0_SYNTH_SEED="$SEED" \
    bash "$CAMPAIGN" "${TAG}_${IDX}" "$PROCS" \
    > "$LOG" 2>&1 < /dev/null
  RC=$?

  HA="$(hsaco_stamp)"
  say "mps hsaco AFTER : $HA  (rc=$RC)"
  if [ "$HB" = "$HA" ]; then
    say "  -> unchanged: this run reused the cached mps_mega build (expected when HEAD did not move)"
  else
    say "  -> CHANGED: mps_mega was rebuilt for this run"
  fi

  python3 - "$IDX" "$CFG" "$ITERS" "$OUT/summary.json" "$LOG" "$CSV" \
             "$RC" "$HB" "$HA" "$OUT" "$HEADSHA" "$SRCREV" <<'PY'
import json, os, re, sys, time

(idx, cfg, iters, sj, logp, csvp, rc, hb, ha, outdir, headsha, srcrev) = sys.argv[1:13]
rc = int(rc)
NA = ""

# Every field is re-derived from scratch on every invocation. The old driver
# leaked the previous point's microseconds into a VOID row; a fresh process
# per point makes that impossible by construction.
def us(d, arm):
    try:
        return float(d["arm_p50_us"][arm]["median"])
    except Exception:
        return None

try:
    with open(sj) as fh:
        summary = json.load(fh)
except Exception:
    summary = None

p = f = m = None
if summary is not None:
    p, f, m = us(summary, "production"), us(summary, "pf6gm_mega"), us(summary, "mps_mega")

try:
    text = open(logp, errors="ignore").read()
except Exception:
    text = ""

def fmt(x, nd=1):
    return NA if x is None else f"{x:.{nd}f}"

def ratio(a, b):
    return None if (a is None or not b) else a / b

# --- gates -----------------------------------------------------------------
gm = re.search(r"\[MOK GATE\] mps_mega max_abs=([0-9.eE+-]+) relative=([0-9.eE+-]+) pass=(\w+)", text)
if gm:
    gate_pass, gate_abs, gate_rel = gm.group(3), gm.group(1), gm.group(2)
elif "[MOK GATE] mps_mega" in text:
    gate_pass, gate_abs, gate_rel = "MALFORMED", NA, NA
else:
    gate_pass, gate_abs, gate_rel = "VOID", NA, NA

cm = re.search(r"\[MARK\] control_fails=(\w+)", text)
control = cm.group(1) if cm else "VOID"

sm = re.search(r"\[MPS SOAK\] completed=(\d+)/(\d+) pperr=(\d+) pass=(\w+)", text)
if sm:
    soak, soak_ep = sm.group(4), f"{sm.group(1)}/{sm.group(2)}"
else:
    soak, soak_ep = ("MALFORMED" if "[MPS SOAK]" in text else "VOID"), NA

perrs = [int(v) for v in re.findall(r"pperr=(\d+)", text)]
pperr = str(max(perrs)) if perrs else "VOID"

# --- device stamps (1 tick = 0.01 us; the wall clock is 100 MHz here) -------
# Only populated when the cfg carries timestamps=1. NOTE these stamps are the
# max over the run and therefore come from the FINAL SOAK epoch, not from a
# timed iteration -- do not add them up and compare to arm_p50_us.
tsd = re.findall(
    r"\[MPS TS DELTA\] planM6=(-?\d+) M7=(-?\d+) combine=(-?\d+) "
    r"servicedrain=(-?\d+) m2_to_end=(-?\d+)", text)
d = [f"{int(v)/100.0:.1f}" for v in tsd[-1]] if tsd else [NA] * 5

tss = re.findall(r"\[MPS TS SPLIT\] plan_M3toM5=(-?\d+) M6=(-?\d+)", text)
s = [f"{int(v)/100.0:.1f}" for v in tss[-1]] if tss else [NA] * 2

spm = re.search(r"\[MPS SPIN\] chunk_poll success_max=(\d+) fail_max=(\d+)", text)
spin_ok, spin_fail = (spm.group(1), spm.group(2)) if spm else (NA, NA)

em = re.search(r"\[MOK SYNTHETIC EAGER\] status=(\S+) .*gate_ok=(\w+)", text)
eager_status, eager_gate = (em.group(1), em.group(2)) if em else ("VOID", "VOID")

# Authoritative build identity comes from the resolved symlink, not the log:
# mori prints nothing at all on a full cache hit.
mps_hsaco = ha.split("|")[-1] if "|" in ha else NA
jl = re.findall(r"jit/gfx950_mlx5/([0-9a-f]{6,})/k0pf6gm_mps_mega", text)
jit_log = jl[-1] if jl else NA

# --- verdict ---------------------------------------------------------------
notes = []
for pat, tagname in (
    (r"refusing to overwrite completed campaign", "campaign-dir-collision"),
    (r"GPU lease is already held", "gpu-lease-held"),
    (r"refusing to run: (\S+) is active", "foreign-job"),
    (r"refusing to run: only (\d+)/8 GPUs report 0% use", "gpus-not-idle"),
    (r"did not produce eight rank JSON files", "missing-rank-json"),
    (r"K0_MPS_CFG requires", "bad-mps-cfg"),
    (r"campaign blocked before timing", "blocked-pre-timing"),
):
    if re.search(pat, text):
        notes.append(tagname)

if rc != 0:
    status = f"FAIL:rc{rc}"
elif m is None:
    status = "FAIL:nosummary"
elif gate_pass != "True":
    status = f"FAIL:gate={gate_pass}"
elif soak != "True":
    status = f"FAIL:soak={soak}"
elif pperr != "0":
    status = f"FAIL:pperr={pperr}"
elif control != "True":
    status = f"FAIL:control={control}"
else:
    status = "OK"

if status != "OK" and not notes:
    tail = [ln.strip() for ln in text.splitlines() if ln.strip()][-1:]
    if tail:
        notes.append(tail[0][:120])

row = [
    idx, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    f'"{cfg}"', status, iters, headsha, srcrev,
    fmt(p), fmt(f), fmt(m),
    fmt(ratio(m, p), 4), fmt(ratio(m, f), 4), fmt(ratio(f, p), 4),
    gate_pass, gate_abs, gate_rel, control, pperr, soak, soak_ep,
    d[0], d[1], d[2], d[3], d[4], s[0], s[1],
    spin_ok, spin_fail, eager_status, eager_gate, mps_hsaco, jit_log,
    hb, ha, str(rc), os.path.basename(outdir), '"' + ";".join(notes) + '"',
]
line = ",".join(row)
print("SCREEN " + line, flush=True)
with open(csvp, "a") as fh:
    fh.write(line + "\n")
PY
  return 0
}

# ---------------------------------------------------------------------- loop
i=0
for CFG in "${CFGS[@]}"; do
  i=$((i + 1))
  run_one "$i" "$CFG" || say "run_one returned nonzero for '$CFG' -- continuing"
done

say "=== SCREEN $TAG DONE ==="
echo "mps hsaco @batch end: $(hsaco_stamp)"
echo
echo "===== $CSV ====="
cat "$CSV"
exit 0
