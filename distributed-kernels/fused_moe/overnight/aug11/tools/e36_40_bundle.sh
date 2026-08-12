#!/usr/bin/env bash
# exp_36: assemble the raw evidence bundle for the repo.
set -u
E36="$HOME/e36"
RAW="$E36/raw"
K0="$HOME/amd-master/auto-gpu-kernel/k0_fused_moe"
rm -rf "$RAW"; mkdir -p "$RAW"

cp "$E36"/scratch/screen_e36*.csv "$RAW"/ 2>/dev/null
cp "$E36/collected.json" "$E36/sensitivity_grid.json" "$RAW"/ 2>/dev/null

# the auditable deviation from the stock harness
diff -u "$K0/benchmarks/mok_synthetic_prefill/run_campaign.sh" "$E36/rc_T.sh" \
  > "$RAW/rc_T.sh.diff"
diff -u "$E36/screen_src.sh" "$E36/screenT.sh" > "$RAW/screenT.sh.diff"

# the three refusals, quoted from the real logs
{
  echo "=== exp_36 refusal evidence (verbatim log excerpts) ==="
  echo
  echo "--- 1. routing-skew axis: K0_SYNTH_ROUTE=skewed_hot at T=4096 ---"
  L="$(ls -t "$E36"/scratch/e36route_1_*.log 2>/dev/null | head -1)"
  echo "log: $L"
  grep -h 'SYNTH_ROUTE' "$L" | head -4
  echo
  echo "source guards:"
  grep -n 'K0_SYNTH_ROUTE cannot be combined' \
    "$K0/prefill_opt/host/e004pf_k0pf_ab.py"
  grep -n 'synthetic routes are decode-only' -A 3 \
    "$K0/prefill_opt/host/e004pf_k0pf_ab.py" | head -6
  echo "families (no std parameter exists):"
  sed -n '17,24p' "$K0/prefill_opt/host/synthetic_routes.py"
  echo
  echo "--- 2. T <= 512 ---"
  L="$(ls -t "$E36"/scratch/e36T512_1_*.log 2>/dev/null | head -1)"
  echo "log: $L"
  grep -h 'exp_36 shape\|prefill-only' "$L" | head -3
  grep -n 'prefill-only' "$K0/prefill_opt/host/e004pf_k0pf_ab.py"
  echo
  echo "--- 3. T = 1024 and T = 2048: megakernel arms produce wrong output ---"
  for t in 1024 2048; do
    for pat in "e36T${t}_1_" "e36T${t}s_1_"; do
      L="$(ls -t "$E36"/scratch/${pat}*.log 2>/dev/null | head -1)"
      [ -n "$L" ] || continue
      echo "log: $L"
      grep -h 'exp_36 shape' "$L" | head -1
      grep -h '\[POISON\] eager\|\[MOK GATE\]\|control_fails' "$L" | head -14
      echo
    done
  done
  echo "--- 4. T = 4096 control, same driver, all gates green ---"
  L="$(ls -t "$E36"/scratch/e36T4096_1_*.log 2>/dev/null | head -1)"
  echo "log: $L"
  grep -h 'exp_36 shape\|\[MOK GATE\]\|control_fails\|MPS SOAK\|POISON\|MPS SPIN\|MPS TS' "$L" | head -20
} > "$RAW/refusal_evidence.txt" 2>&1

# node/lease state
{
  echo "host: $(hostname)   utc: $(date -u +%FT%TZ)"
  git -C "$HOME/Distributed-HipKittens" log --oneline -1
  echo "HEAD: $(git -C "$HOME/Distributed-HipKittens" rev-parse HEAD)"
  echo "dirty: [$(git -C "$HOME/Distributed-HipKittens" status --porcelain | head -3 | tr '\n' ';')]"
  echo "--- rocm-smi --showpids ---"
  /opt/rocm/bin/rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+[ \t]/ {print $1" "$2}'
  echo "--- gpu lock ---"
  ls -d /tmp/k0_mok_synthetic_gpu_lock 2>/dev/null || echo "no lock dir"
} > "$RAW/node_state.txt" 2>&1

ls -la "$RAW"
exit 0
