#!/usr/bin/env bash
# exp_22 CPU gate: rebuild, resource remark, ISA census, grid census, and a
# summarizer self-test on synthetic points. NO GPU WORK ANYWHERE IN HERE.
set -uo pipefail
E22=/home/subvadla/e22
OUT=$E22/out
mkdir -p "$OUT"

echo "############ BUILD + RESOURCE + ISA ############"
docker exec subha_k1 bash "$E22/src/build_in.sh" isa > "$OUT/gate_build.log" 2>&1
grep -E 'exit=|CPU_GATE|ISA_GATE|^note:|sha256|E22_SRC_REV' "$OUT/gate_build.log"

echo
echo "############ GRID CENSUS ############"
for t in quick plan full; do
  docker exec subha_k1 "$OUT/e22_saturation" --list --tier "$t" \
      > "$OUT/list_$t.txt" 2>&1
  n=$(grep -vc '^\[e22\]' "$OUT/list_$t.txt")
  echo "tier=$t points=$n"
  awk -F'|' '{print $1}' "$OUT/list_$t.txt" | grep -v '^\[e22\]' | sort | uniq -c \
    | awk '{printf "    %-8s %s\n", $2, $1}'
  awk -F'|' '{print $5}' "$OUT/list_$t.txt" | grep -v '^\[e22' | sort | uniq -c \
    | awk '{printf "    %-14s %s\n", $2, $1}'
done

echo
echo "############ DUPLICATE KEY CHECK ############"
for t in quick plan full; do
  d=$(grep -v '^\[e22\]' "$OUT/list_$t.txt" | sort | uniq -d | wc -l)
  echo "tier=$t duplicate_keys=$d"
done

echo
echo "############ SUMMARIZER SELF-TEST ############"
python3 "$E22/src/selftest_summarize.py"

echo
echo "############ END-TO-END DRY RUN (no GPU touched) ############"
# --dry-run emits one synthetic record per point through the REAL emit() path,
# so this validates the exact on-disk format against summarize.py, plus the
# resume-by-key logic, without a device.
rm -f "$OUT/dry.jsonl" "$OUT/dry.json"
docker exec subha_k1 "$OUT/e22_saturation" --dry-run --tier plan \
    --out "$OUT/dry.jsonl" > "$OUT/dry_run.log" 2>&1
echo "emitted=$(wc -l < "$OUT/dry.jsonl")  (expect 244)"
python3 -c 'import json,sys; [json.loads(l) for l in open(sys.argv[1])]; print("every line is valid JSON")' "$OUT/dry.jsonl"
python3 "$E22/src/summarize.py" "$OUT/dry.jsonl" "$OUT/dry.json" --print \
    | head -30
echo "--- resume check: rerun must add 0 points ---"
docker exec subha_k1 "$OUT/e22_saturation" --dry-run --tier plan \
    --out "$OUT/dry.jsonl" >> "$OUT/dry_run.log" 2>&1
echo "after_rerun=$(wc -l < "$OUT/dry.jsonl")  (must still be 244)"
grep -c dry_run "$OUT/dry.jsonl" | sed 's/^/records tagged dry_run: /'

echo
echo "############ DONE ############"
