#!/usr/bin/env bash
# exp_24: unbundle the two genco outputs and diff the ISA evidence properly.
set -uo pipefail
docker exec subha_k1 bash -lc '
set -uo pipefail
HOME=/home/subvadla
OUTD=$HOME/exp24-stage/build
cd "$OUTD"
BUNDLER=/opt/rocm/llvm/bin/clang-offload-bundler
[ -x "$BUNDLER" ] || BUNDLER=/opt/rocm/lib/llvm/bin/clang-offload-bundler
OBJDUMP=/opt/rocm/llvm/bin/llvm-objdump
[ -x "$OBJDUMP" ] || OBJDUMP=/opt/rocm/lib/llvm/bin/llvm-objdump

for t in base exp24; do
  "$BUNDLER" --type=o --unbundle --inputs="$OUTD/${t}.hsaco" \
    --targets=hipv4-amdgcn-amd-amdhsa--gfx950 --outputs="$OUTD/${t}.elf" \
    || "$BUNDLER" --type=o --unbundle --input="$OUTD/${t}.hsaco" \
         --targets=hipv4-amdgcn-amd-amdhsa--gfx950 --output="$OUTD/${t}.elf"
  "$OBJDUMP" -d --mcpu=gfx950 "$OUTD/${t}.elf" > "$OUTD/${t}.s"
  echo "$t: elf=$(stat -c %s "$OUTD/${t}.elf") B, disasm=$(wc -l < "$OUTD/${t}.s") lines"
done

echo
echo "=== s_waitcnt vmcnt(N) census, N != 0 (the throttle literals)"
for t in base exp24; do
  echo "--- $t"
  grep -oE "s_waitcnt +vmcnt\(([1-9][0-9]*)\)" "$OUTD/${t}.s" | sort | uniq -c
done

echo
echo "=== ALL s_waitcnt forms (top 12 by count)"
for t in base exp24; do
  echo "--- $t"
  grep -oE "s_waitcnt +[a-z0-9_() ]+" "$OUTD/${t}.s" | sed "s/ *$//" | sort | uniq -c | sort -rn | head -12
done

echo
echo "=== MFMA census"
for t in base exp24; do
  echo "--- $t total=$(grep -cE "v_mfma" "$OUTD/${t}.s")"
  grep -oE "v_mfma_[a-z0-9_]+" "$OUTD/${t}.s" | sort | uniq -c
done

echo
echo "=== scratch ops"
for t in base exp24; do
  printf -- "--- %s scratch_load=%s scratch_store=%s buffer_(load|store)_dword.*off=%s\n" "$t" \
    "$(grep -cE "scratch_load" "$OUTD/${t}.s")" \
    "$(grep -cE "scratch_store" "$OUTD/${t}.s")" \
    "$(grep -cE "buffer_(load|store)_dword.* off" "$OUTD/${t}.s")"
done

echo
echo "=== atomics"
for t in base exp24; do
  printf -- "--- %s pk_add_bf16=%s cmpswap=%s\n" "$t" \
    "$(grep -cE "global_atomic_pk_add_bf16" "$OUTD/${t}.s")" \
    "$(grep -cE "atomic_cmpswap" "$OUTD/${t}.s")"
done

echo
echo "=== MFMA-span scratch check: min instruction distance from any scratch op"
echo "    to the nearest v_mfma, per build (0 => a scratch op sits inside an MFMA run)"
for t in base exp24; do
python3 - "$OUTD/${t}.s" "$t" <<'"'"'PY'"'"'
import re, sys
path, tag = sys.argv[1], sys.argv[2]
ins = []
for ln in open(path, errors="ignore"):
    m = re.match(r"^\s*[0-9a-f]+:\s+(?:[0-9a-f]{2} )+\s*(\S+)", ln)
    if m: ins.append(m.group(1))
    else:
        m2 = re.match(r"^\s*(v_mfma\S*|scratch_\S+|s_waitcnt|\S+)\s", ln)
mf = [i for i,x in enumerate(ins) if x.startswith("v_mfma")]
sc = [i for i,x in enumerate(ins) if x.startswith("scratch_")]
# contiguous MFMA runs (allow up to 40 non-mfma insns inside a "span")
spans = []
if mf:
    s = p = mf[0]
    for i in mf[1:]:
        if i - p <= 40: p = i
        else: spans.append((s,p)); s = p = i
    spans.append((s,p))
inside = [i for i in sc if any(a <= i <= b for a,b in spans)]
print(f"{tag}: insns={len(ins)} mfma={len(mf)} spans={len(spans)} scratch_ops={len(sc)} scratch_INSIDE_mfma_span={len(inside)}")
if spans:
    big = sorted(((b-a+1, a, b) for a,b in spans), reverse=True)[:4]
    print(f"{tag}: largest MFMA spans (len,start,end) = {big}")
if sc and mf:
    print(f"{tag}: min |scratch - mfma| distance = {min(abs(i-j) for i in sc for j in mf)}")
PY
done
'
exit 0
