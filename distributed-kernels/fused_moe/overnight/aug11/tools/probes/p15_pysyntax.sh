#!/usr/bin/env bash
awk "/<<'PY'/{f=1;next} /^PY\$/{f=0} f" ~/.overnight-scripts/screen.sh > /tmp/screenparse.py
echo "extracted lines: $(wc -l < /tmp/screenparse.py)"
python3 -c "import ast; ast.parse(open('/tmp/screenparse.py').read()); print('python syntax OK')"
echo "===dry-run the parser on an OLD log+summary (no GPU)==="
python3 /tmp/screenparse.py 0 "C=64,g=1,mode=2,flush_rows=16" "w1t1p1" \
  "$HOME/k0-mok-mps/verify_C64g1mode2flush_rows16/summary.json" \
  "$HOME/overnight-scratch/verify_C64g1mode2flush_rows16.log" \
  /tmp/selftest.csv 0 "x|y" "x|y" "$HOME/k0-mok-mps/verify_C64g1mode2flush_rows16"
echo "===dry-run on a timestamps=1 log==="
ls -d $HOME/k0-mok-mps/E19smallC_C16g1mode2flush_rows16timestamps1 2>/dev/null || ls -d $HOME/k0-mok-mps/* | head -5
exit 0
