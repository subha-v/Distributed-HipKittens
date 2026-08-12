#!/usr/bin/env bash
echo "===cleanup stray tilde dir==="
ls -la "$HOME/~" 2>/dev/null && rm -rf "$HOME/~" && echo "removed stray ~ dir"
echo "===size/CR check==="
ls -l ~/.overnight-scripts/screen.sh
if grep -qU $'\r' ~/.overnight-scripts/screen.sh; then echo "HAS CR -- BAD"; else echo "no CR -- good"; fi
echo "===bash -n==="
bash -n ~/.overnight-scripts/screen.sh && echo "bash syntax OK"
echo "===python heredoc syntax==="
sed -n '/^  python3 - /,/^PY$/p' ~/.overnight-scripts/screen.sh | sed '1d;$d' > /tmp/screenparse.py
python3 -c "import ast; ast.parse(open('/tmp/screenparse.py').read()); print('python syntax OK')"
echo "===usage guard (no args)==="
bash ~/.overnight-scripts/screen.sh ; echo "rc=$?"
echo "===empty cfg guard==="
: > /tmp/empty.txt
bash ~/.overnight-scripts/screen.sh selftest /tmp/empty.txt ; echo "rc=$?"
exit 0
