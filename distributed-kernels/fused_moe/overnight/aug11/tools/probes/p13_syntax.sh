#!/usr/bin/env bash
echo "===size/CR check==="
ls -l ~/.overnight-scripts/screen.sh
if grep -qU $'\r' ~/.overnight-scripts/screen.sh; then echo "HAS CR -- BAD"; else echo "no CR -- good"; fi
echo "===bash -n==="
bash -n ~/.overnight-scripts/screen.sh && echo "bash syntax OK"
echo "===python heredoc syntax==="
sed -n '/^  python3 - /,/^PY$/p' ~/.overnight-scripts/screen.sh | sed '1d;$d' > /tmp/screenparse.py
python3 -c "import ast,sys; ast.parse(open('/tmp/screenparse.py').read()); print('python syntax OK')"
echo "===usage guard==="
bash ~/.overnight-scripts/screen.sh ; echo "rc=$?"
exit 0
