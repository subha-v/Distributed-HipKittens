#!/usr/bin/env bash
set -uo pipefail
echo "== ~/tools =="; ls -l "$HOME/tools/" 2>/dev/null
echo "== stray tilde dir from the bad push? =="; ls -ld "$HOME/~" 2>/dev/null && find "$HOME/~" -type f 2>/dev/null
echo "== syntax + the two patched spots =="
bash -n "$HOME/tools/screen.sh" && echo "bash -n OK"
grep -n 'MPS SOAK\] completed\|selftest_dead\|poison_alive\|VOID:selftest' "$HOME/tools/screen.sh"
echo "== sha =="; sha256sum "$HOME/tools/screen.sh" | cut -c1-16
exit 0
