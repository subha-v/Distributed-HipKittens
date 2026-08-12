#!/usr/bin/env bash
ls -la ~/ovn/
echo '===SCRIPTS==='
for f in ~/ovn/*.sh; do
  echo "##### $f"
  cat "$f"
done
