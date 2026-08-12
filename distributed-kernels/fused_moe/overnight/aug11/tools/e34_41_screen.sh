#!/usr/bin/env bash
set -uo pipefail
echo "############ ~/tools/screen.sh ############"
cat $HOME/tools/screen.sh
echo
echo "############ other runners in ~/tools ############"
ls -l $HOME/tools/ | head -20
echo "===DONE==="
