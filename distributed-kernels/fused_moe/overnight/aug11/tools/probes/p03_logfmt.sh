#!/usr/bin/env bash
S=~/overnight-scratch
L=$S/verify_C64g1mode2flush_rows16.log
echo "===LOG $L size==="; wc -l $L
echo "===UNIQUE BRACKET TAGS in log==="
grep -o '^\[[A-Za-z0-9_ ]*\]' $L | sort | uniq -c | sort -rn | head -40
echo "===ANY BRACKET TAG anywhere in line==="
grep -oE '\[[A-Z][A-Z0-9 _]*\]' $L | sort | uniq -c | sort -rn | head -40
echo "===MARK lines==="; grep -n 'MARK' $L | head -20
echo "===MOK GATE lines==="; grep -n 'MOK GATE' $L | head -20
echo "===pperr lines==="; grep -n 'pperr' $L | head -10
echo "===SOAK lines==="; grep -n 'SOAK' $L | head -10
echo "===MPS TS lines==="; grep -n 'MPS TS' $L | head -20
echo "===arm_p50 lines==="; grep -n 'arm_p50\|p50_us\|ARM ' $L | head -20
echo "===control lines==="; grep -n 'control' $L | head -10
exit 0
