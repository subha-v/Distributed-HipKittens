#!/usr/bin/env bash
S=~/overnight-scratch
L=$S/verify_C64g1mode2flush_rows16.log
echo "===lines 520-560==="; sed -n '520,560p' $L
echo; echo "===K0PF PROFILE / GATE / SPIN / EAGER verbatim==="
grep -n 'K0PF PROFILE\|K0PF GATE\|MPS SPIN\|MOK SYNTHETIC EAGER' $L
echo; echo "===where do us numbers live? grep us==="
grep -n 'us\b\|_us\|p50\|median' $L | grep -v aiter | head -30
echo; echo "===OUTPUT ROOT LAYOUT k0-mok-dec21a==="
find ~/k0-mok-dec21a -maxdepth 3 | head -60
echo; echo "===OUTPUT ROOT LAYOUT k0-mok-exp21==="
find ~/k0-mok-exp21 -maxdepth 3 | head -60
exit 0
