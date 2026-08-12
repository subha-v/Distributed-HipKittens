#!/usr/bin/env bash
S=~/overnight-scratch
echo "===CSV screen_E20confirm2.csv==="; cat $S/screen_E20confirm2.csv
echo; echo "===CSV screen_E20attr.csv==="; cat $S/screen_E20attr.csv
echo; echo "===CSV screen_verify.csv==="; cat $S/screen_verify.csv
echo; echo "===OUT screen_E20confirm2.out==="; cat $S/screen_E20confirm2.out
echo; echo "===OUT screen_verify.out==="; cat $S/screen_verify.out
echo; echo "===CSV screen_E19smallC.csv==="; cat $S/screen_E19smallC.csv
echo; echo "===ALL k0-mok dirs (full)==="; ls -d ~/k0-mok-* | head -200
echo; echo "===exp21/dec21 dirs==="; ls -d ~/k0-mok-*exp21* ~/k0-mok-*dec21* 2>&1 | head -40
exit 0
