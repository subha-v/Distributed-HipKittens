#!/usr/bin/env bash
# exp_27 batch 3: CONTROL-LAST (arm 0), pinned d3d22ce4, n=5.
# Mandatory, not optional: the M6 stamp drifts ~50 us between batches on a
# PROVABLY IDENTICAL binary (exp_26's control read 2581.7 at 06:35Z; batch 1, the
# .text-identical ratchet, read 2531.6 at 07:07Z). One control cannot bound that.
bash "$HOME/tools/e27_batch.sh" e27b3_ctl d3d22ce4 0 5
