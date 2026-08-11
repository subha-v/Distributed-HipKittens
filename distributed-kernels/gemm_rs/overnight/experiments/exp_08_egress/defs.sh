#!/usr/bin/env bash
# Print the verbatim rocprofv3 description of every counter exp_08 intends to
# use. A mislabelled counter is worse than no counter, so these go in result.md
# word for word. Read-only.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
AV=$ON/experiments/exp_08_egress/rocprofv3_list_avail.txt

WANT="TCC_EA0_WRREQ TCC_EA0_WRREQ_64B TCC_EA0_WRREQ_DRAM TCC_EA0_WRREQ_STALL
TCC_EA0_WRREQ_GMI_CREDIT_STALL TCC_EA0_WRREQ_DRAM_CREDIT_STALL
TCC_EA0_WRREQ_LEVEL TCC_EA0_RDREQ TCC_EA0_RDREQ_32B TCC_EA0_RDREQ_DRAM
TCC_EA0_WR_UNCACHED_32B TCC_WRITE TCC_READ TCC_REQ TCC_HIT TCC_MISS
TCC_WRITEBACK TCC_NORMAL_WRITEBACK TCC_ALL_TC_OP_WB_WRITEBACK
TCC_ALL_TC_OP_INV_EVICT TCC_NORMAL_EVICT TCC_TOO_MANY_EA_WRREQS_STALL
TCC_TAG_STALL TCC_BUSY TCC_CYCLE TCC_UC_REQ TCC_NC_REQ TCC_CC_REQ TCC_RW_REQ
TCC_STREAMING_REQ TCC_PROBE TCC_ATOMIC GRBM_GUI_ACTIVE SQ_WAIT_ANY"

for c in $WANT; do
  # The inventory repeats once per agent; one exact-name block is enough.
  line=$(grep -n "^Counter_Name        :	${c}\$" "$AV" | head -1 | cut -d: -f1)
  if [ -z "$line" ]; then echo "== $c : NOT FOUND"; continue; fi
  echo "== $c"
  sed -n "$((line+1)),$((line+3))p" "$AV" | sed 's/^/   /'
done
