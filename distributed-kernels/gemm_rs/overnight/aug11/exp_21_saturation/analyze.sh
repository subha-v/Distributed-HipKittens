#!/usr/bin/env bash
# Validate the sweep JSON and extract knees. CPU-only; touches no GPU.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
EXP=$ON/aug11/exp_21_saturation
J=${1:-$EXP/saturation.json}
OUT=${2:-$EXP/knees.json}

echo "########## integrity of $(basename "$J") ##########"
docker exec dhk-gemmrs python3 -c "
import json, collections
d = json.load(open('$J'))
p = d['points']
print('  parses OK; points =', len(p))
print('  wall_minutes =', round(d['wall_minutes'], 2))
print('  generated    =', d['generated'])
print('  tick_rate_hz =', d['tick_rate_hz'], ' spread_pct =', round(d['tick_rate_spread_pct'], 4))
print('  tick samples =', [round(x/1e6, 4) for x in d['tick_rate_samples_hz']], 'MHz')
c = collections.Counter(x['mode'] for x in p)
print('  by mode      =', dict(c))
for m in 'abc':
    cs = sorted({x['ctas'] for x in p if x['mode'] == m})
    print(f'  mode {m} C     =', cs)
zero = [ (x['mode'], x['ctas'], x['overlay'], x['depth'], x['fanout'], x['protocol'])
         for x in p if not x['value'] ]
print('  zero-valued points =', len(zero), zero[:6])
bad = [x for x in p if x.get('checksum_ok') is False]
print('  checksum FAILURES  =', len(bad))
chk = [x for x in p if x.get('checksum_ok') is True]
print('  checksums verified =', len(chk))
folds = {x['checksum_fold'] for x in chk}
print('  distinct folds     =', len(folds))
"

echo
echo "########## knees ##########"
docker exec dhk-gemmrs python3 "$EXP/knees.py" --inp "$J" --out "$OUT"
