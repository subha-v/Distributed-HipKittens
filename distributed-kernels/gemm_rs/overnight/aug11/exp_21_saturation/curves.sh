#!/usr/bin/env bash
# Print the mode-a and mode-b curves, the mode-c knee neighbourhood, and the GEMM-side
# interference against the C-matched reserve control. CPU-only; touches no GPU.
set -uo pipefail
EXP=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/aug11/exp_21_saturation
J=${1:-$EXP/saturation.json}

docker exec dhk-gemmrs python3 -c "
import json
d = json.load(open('$J')); P = d['points']
def sel(**kw): return sorted([p for p in P if all(p[k]==v for k,v in kw.items())], key=lambda p:p['ctas'])

print('=== ceilings the node itself reports ===')
c = d['ceilings']
print('  HBM analytic :', round(c['hbm_peak_gbps']['value'],1), 'GB/s')
print('    source     :', c['hbm_peak_gbps']['source'])
pr = d['node']['device_props']
print('  arch/CU      :', pr['gcn_arch_name'], pr['multi_processor_count'], 'CU')
print('  core clk kHz :', pr.get('clock_rate_khz'))

print()
print('=== mode a: MFMA vs CTA count (isolated) ===')
a = sel(mode='a', overlay='isolated')
base = a[0]['value']/a[0]['ctas']
for p in a:
    per = p['value']/p['ctas']
    print(f\"  C={p['ctas']:<4} {p['value']:8.2f} TFLOPS   per-CTA {per:6.4f}  \"
          f\"vs C={a[0]['ctas']}: {per/base*100:6.2f}%   span {p['res_span_us']/1e3:7.2f} ms\")

print()
print('=== mode b: REDV=1 reduce, GB/s vs CTA count ===')
for ov in ('isolated','concurrent'):
    for p in sel(mode='b', overlay=ov):
        sd = p.get('gemm_slowdown')
        print(f\"  {ov:<11} C={p['ctas']:<4} {p['value']:9.1f} GB/s\"
              + (f\"   gemm {p['concurrent_gemm_tflops']:6.1f} TF  slowdown {sd:.3f}\" if sd else ''))
    print()
print('  mode-b reserve control (GEMM-only, C-matched):')
for p in sel(mode='b', overlay='reserve_control'):
    print(f\"    C={p['ctas']:<4} {p['value']:8.1f} TFLOPS\")

print()
print('=== mode c: full C curve, proto off vs on, isolated vs concurrent ===')
for fan in ('single','rr7'):
    for dep in (0,1,4,8):
        print(f'  -- fanout={fan} depth={dep} --')
        i0 = {p['ctas']:p for p in sel(mode='c',overlay='isolated',fanout=fan,depth=dep,protocol=False)}
        i1 = {p['ctas']:p for p in sel(mode='c',overlay='isolated',fanout=fan,depth=dep,protocol=True)}
        c0 = {p['ctas']:p for p in sel(mode='c',overlay='concurrent',fanout=fan,depth=dep,protocol=False)}
        c1 = {p['ctas']:p for p in sel(mode='c',overlay='concurrent',fanout=fan,depth=dep,protocol=True)}
        print('     C   iso_p0   iso_p1   con_p0   con_p1   p1/p0   con/iso  gemmTF  slowdown')
        for C in sorted(i0):
            r = lambda m: (m[C]['value'] if C in m else 0.0)
            sd = c0[C].get('gemm_slowdown') if C in c0 else None
            g = c0[C]['concurrent_gemm_tflops'] if C in c0 else None
            print(f\"  {C:4} {r(i0):8.2f} {r(i1):8.2f} {r(c0):8.2f} {r(c1):8.2f}\"
                  f\"  {(r(i1)/r(i0) if r(i0) else 0):6.3f}  {(r(c0)/r(i0) if r(i0) else 0):7.3f}\"
                  f\"  {(g or 0):6.1f}  {(sd or 0):7.3f}\")
print()
print('  mode-c reserve control (GEMM-only on 304-C CTAs, C-matched, deadline-matched):')
for p in sel(mode='c', overlay='reserve_control'):
    print(f\"    C={p['ctas']:<4} {p['value']:8.1f} TFLOPS  span {p['res_span_us']/1e3:6.2f} ms\")
"
