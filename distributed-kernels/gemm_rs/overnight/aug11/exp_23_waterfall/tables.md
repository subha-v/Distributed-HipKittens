draws: 4 (2 forward, 2 reversed construction)

### Waterfall: rung x shape, best / median us

| rung | 64x7168x18432 | 512x4096x12288 | 2048x2880x2880 | 4096x4096x4096 | 8192x4096x14336 | 8192x8192x29568 | geomean |
|---|---|---|---|---|---|---|---|
| a | 61.3 / 63.5 | 64.3 / 65.8 | 83.3 / 85.0 | 197.7 / 199.0 | 743.9 / 751.8 | 2595.0 / 2616.3 | **223.65 / 227.62** |
| b | 62.1 / 64.0 | 64.6 / 66.0 | 85.0 / 85.7 | 197.9 / 198.9 | 634.1 / 652.5 | 1801.8 / 1840.8 | **206.29 / 210.28** |
| c | 60.4 / 62.7 | 64.8 / 66.0 | 83.6 / 84.5 | 198.0 / 199.1 | 624.9 / 642.5 | 1609.8 / 1635.7 | **200.60 / 204.52** |
| null | 60.0 / 61.8 | 64.6 / 65.3 | 83.9 / 84.5 | 197.5 / 198.8 | 625.6 / 657.9 | 1601.6 / 1640.9 | **200.18 / 204.50** |

### Rung deltas (median gain % vs previous rung, widened null floor, verdict)

| shape | mechanism active | null floor % | a->b | b->c |
|---|---|---|---|---|
| 64x7168x18432 | a->b False, b->c False | 2.82 (1-pair: 2.82) | -0.60% [-2.10,+0.65] p=0.0571 **UNRESOLVED** | +1.74% [+0.69,+3.33] p=0.3429 **UNRESOLVED** |
| 512x4096x12288 | a->b False, b->c False | 2.09 (1-pair: 2.09) | -0.29% [-0.66,+0.36] p=0.0571 **UNRESOLVED** | -0.08% [-0.39,+0.24] p=0.1000 **UNRESOLVED** |
| 2048x2880x2880 | a->b False, b->c False | 1.74 (1-pair: 0.34) | -0.64% [-2.76,+0.35] p=0.1714 **UNRESOLVED** | +1.41% [+0.38,+2.52] p=0.0143 **UNRESOLVED** |
| 4096x4096x4096 | a->b False, b->c False | 0.85 (1-pair: 0.28) | -0.02% [-0.57,+0.22] p=0.2429 **UNRESOLVED** | +0.05% [-0.70,+0.74] p=0.4429 **UNRESOLVED** |
| 8192x4096x14336 | a->b True, b->c False | 4.97 (1-pair: 3.98) | +15.11% [+12.57,+19.07] p=0.0143 **RESOLVED faster** | +0.79% [-1.27,+4.70] p=0.1714 **UNRESOLVED** |
| 8192x8192x29568 | a->b True, b->c True | 4.44 (1-pair: 1.10) | +42.11% [+40.48,+44.79] p=0.0143 **RESOLVED faster** | +12.22% [+10.98,+12.98] p=0.0143 **RESOLVED faster** |

### NR sweep at the rung-c config (median gain % vs shipped NR)

| shape | shipped NR | NR=8 | NR=16 | NR=32 | NR=48 |
|---|---|---|---|---|---|
| 64x7168x18432 | 56 | -51.48% resolved slower | -28.11% resolved slower | -7.96% resolved slower | +0.10% unresolved |
| 512x4096x12288 | 32 | -25.00% resolved slower | -8.94% resolved slower | +0.69% null **(=shipped, null pair)** | +0.35% unresolved |
| 2048x2880x2880 | 32 | -27.26% resolved slower | -11.04% resolved slower | -0.04% null **(=shipped, null pair)** | +0.48% unresolved |
| 4096x4096x4096 | 32 | -24.55% resolved slower | -9.80% resolved slower | -0.12% null **(=shipped, null pair)** | +0.34% unresolved |
| 8192x4096x14336 | 32 | -13.75% resolved slower | -5.01% unresolved | -2.37% null **(=shipped, null pair)** | +0.44% unresolved |
| 8192x8192x29568 | 48 | -18.80% resolved slower | -8.87% resolved slower | -3.53% resolved slower | +3.92% null **(=shipped, null pair)** |

### NR sweep geomeans

| arm | geomean best us | geomean median us | vs rung c |
|---|---|---|---|
| a | 223.65 | 227.62 | 1.1129 |
| b | 206.29 | 210.28 | 1.0282 |
| c | 200.60 | 204.52 | 1.0000 |
| null | 200.18 | 204.50 | 0.9999 |
| nr8 | 281.92 | 284.66 | 1.3918 |
| nr16 | 230.64 | 233.37 | 1.1411 |
| nr32 | 205.28 | 209.11 | 1.0225 |
| nr48 | 198.81 | 202.82 | 0.9917 |
