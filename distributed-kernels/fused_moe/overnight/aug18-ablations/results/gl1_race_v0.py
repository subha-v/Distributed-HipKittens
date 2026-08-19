import torch, importlib
torch.manual_seed(0)
dev="cuda:0"
T,D,E,FF = 16384,7168,32,2048
KG = D//128            # 56 k-groups
NG13 = (2*FF)//128     # 32 n-groups for w13
NG2  = D//128          # 56 n-groups for w2
KG2  = FF//128         # 16 k-groups for w2
BLK = 32               # guess: sorted rows per expert-id entry (kMrows, G=1)
n2 = importlib.import_module("vllm.model_executor.layers.fused_moe.experts.pf4h_integration.native.k0_n2")

# ---- fp8 input, token-major per-128-group scales
h = (torch.randn(T,D,dtype=torch.float32,device=dev)*0.02)
hg = h.view(T,KG,128)
sc = hg.abs().amax(-1).clamp(min=1e-6)/448.0          # [T,KG] fp32
a_fp8 = (hg/sc.unsqueeze(-1)).to(torch.float8_e4m3fn) # [T,KG,128]
a_bf16view = a_fp8.view(torch.uint8).view(T,D).view(torch.bfloat16)  # [T,D/2] bf16 view
sc_dst = sc.contiguous()                               # [T,56]

# ---- weights: random fp8 bytes + blockscales (timing-first; layout=AITER TBD)
w1 = (torch.randn(E,2*FF,D,dtype=torch.float32,device=dev)*0.05).to(torch.float8_e4m3fn)
w1_bf16 = w1.view(torch.uint8).view(torch.bfloat16).view(E*2*FF,D//2)
w1_scale = torch.full((E,NG13,KG),1.0,dtype=torch.float32,device=dev)
w2 = (torch.randn(E,D,FF,dtype=torch.float32,device=dev)*0.05).to(torch.float8_e4m3fn)
w2_bf16 = w2.view(torch.uint8).view(torch.bfloat16).view(E*D,FF//2)
w2_scale = torch.full((E,NG2,KG2),1.0,dtype=torch.float32,device=dev)

# ---- uniform top-1 sort: token t -> expert t%E, grouped contiguous
perm = torch.argsort(torch.arange(T,device=dev)%E, stable=True).to(torch.int32)
sti = perm.contiguous()                                # [T] sorted token ids
sei = (torch.arange(T//BLK,device=dev,dtype=torch.int32)//((T//E)//BLK)).contiguous()
nvi = torch.tensor([T,T],dtype=torch.int32,device=dev)
swt = torch.ones(T,dtype=torch.float32,device=dev)

rowcap = T
a2q = torch.empty(rowcap,1024,dtype=torch.bfloat16,device=dev)   # fp8 [rowcap,2048] bf16-view
dq2 = torch.empty(rowcap,16,dtype=torch.float32,device=dev)
out = torch.zeros(T,D,dtype=torch.bfloat16,device=dev)
stream = torch.cuda.current_stream().cuda_stream

def p1(): n2.n2_phase1(a_bf16view, sc_dst, w1_bf16, w1_scale, sti, sei, nvi, a2q, dq2, stream)
def p2(): n2.n2_phase2(a2q, dq2, w2_bf16, w2_scale, sti, swt, sei, nvi, out, stream)
try:
    p1(); torch.cuda.synchronize(); print("phase1 OK")
except Exception as e: print("phase1:", repr(e)[:400])
try:
    p2(); torch.cuda.synchronize(); print("phase2 OK")
except Exception as e: print("phase2:", repr(e)[:400])

def timeit(fn, tag, reps=20, warm=5):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    ts=[]
    for _ in range(reps):
        a,b = torch.cuda.Event(True),torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b)*1e3)
    ts.sort(); print(f"{tag}: median={ts[len(ts)//2]:.1f} us min={ts[0]:.1f} max={ts[-1]:.1f}")
    return ts[len(ts)//2]

t_ours = timeit(lambda: (p1(), p2()), "OURS_n2_phase1+2_fp8")

import aiter
from aiter.fused_moe import fused_moe as afm
from aiter import QuantType
hb = h.to(torch.bfloat16)
ids1 = (torch.arange(T,device=dev)%E).view(T,1).to(torch.int32)
tw1 = torch.ones(T,1,dtype=torch.float32,device=dev)
w1s3 = w1_scale.view(E,NG13,KG)
w2s3 = w2_scale.view(E,NG2,KG2)
try:
    o = afm(hb, w1.view(E,2*FF,D), w2.view(E,D,FF), tw1, ids1,
            quant_type=QuantType.per_128x128, w1_scale=w1s3, w2_scale=w2s3)
    torch.cuda.synchronize()
    t_a = timeit(lambda: afm(hb, w1.view(E,2*FF,D), w2.view(E,D,FF), tw1, ids1,
            quant_type=QuantType.per_128x128, w1_scale=w1s3, w2_scale=w2s3),
            "AITER_fused_moe_fp8_blockscale")
    print(f"RACE: ours={t_ours:.1f} aiter_fp8={t_a:.1f} delta={t_a-t_ours:+.1f} ratio={t_ours/t_a:.3f}")
except Exception as e:
    print("aiter fp8 fail:", repr(e)[:400])

# ---- verification 1: our output vs float reference (unit weight scales)
r = torch.randperm(T, device=dev)[:4]
af = a_fp8.view(T,D).to(torch.float32) * sc.repeat_interleave(128,dim=1)
w1f = w1.view(E,2*FF,D).to(torch.float32); w2f = w2.view(E,D,FF).to(torch.float32)
ok = True
for t in r.tolist():
    e = t % E
    g = af[t] @ w1f[e,:FF].T; u = af[t] @ w1f[e,FF:].T
    act = torch.nn.functional.silu(g) * u
    # phase1 requantizes act to fp8 per-128-group before phase2 (a2q/dq2)
    ag = act.view(FF//128,128); s2 = ag.abs().amax(-1).clamp(min=1e-6)/448.0
    aq = ((ag/s2.unsqueeze(-1)).to(torch.float8_e4m3fn).to(torch.float32)*s2.unsqueeze(-1)).view(FF)
    ref = aq @ w2f[e].T
    got = out[t].to(torch.float32)
    rel = (got-ref).abs().max()/(ref.abs().max()+1e-6)
    ok &= bool(rel < 0.05)
    print(f"row {t}: relmax={rel:.4f}")
print("CORRECTNESS:", "PASS" if ok else "FAIL")

# ---- verification 2: AITER with its own shuffled-weight layout
try:
    from aiter.ops.shuffle import shuffle_weight
    w1s = shuffle_weight(w1.view(E,2*FF,D), layout=(16,16))
    w2s = shuffle_weight(w2.view(E,D,FF), layout=(16,16))
    o = afm(hb, w1s, w2s, tw1, ids1, quant_type=QuantType.per_128x128,
            w1_scale=w1s3, w2_scale=w2s3)
    torch.cuda.synchronize()
    timeit(lambda: afm(hb, w1s, w2s, tw1, ids1, quant_type=QuantType.per_128x128,
            w1_scale=w1s3, w2_scale=w2s3), "AITER_fp8_SHUFFLED")
except Exception as e:
    print("shuffled aiter fail:", repr(e)[:300])

# ---- verification 3: OUR arm with AITER-shuffled weight layout
w1_bf16_s = w1s.view(torch.uint8).view(torch.bfloat16).reshape(E*2*FF, D//2)
w2_bf16_s = w2s.view(torch.uint8).view(torch.bfloat16).reshape(E*D, FF//2)
out.zero_()
def p1s(): n2.n2_phase1(a_bf16view, sc_dst, w1_bf16_s, w1_scale, sti, sei, nvi, a2q, dq2, stream)
def p2s(): n2.n2_phase2(a2q, dq2, w2_bf16_s, w2_scale, sti, swt, sei, nvi, out, stream)
p1s(); p2s(); torch.cuda.synchronize()
ok = True
for t in r.tolist():
    e = t % E
    g = af[t] @ w1f[e,:FF].T; u = af[t] @ w1f[e,FF:].T
    act = torch.nn.functional.silu(g) * u
    ag = act.view(FF//128,128); s2 = ag.abs().amax(-1).clamp(min=1e-6)/448.0
    aq = ((ag/s2.unsqueeze(-1)).to(torch.float8_e4m3fn).to(torch.float32)*s2.unsqueeze(-1)).view(FF)
    ref = aq @ w2f[e].T
    got = out[t].to(torch.float32)
    rel = (got-ref).abs().max()/(ref.abs().max()+1e-6)
    ok &= bool(rel < 0.05)
    print(f"row {t}: relmax={rel:.4f}")
print("CORRECTNESS(shuffled):", "PASS" if ok else "FAIL")

import torch.nn.functional as F
print("---- distribution-level check (shuffled) ----")
cos_all, mre_all = [], []
for t in r.tolist():
    e = t % E
    g = af[t] @ w1f[e,:FF].T; u = af[t] @ w1f[e,FF:].T
    act = torch.nn.functional.silu(g) * u
    ag = act.view(FF//128,128); s2 = ag.abs().amax(-1).clamp(min=1e-6)/448.0
    aq = ((ag/s2.unsqueeze(-1)).to(torch.float8_e4m3fn).to(torch.float32)*s2.unsqueeze(-1)).view(FF)
    ref = aq @ w2f[e].T
    got = out[t].to(torch.float32)
    cos = F.cosine_similarity(got.unsqueeze(0), ref.unsqueeze(0)).item()
    mre = ((got-ref).abs().mean()/(ref.abs().mean()+1e-9)).item()
    cos_all.append(cos); mre_all.append(mre)
    print(f"row {t}: cos={cos:.5f} mean_rel={mre:.4f}")
print("VERDICT:", "PASS" if min(cos_all)>0.99 and max(mre_all)<0.10 else "FAIL")
# timing at shuffled layout (the honest config)
timeit(lambda: (p1s(), p2s()), "OURS_shuffled_fp8")
