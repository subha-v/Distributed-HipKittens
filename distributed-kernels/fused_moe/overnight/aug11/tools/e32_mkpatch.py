#!/usr/bin/env python3
"""exp_32: build the poison patch mechanically so the unified diff is guaranteed
to apply. Operates on a LOCAL scp'd copy of the node's harness files; it never
touches the node. Usage:

    python e32_mkpatch.py <tempdir>

Writes <tempdir>/mod/... alongside <tempdir>/<orig paths> so that
`git diff --no-index -U3` produces the deliverable diff.
"""
import os
import sys

TD = sys.argv[1]
AB = os.path.join(TD, "prefill_opt", "host", "e004pf_k0pf_ab.py")
RC = os.path.join(TD, "benchmarks", "mok_synthetic_prefill", "run_campaign.sh")


def edit(text, old, new, label):
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"anchor {label!r} occurs {n} times, expected 1")
    return text.replace(old, new)


# ---------------------------------------------------------------- ab.py
src = open(AB, encoding="utf-8", newline="").read()

src = edit(
    src,
    '    raise ValueError("MoK correctness tolerances must be nonnegative")\n',
    '    raise ValueError("MoK correctness tolerances must be nonnegative")\n'
    '# exp_32 gate hardening. `cand_out` is cleared once per gate episode but NEVER\n'
    '# between epochs, and the MoK corpus feeds identical input and identical routing\n'
    '# every iteration, so a row a candidate fails to write returns the previous\n'
    "# epoch's bit-identical correct answer -- invisible to [MOK GATE], to\n"
    '# combine_bit_exact, and to 600 soak epochs. Poison with a value that cannot\n'
    "# survive a correct run; correctness.py's `nonfinite == 0` gate then catches it.\n"
    'MOK_POISON_OUT = os.environ.get("K0_MOK_POISON_OUT", "1").strip() not in ("", "0")\n'
    '# bf16 quiet NaN, payload 0x55. NOT 0x7FC0 (what a computed 0/0 rounds to) and\n'
    '# NOT 0x7F80 (inf), so a survivor is distinguishable from a kernel-produced NaN.\n'
    'MOK_POISON_I16 = 0x7FD5\n',
    "knob",
)

src = edit(
    src,
    'cand_out = torch.zeros((T, H), dtype=torch.bfloat16, device=dev)\n',
    'cand_out = torch.zeros((T, H), dtype=torch.bfloat16, device=dev)\n'
    '\n'
    '\n'
    'def _poison_out():\n'
    '    """exp_32: fill the shared candidate output with a recognisable bf16 NaN.\n'
    '    Called ONLY from untimed regions (eager gate, soak, post-timing verify)."""\n'
    '    if MOK_POISON_OUT:\n'
    '        cand_out.view(torch.int16).fill_(MOK_POISON_I16)\n'
    '    else:\n'
    '        cand_out.zero_()\n'
    '\n'
    '\n'
    'def _poison_count():\n'
    '    """Elements still holding the poison pattern. Zero in a correct run."""\n'
    '    if not MOK_POISON_OUT:\n'
    '        return 0\n'
    '    return int((cand_out.view(torch.int16) == MOK_POISON_I16).sum().item())\n'
    '\n'
    '\n'
    'def _poison_rows(limit=16):\n'
    '    _bad = (cand_out.view(torch.int16) == MOK_POISON_I16).any(dim=1)\n'
    '    return torch.nonzero(_bad, as_tuple=False).flatten()[:limit].cpu().tolist()\n'
    '\n'
    '\n'
    'def _poison_report(tag, name):\n'
    '    """Print and return the survivor count. Nonzero means the arm left a row of\n'
    '    `out` unwritten this epoch -- the exact bug class a stale `out` hides."""\n'
    '    _n = _poison_count()\n'
    '    if _n:\n'
    '        print(f"[POISON] {tag} arm={name} rank={rank} survivors={_n} "\n'
    '              f"first_rows={_poison_rows()}", flush=True)\n'
    '    elif rank == 0:\n'
    '        print(f"[POISON] {tag} arm={name} survivors=0", flush=True)\n'
    '    return _n\n'
    '\n',
    "helpers",
)

src = edit(
    src,
    'R["eager"] = {}\n_eager_all_local = True\nfor name, body, buf in ARMS:\n',
    'R["eager"] = {}\n_eager_all_local = True\n'
    '# exp_32 P0: the [MOK GATE] loop below runs AFTER this loop has finished, so for\n'
    '# every candidate arm it re-reads the SAME cand_out bytes -- whichever candidate\n'
    '# ran last (_obuf returns the identical object for all of them). Compute each\n'
    "# arm's MoK gate HERE, while its own bytes are still live, and have that loop\n"
    "# consume the stash. Without this, P1's poison for a given arm is only checked\n"
    '# when that arm happens to be last in the rotated order.\n'
    '_mok_eager_stash = {}\n'
    'for name, body, buf in ARMS:\n',
    "eager-stash-decl",
)

src = edit(
    src,
    '    if name not in _PROD_LIKE: cand_out.zero_()   # FAIRNESS: never zero '
    "production's own combine-output view\n"
    '    perr.zero_(); pperr.zero_(); torch.cuda.synchronize(); body(sp()); '
    'torch.cuda.synchronize()\n'
    '    R["eager"][name] = gate(_obuf(name))',
    '    if name not in _PROD_LIKE: _poison_out()   # FAIRNESS: never touch '
    "production's own combine-output view\n"
    '    perr.zero_(); pperr.zero_(); torch.cuda.synchronize(); body(sp()); '
    'torch.cuda.synchronize()\n'
    '    if name not in _PROD_LIKE:\n'
    '        R.setdefault("poison", {})[name] = _poison_report("eager", name)\n'
    '    if K0_BENCHMARK_PROTOCOL == "mok_eager":\n'
    '        _mok_eager_stash[name] = mok_gate(_obuf(name))\n'
    '    R["eager"][name] = gate(_obuf(name))',
    "eager-poison",
)

src = edit(
    src,
    '        _mok_stats = mok_gate(_obuf(name))\n',
    '        _mok_stats = _mok_eager_stash[name]   # exp_32 P0: that arm\'s OWN bytes\n',
    "mok-gate-stash-use",
)

src = edit(
    src,
    '        _mps_soak_error = 0\n'
    '        _mps_soak_completed = 0\n'
    '        for _mps_epoch in range(_mps_soak_iters):\n'
    '            _mps_body(sp())\n'
    '            torch.cuda.synchronize()\n'
    '            _mps_local_error = int(pperr.item())\n'
    '            _mps_error_tensor = torch.tensor(\n'
    '                _mps_local_error, dtype=torch.int32, device="cuda"\n'
    '            )\n'
    '            dist.all_reduce(_mps_error_tensor, op=dist.ReduceOp.MAX)\n'
    '            _mps_soak_error = int(_mps_error_tensor.item())\n'
    '            _mps_soak_completed = _mps_epoch + 1\n'
    '            if _mps_soak_error != 0:\n'
    '                break\n',
    '        _mps_soak_error = 0\n'
    '        _mps_soak_completed = 0\n'
    '        _mps_soak_poison = 0\n'
    '        _mps_soak_poison_epoch = -1\n'
    '        for _mps_epoch in range(_mps_soak_iters):\n'
    '            # exp_32 P2: re-poison before EVERY epoch. The soak is fully untimed\n'
    '            # (no HIP events anywhere in this loop), so this turns one\n'
    '            # end-of-soak check into 600 independent single-epoch coverage trials.\n'
    '            _poison_out()\n'
    '            _mps_body(sp())\n'
    '            torch.cuda.synchronize()\n'
    '            _mps_local_error = int(pperr.item())\n'
    '            _mps_local_poison = _poison_count()\n'
    '            _mps_error_tensor = torch.tensor(\n'
    '                [_mps_local_error, _mps_local_poison],\n'
    '                dtype=torch.int32, device="cuda",\n'
    '            )\n'
    '            dist.all_reduce(_mps_error_tensor, op=dist.ReduceOp.MAX)\n'
    '            _mps_soak_error = int(_mps_error_tensor[0].item())\n'
    '            _mps_soak_poison = int(_mps_error_tensor[1].item())\n'
    '            _mps_soak_completed = _mps_epoch + 1\n'
    '            if _mps_soak_error != 0 or _mps_soak_poison != 0:\n'
    '                if _mps_soak_poison != 0:\n'
    '                    _mps_soak_poison_epoch = _mps_epoch\n'
    '                    _poison_report("soak", "mps_mega")\n'
    '                break\n',
    "soak-loop",
)

src = edit(
    src,
    '            "pperr": _mps_soak_error,\n'
    '            "correctness": _mps_soak_gate,\n'
    '            "pass": bool(\n'
    '                _mps_soak_completed == _mps_soak_iters\n'
    '                and _mps_soak_error == 0\n'
    '                and _mps_soak_gate["pass_all_ranks"]\n'
    '            ),\n',
    '            "pperr": _mps_soak_error,\n'
    '            "poison_survivors": _mps_soak_poison,\n'
    '            "poison_first_bad_epoch": _mps_soak_poison_epoch,\n'
    '            "correctness": _mps_soak_gate,\n'
    '            "pass": bool(\n'
    '                _mps_soak_completed == _mps_soak_iters\n'
    '                and _mps_soak_error == 0\n'
    '                and _mps_soak_poison == 0\n'
    '                and _mps_soak_gate["pass_all_ranks"]\n'
    '            ),\n',
    "soak-pass",
)

src = edit(
    src,
    '                f"[MPS SOAK] completed={_mps_soak_completed}/{_mps_soak_iters} "\n'
    '                f"pperr={_mps_soak_error} pass={R[\'mps_soak\'][\'pass\']}",\n',
    '                f"[MPS SOAK] completed={_mps_soak_completed}/{_mps_soak_iters} "\n'
    '                f"pperr={_mps_soak_error} poison={_mps_soak_poison} "\n'
    '                f"poison_epoch={_mps_soak_poison_epoch} "\n'
    '                f"pass={R[\'mps_soak\'][\'pass\']}",\n',
    "soak-print",
)

src = edit(
    src,
    '    def _mok_measure_arm(name, body):\n'
    '        if name != "production":\n'
    '            cand_out.zero_()\n',
    '    def _mok_measure_arm(name, body):\n'
    '        if name != "production":\n'
    '            _poison_out()\n',
    "measure-entry",
)

src = edit(
    src,
    '        aligned_us = _mok_rank_max(local_us)\n'
    '        strict_gate = gate(_obuf(name))\n',
    '        aligned_us = _mok_rank_max(local_us)\n'
    '        # exp_32 P3: the timed loop leaves cand_out holding 600 epochs of\n'
    '        # accumulated writes, so a row a LATE epoch failed to write still reads\n'
    "        # an earlier epoch's bit-identical correct answer and the gates below\n"
    '        # cannot see it. Re-poison and run ONE untimed epoch so they see\n'
    '        # single-epoch coverage in the steady state the timed loop just left.\n'
    '        # This is strictly after every HIP event has been consumed and after the\n'
    '        # rank-max all-reduce above, so it cannot touch a measurement. pperr is\n'
    '        # NOT cleared: it is atomicOr-accumulated, so output_gate["pperr"] below\n'
    '        # still reports the OR over warmup + timed + this epoch. Skipped when\n'
    '        # pperr is already set, because M8 suppresses every batch on a live error\n'
    '        # bit and would leave a false poison survivor.\n'
    '        if MOK_POISON_OUT and name != "production" and int(pperr.item()) == 0:\n'
    '            _poison_out()\n'
    '            torch.cuda.synchronize()\n'
    '            hbarrier()\n'
    '            body(sp())\n'
    '            torch.cuda.synchronize()\n'
    '            hbarrier()\n'
    '            R.setdefault("poison", {})[name + "_post_timing"] = _poison_report(\n'
    '                "post_timing", name)\n'
    '        strict_gate = gate(_obuf(name))\n',
    "measure-verify",
)

# ---------------------------------------------------------------- run_campaign.sh
rc = open(RC, encoding="utf-8", newline="").read()
rc = edit(
    rc,
    '      -e "K0_MOK_RELATIVE_TOLERANCE=${K0_MOK_RELATIVE_TOLERANCE:-0.1}" \\\n',
    '      -e "K0_MOK_RELATIVE_TOLERANCE=${K0_MOK_RELATIVE_TOLERANCE:-0.1}" \\\n'
    '      -e "K0_MOK_POISON_OUT=${K0_MOK_POISON_OUT:-1}" \\\n',
    "campaign-forward",
)

mod_ab = os.path.join(TD, "mod", "prefill_opt", "host", "e004pf_k0pf_ab.py")
mod_rc = os.path.join(TD, "mod", "benchmarks", "mok_synthetic_prefill", "run_campaign.sh")
os.makedirs(os.path.dirname(mod_ab), exist_ok=True)
os.makedirs(os.path.dirname(mod_rc), exist_ok=True)
open(mod_ab, "w", encoding="utf-8", newline="").write(src)
open(mod_rc, "w", encoding="utf-8", newline="").write(rc)
print("wrote", mod_ab)
print("wrote", mod_rc)
print("ab lines", src.count("\n"), "rc lines", rc.count("\n"))
