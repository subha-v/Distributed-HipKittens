#!/usr/bin/env python3
"""Static instruction census of the GEMM k-loop, per template instantiation.

Read-only analysis of an ISA (.s) file produced by `hipcc --save-temps`.
Nothing here runs on a GPU; nothing here writes outside exp_27_mainloop/.

Method, stated so the numbers can be labelled honestly in result.md:

  1. The .s is split into per-function regions on `.type <sym>,@function`.
  2. Inside a function, basic blocks are the spans between `.LBB*` labels and
     `; %bb.N` comments. Block order is program order, which is what the
     branch analysis below needs.
  3. A block is a *back-edge target* if some later block branches to it
     (`s_cbranch_*`/`s_branch` naming a label at a lower program-order index).
     The innermost loop is taken to be the smallest such [target, latch] span
     that contains at least one `v_mfma`. exp_03 recorded that the assembler's
     own `in Loop: Header=` comments UNDER-report k-loop membership after the
     guard was turned into a choice-of-latch, so those comments are ignored
     and the branch targets are used instead.
  4. Instructions are classified by opcode prefix into the pipe that executes
     them. Counts are per wave per loop trip.

Everything this prints is COUNTED from the ISA. It contains no cycle model;
cycles are applied in design.md where the assumption behind each rate is
stated separately.
"""
import re
import sys
import json
import collections

MANGLE = re.compile(r'Li(\d+)E')
TAIL = re.compile(r'Lb(\d)E')


def pretty(sym: str) -> str:
    nums = MANGLE.findall(sym)
    tail = TAIL.search(sym)
    if len(nums) >= 3:
        return "<%s,%s,%s,%s>" % (nums[0], nums[1], nums[2],
                                  "true" if tail and tail.group(1) == "1" else "false")
    return sym


def classify(op: str) -> str:
    if op.startswith('v_mfma'):
        return 'mfma'
    if op.startswith('ds_read'):
        return 'ds_read'
    if op.startswith('ds_write'):
        return 'ds_write'
    if op.startswith(('global_load', 'buffer_load', 'flat_load')):
        return 'vmem_load'
    if op.startswith(('global_store', 'buffer_store', 'flat_store')):
        return 'vmem_store'
    if op.startswith(('global_atomic', 'buffer_atomic', 'flat_atomic')):
        return 'vmem_atomic'
    if op.startswith('s_waitcnt'):
        return 'waitcnt'
    if op.startswith('s_barrier'):
        return 'barrier'
    if op.startswith('s_nop'):
        return 'nop'
    if op.startswith('v_accvgpr'):
        return 'accvgpr_move'
    if op.startswith('scratch_'):
        return 'scratch'
    if op.startswith('v_'):
        return 'valu'
    if op.startswith('s_load'):
        return 'smem'
    if op.startswith(('s_branch', 's_cbranch', 's_endpgm', 's_setpc', 's_swappc')):
        return 'branch'
    if op.startswith('s_'):
        return 'salu'
    return 'other'


INSTR = re.compile(r'^\s+([a-z][a-z0-9_]*)\b(.*)$')
LABEL = re.compile(r'^(\.LBB[0-9_]+):')
BBCOM = re.compile(r'^;\s*%bb\.(\d+):')
BRANCH = re.compile(r'^\s+s_(?:cbranch\w*|branch)\s+(\.LBB[0-9_]+)')


class Block:
    def __init__(self, name, idx):
        self.name = name
        self.idx = idx
        self.instrs = []          # (op, operands)
        self.branch_targets = []


def split_functions(lines):
    """-> list of (symbol, [lines])."""
    out, cur, name = [], None, None
    for ln in lines:
        m = re.match(r'^\s*\.type\s+([^,]+),@function', ln)
        if m:
            if cur is not None:
                out.append((name, cur))
            name, cur = m.group(1).strip(), []
            continue
        if cur is not None:
            cur.append(ln)
            if re.match(r'^\s*\.size\s', ln):
                out.append((name, cur))
                cur, name = None, None
    if cur is not None:
        out.append((name, cur))
    return out


def blocks_of(body):
    blks, cur, n = [], Block('entry', 0), 0
    for ln in body:
        m = LABEL.match(ln)
        if m:
            blks.append(cur)
            n += 1
            cur = Block(m.group(1), n)
            continue
        m = BBCOM.match(ln.strip()) or BBCOM.match(ln)
        if m:
            blks.append(cur)
            n += 1
            cur = Block('%%bb.%s' % m.group(1), n)
            continue
        mb = BRANCH.match(ln)
        if mb:
            cur.branch_targets.append(mb.group(1))
        mi = INSTR.match(ln)
        if mi and not ln.strip().startswith('.'):
            cur.instrs.append((mi.group(1), mi.group(2).strip()))
    blks.append(cur)
    return blks


def innermost_mfma_loop(blks):
    """Smallest [header, latch] program-order span closed by a back edge and
    containing a v_mfma. Returns (lo, hi) inclusive block indices or None."""
    byname = {b.name: i for i, b in enumerate(blks)}
    best = None
    for i, b in enumerate(blks):
        for t in b.branch_targets:
            j = byname.get(t)
            if j is None or j > i:
                continue                       # forward edge, not a loop
            span = list(range(j, i + 1))
            if not any(op.startswith('v_mfma')
                       for k in span for op, _ in blks[k].instrs):
                continue
            if best is None or (i - j) < (best[1] - best[0]):
                best = (j, i)
    return best


def census(blks, lo, hi):
    per_block, total = [], collections.Counter()
    widths = collections.Counter()
    waits = collections.Counter()
    for k in range(lo, hi + 1):
        c = collections.Counter()
        for op, ops in blks[k].instrs:
            cls = classify(op)
            c[cls] += 1
            total[cls] += 1
            if cls in ('ds_read', 'ds_write', 'vmem_load', 'vmem_store'):
                widths[op] += 1
            if cls == 'waitcnt':
                waits[ops if ops else '(bare)'] += 1
        per_block.append({'block': blks[k].name,
                          'n': sum(c.values()),
                          'hist': dict(sorted(c.items()))})
    return per_block, total, widths, waits


def main(path, out_json):
    lines = open(path, errors='replace').read().splitlines()
    funcs = split_functions(lines)
    report = {'isa_file': path, 'kernels': []}
    for sym, body in funcs:
        if 'gemm_rs_mi300x_kernel' not in sym:
            continue
        blks = blocks_of(body)
        loop = innermost_mfma_loop(blks)
        entry = {'symbol': sym, 'instantiation': pretty(sym),
                 'n_blocks': len(blks)}
        if loop is None:
            entry['k_loop'] = None
            report['kernels'].append(entry)
            continue
        lo, hi = loop
        per_block, total, widths, waits = census(blks, lo, hi)
        entry['k_loop'] = {
            'header_block': blks[lo].name,
            'latch_block': blks[hi].name,
            'n_blocks': hi - lo + 1,
            'per_block': per_block,
            'total_per_trip': dict(sorted(total.items())),
            'memory_widths': dict(sorted(widths.items())),
            'waitcnt_forms': dict(sorted(waits.items())),
        }
        report['kernels'].append(entry)

    print("=" * 78)
    print("k-loop instruction census, per wave per trip  (COUNTED from ISA)")
    print("=" * 78)
    for e in report['kernels']:
        print("\n### %s   (%s)" % (e['instantiation'], e['symbol'][:60]))
        kl = e['k_loop']
        if kl is None:
            print("   no mfma loop found")
            continue
        print("   loop blocks %s .. %s  (%d blocks)"
              % (kl['header_block'], kl['latch_block'], kl['n_blocks']))
        for b in kl['per_block']:
            print("     %-12s n=%-4d %s" % (b['block'], b['n'], b['hist']))
        print("   TOTAL/trip: %s" % kl['total_per_trip'])
        print("   widths    : %s" % kl['memory_widths'])
        print("   waitcnts  : %s" % kl['waitcnt_forms'])
    with open(out_json, 'w') as f:
        json.dump(report, f, indent=1)
    print("\nwrote %s" % out_json)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
