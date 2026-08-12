#!/usr/bin/env python3
"""exp_32: replace the hand-written diff in patch.md section 6 with the
mechanically generated, dry-run-verified patch."""
import sys

MD = sys.argv[1]
PATCH = sys.argv[2]

md = open(MD, encoding="utf-8", newline="").read()
patch = open(PATCH, encoding="utf-8", newline="").read().replace("\r\n", "\n")

start = md.index("## 6. The patch")
end = md.index("### Two presentation artifacts")

header = """## 6. The patch

Two files, one diff. **Verified**: generated mechanically from the two node files
(scp'd locally, sha256 confirmed against the node), then dry-run applied to a
clean copy with `git apply -p1` -- **applies cleanly**, and the result is
byte-identical to the intended file. The patched `e004pf_k0pf_ab.py` parses
(`ast.parse`) and the patched `run_campaign.sh` passes `bash -n`. Apply from
`~/amd-master/auto-gpu-kernel/k0_fused_moe`:

```bash
git apply -p1 exp_32_poison.patch      # or: patch -p1 < exp_32_poison.patch
```

```diff
%s```

""" % patch

open(MD, "w", encoding="utf-8", newline="").write(md[:start] + header + md[end:])
print("embedded %d patch lines" % patch.count("\n"))
