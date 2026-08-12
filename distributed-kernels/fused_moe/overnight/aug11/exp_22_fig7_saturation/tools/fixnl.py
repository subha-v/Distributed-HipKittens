#!/usr/bin/env python3
"""Normalize CRLF -> LF for every text artifact under ~/e22 (scp from Windows).

Run after every scp; shell scripts with CR die in obscure ways and a CR inside a
`-D` macro or an inline-asm string is worse.
"""
import os
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/e22")
EXTS = (".sh", ".hip", ".cuh", ".cpp", ".py", ".md", ".json", ".csv", ".txt")
n = 0
for base, _dirs, files in os.walk(ROOT):
    if "/DHK" in base or "\\DHK" in base:
        continue
    for f in files:
        if not f.endswith(EXTS):
            continue
        p = os.path.join(base, f)
        b = open(p, "rb").read()
        if b"\r" in b:
            open(p, "wb").write(b.replace(b"\r\n", b"\n").replace(b"\r", b"\n"))
            n += 1
            print("fixed", p)
print("fixnl: normalized", n, "file(s) under", ROOT)
