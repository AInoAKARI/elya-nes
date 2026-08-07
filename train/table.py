#!/usr/bin/env python3
"""Assemble the training table from runs/*.json, including whether each arm's
sparsity actually fits the ROM's 7-bank weight-stream window."""
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "host"))
import ref

CAP = ref.STREAM_BANKS * ref.BANK - ref.STREAM_BANKS * ref.BLOCK

rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1] if len(sys.argv) > 1 else "runs", "*.json"))):
    m = json.load(open(f))
    rows.append(m)

print("| arm | vocab | quant | tau | fit | val | density | nnz | stream banks | fits 7? |")
print("|---|---|---|---|---|---|---|---|---|---|")
for m in sorted(rows, key=lambda r: r["val"]):
    nnz = m["nnz"]
    banks = int(math.ceil((nnz + ref.STREAM_BANKS * ref.BLOCK) / float(ref.BANK)))
    q = {2: "QAT", 1: "float W", 0: "fp32"}[m["quant"]]
    print("| %s | %s | %s | %.2f | %.4f | %.4f | %.4f | %d | %d | %s |"
          % (m["name"], m["vocab"], q, m["tau"], m["fit"], m["val"],
             m["density"], nnz, banks, "yes" if nnz <= CAP else "**NO**"))
print("\nuniform baseline ln(64) = %.4f" % math.log(64))
print("7-bank stream window holds at most %d index bytes -> density %.4f"
      % (CAP, CAP / 102400.0))
