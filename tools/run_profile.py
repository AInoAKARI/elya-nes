#!/usr/bin/env python3
"""Per-stage cycle profile of the transformer ROM.

Uses the PROFILE build, whose stage markers are X-preserving and cost exactly
12 cycles each.  That overhead is counted and subtracted so the numbers are
the real cost of the stage, not the cost of measuring it.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nes_bench import run_rom, cycles, CPU_HZ

PMARK_COST = 12


def main():
    rom = sys.argv[1] if len(sys.argv) > 1 else "out/nnprof.nes"
    info = json.load(open("out/model/expected.json"))["info"]
    r = run_rom(rom, seconds=400, timeout=7200)
    if r["status"] != "OK":
        raise SystemExit("status %s" % r["status"])
    ev = r["events"]

    tok = []          # per-token totals
    gath = []         # (start, ) accumulators
    tstart = None
    gstart = None
    astart = None
    gsum = 0
    asum = 0
    gcount = 0
    per_tok_g = []
    per_tok_a = []
    for v, t in ev:
        if v == 1:
            tstart = t
            gsum = asum = 0
            gcount = 0
        elif v == 2:
            tok.append(t - tstart)
            per_tok_g.append((gsum, gcount))
            per_tok_a.append(asum)
        elif v == 20:
            gstart = t
        elif v == 21:
            gsum += t - gstart
            gcount += 1
        elif v == 30:
            astart = t
        elif v == 31:
            asum += t - astart

    n = len(tok)
    nnz = info["nnz"]
    rows = info["rows"]
    print("tokens profiled: %d   rows/token: %d   nnz/token: %d" % (n, rows, nnz))

    # last position = full context, the worst case
    for label, i in (("first token (pos 0)", 0), ("last token (pos %d)" % (n - 1), n - 1)):
        total = cycles(tok[i])
        g = cycles(per_tok_g[i][0])
        gc = per_tok_g[i][1]
        a = cycles(per_tok_a[i])
        # remove the marker overhead that sits inside each measured region
        # (none is inside a gather/attention window; the markers bracket them)
        overhead = (gc * 2 + 2) * PMARK_COST
        other = total - g - a - overhead
        print("\n%s" % label)
        print("  total (profiled)        %10d cycles" % total)
        print("  marker overhead         %10d cycles  (%d markers x %d)"
              % (overhead, gc * 2 + 2, PMARK_COST))
        print("  ternary gather_row      %10d cycles  (%5.1f%%)  over %d rows"
              % (g, 100.0 * g / total, gc))
        print("  attention               %10d cycles  (%5.1f%%)"
              % (a, 100.0 * a / total))
        print("  everything else         %10d cycles  (%5.1f%%)"
              % (other, 100.0 * other / total))
        print("  --> ternary kernel: %.2f cycles per MAC (nnz = %d)"
              % (g / float(nnz), nnz))
        print("  --> ternary kernel: %.2f cycles per WEIGHT (%d weights)"
              % (g / float(info["weights_per_token"]), info["weights_per_token"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
