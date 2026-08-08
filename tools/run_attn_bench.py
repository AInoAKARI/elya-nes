#!/usr/bin/env python3
"""Isolated slope/intercept for the attention kernels (ATTNBENCH build).

Each measured window is BENCH_REP calls of the kernel at a fixed t-count,
plus BENCH_REP copies of a constant driver.  The driver cancels out of the
SLOPE, so the per-MAC cost falls out of a straight-line fit and the intercept
is whatever the call frame and the carry folds cost.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nes_bench import run_rom, cycles

REP = 64
DRIVER = 20          # lda/sta/sta/ldy/clc/dec/bne around each call


def main():
    rom = sys.argv[1] if len(sys.argv) > 1 else "out/nnabench.nes"
    r = run_rom(rom, seconds=30, timeout=1200)
    if r["status"] != "OK":
        raise SystemExit("status %s" % r["status"])
    per = []
    start = None
    for v, t in r["events"]:
        if v == 1:
            start = t
        elif v == 2 and start is not None:
            per.append(cycles(t - start) / float(REP))
            start = None
    qk = per.pop() if len(per) > 40 else None
    ptr = per[20:40] if len(per) >= 40 else None
    per = per[:20]
    print("AV kernel, isolated   (call frame + %d-cycle driver included)" % DRIVER)
    print("  %-4s %10s %10s %10s" % ("MACs", "cyc/call", "kernel", "cyc/MAC"))
    prev = None
    for i, c in enumerate(per):
        n = i + 1
        k = c - DRIVER
        d = "" if prev is None else "   d=%+.2f" % (c - prev)
        print("  %-4d %10.2f %10.2f %10.3f%s" % (n, c, k, k / n, d))
        prev = c
    if len(per) > 1:
        n0, n1 = 1, len(per)
        slope = (per[-1] - per[0]) / float(n1 - n0)
        icept = per[0] - slope * n0
        print("\n  least-squares over the whole range:")
        xs = list(range(1, len(per) + 1))
        mx = sum(xs) / float(len(xs))
        my = sum(per) / float(len(per))
        num = sum((x - mx) * (y - my) for x, y in zip(xs, per))
        den = sum((x - mx) ** 2 for x in xs)
        s2 = num / den
        print("    endpoint slope   %.4f cycles/MAC   intercept %.2f" % (slope, icept))
        print("    fitted   slope   %.4f cycles/MAC   intercept %.2f"
              % (s2, my - s2 * mx - DRIVER))
    if ptr:
        print("\nNO-self-modifying-code alternative: `adc (mulp),y` "
              "with the multiply row in a zero page pointer")
        print("  %-4s %10s %10s %10s" % ("MACs", "cyc/call", "kernel", "d"))
        prev = None
        for i, c in enumerate(ptr):
            d = "" if prev is None else "%+.2f" % (c - prev)
            print("  %-4d %10.2f %10.2f %10s" % (i + 1, c, c - DRIVER, d))
            prev = c
        sl = (ptr[-1] - ptr[0]) / float(len(ptr) - 1)
        print("  endpoint slope %.4f cycles/MAC "
              "(self-modified form: see above)" % sl)
    if qk is not None:
        k = qk - 10          # ldy/dec/bne driver
        print("\nQK kernel, isolated (fixed 32 MACs)")
        print("  %10.2f cyc/call incl driver   %10.2f kernel   %8.3f cyc/MAC"
              % (qk, k, k / 32.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
