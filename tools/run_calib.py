#!/usr/bin/env python3
"""Run the datasheet calibration ROM and report every payload against its
datasheet cycle count.  Also derives the CPU clock from a known 6-cycle
sequence rather than assuming it, and checks bit-identical reproducibility
(including absolute timestamps) across repeated runs.
"""
import subprocess
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nes_bench import run_rom, pair_windows, derive_clock, cycles, CPU_HZ

MARK_COST = 6  # ldx/ldy #imm (2) + stx/sty abs (4)

# order must match the jsr list in rom/calib.s
TESTS = [
    ("empty",                0, "no payload - proves no constant is smuggled in"),
    ("nop",                  2, ""),
    ("lda #imm",             2, ""),
    ("lda zp",               3, ""),
    ("lda abs",              4, ""),
    ("sta abs",              4, ""),
    ("lda abs,x  no cross",  4, ""),
    ("lda abs,x  CROSS",     5, "indexed LOAD takes +1"),
    ("lda abs,y  no cross",  4, ""),
    ("lda abs,y  CROSS",     5, "indexed LOAD takes +1"),
    ("sta abs,x  no cross",  5, "indexed STORE always 5"),
    ("sta abs,x  CROSS",     5, "indexed STORE takes NO +1"),
    ("sta abs,y  CROSS",     5, "indexed STORE takes NO +1"),
    ("lda (zp),y no cross",  5, ""),
    ("lda (zp),y CROSS",     6, ""),
    ("sta (zp),y CROSS",     6, "always 6"),
    ("lda (zp,x)",           6, ""),
    ("inc abs",              6, "RMW dummy write modelled"),
    ("inc zp",               5, ""),
    ("asl a",                2, ""),
    ("jsr + rts",           12, ""),
    ("pha + pla",            7, ""),
    ("lda zp,x",             4, ""),
    ("inc abs,x",            7, "RMW indexed always 7"),
    ("branch not taken",     2, ""),
    ("branch taken",         3, ""),
    ("branch taken CROSS",   4, ""),
    ("lda #imm + sta abs",   6, "the clock reference"),
]


def main():
    rom = sys.argv[1] if len(sys.argv) > 1 else "out/calib.nes"
    runs = [run_rom(rom, seconds=20) for _ in range(3)]
    for r in runs:
        if r["status"] != "OK":
            raise SystemExit("run status %s" % r["status"])

    ev = [r["events"] for r in runs]
    identical = all(e == ev[0] for e in ev)
    print("REPRODUCIBILITY over 3 runs, including absolute timestamps: %s"
          % ("BIT-IDENTICAL" if identical else "DIFFERS"))
    if not identical:
        for i, e in enumerate(ev):
            print("  run %d first 4: %s" % (i, e[:4]))

    deltas = pair_windows(ev[0])
    if len(deltas) != len(TESTS):
        raise SystemExit("expected %d windows, got %d" % (len(TESTS), len(deltas)))

    # ---- derive the clock from the known 6-cycle lda #imm + sta abs -------
    ref_raw = deltas[TESTS.index(("lda #imm + sta abs", 6, "the clock reference"))]
    print("\nCLOCK DERIVATION")
    print("  reference payload: lda #imm (2) + sta abs (4) = 6 cycles")
    print("  measured window   : %d as (payload + %d-cycle marker macro = 12)"
          % (ref_raw, MARK_COST))
    for name, hz, c in derive_clock(ref_raw, 12.0):
        print("    %-28s %12.4f Hz -> %.12f cycles  err %+.3e"
              % (name, hz, c, c - 12.0))
    print("  chosen: %.0f Hz" % CPU_HZ)

    # ---- resolution: one CPU cycle in attoseconds -------------------------
    print("\nRESOLUTION: 1 CPU cycle = %.0f as = %.4f ns"
          % (1e18 / CPU_HZ, 1e9 / CPU_HZ))

    print("\nDATASHEET CALIBRATION")
    print("  %-24s %8s %8s %s" % ("payload", "measured", "expect", ""))
    bad = 0
    for (name, exp, note), d in zip(TESTS, deltas):
        got = cycles(d) - MARK_COST
        flag = "ok" if got == exp else "*** MISMATCH ***"
        if got != exp:
            bad += 1
        print("  %-24s %8d %8d  %-16s %s" % (name, got, exp, flag, note))
    print("\nCALIBRATION SCORE: %d/%d, %d mismatches"
          % (len(TESTS) - bad, len(TESTS), bad))

    # ---- exactness of the attosecond -> cycle conversion ------------------
    worst = 0.0
    for d in deltas:
        c = d * CPU_HZ / 1e18
        worst = max(worst, abs(c - round(c)))
    print("worst deviation of any window from an integer cycle count: %.3e" % worst)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
