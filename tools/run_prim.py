#!/usr/bin/env python3
"""Run the MMC5 primitive ROM and report every primitive the port's cost
model rests on, against the values measured before the scratchpad wipe.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nes_bench import run_rom, pair_windows, cycles

MARK_COST = 6

# order must match the jsr list in rom/prim.s.  `expect` is the PRIOR-RUN
# value; a mismatch is a signal that something is wrong, not a new result.
PRIMS = [
    ("MMC5 bank switch  lda#+sta $5114",  6,  "MMC5"),
    ("MMC5 bank switch  sta $5114 (A set)", 4, "MMC5, value already in A"),
    ("$6000 PRG-RAM  lda abs",            4,  "= absolute, NO cartridge penalty"),
    ("$6000 PRG-RAM  sta abs",            4,  ""),
    ("$6000 PRG-RAM  lda abs,y aligned",  4,  ""),
    ("$6000 PRG-RAM  lda abs,y CROSS",    5,  "same page-cross hazard as RAM"),
    ("PRG-ROM  lda tbl,y page-aligned",   4,  ""),
    ("PRG-ROM  lda tbl,y CROSS",          5,  ""),
    ("8-bit accumulate, acc in A",        4,  "per element"),
    ("  x16 (linearity)",                64,  "16 x 4"),
    ("8-bit accumulate, spilled to zp",  12,  ""),
    ("16-bit accumulate",                20,  ""),
    ("32-bit accumulate",                36,  ""),
    ("ternary sign-separated gather",     8,  "ldy idx,x + adc act,y"),
    ("  x16 (linearity)",               128,  "16 x 8"),
    ("ternary branchy: skip (trit 0)",    7,  ""),
    ("ternary branchy: add  (trit +1)",  20,  ""),
    ("ternary branchy: sub  (trit -1)",  21,  ""),
    ("zero page  lda zp",                 3,  "for the $6000 comparison"),
]


def main():
    rom = sys.argv[1] if len(sys.argv) > 1 else "out/prim.nes"
    dumps = [("rambank", 0x0200, 8), ("sig", 0x0208, 4)]
    runs = [run_rom(rom, seconds=20, dump=dumps) for _ in range(3)]
    for r in runs:
        if r["status"] != "OK":
            raise SystemExit("run status %s" % r["status"])
    identical = all(r["events"] == runs[0]["events"] for r in runs)
    print("REPRODUCIBILITY over 3 runs (absolute timestamps): %s"
          % ("BIT-IDENTICAL" if identical else "DIFFERS"))

    r = runs[0]
    deltas = pair_windows(r["events"])
    if len(deltas) != len(PRIMS):
        raise SystemExit("expected %d windows, got %d" % (len(PRIMS), len(deltas)))

    print("\nMEASURED PRIMITIVES")
    print("  %-38s %8s %8s" % ("primitive", "measured", "prior"))
    bad = 0
    for (name, exp, note), d in zip(PRIMS, deltas):
        got = cycles(d) - MARK_COST
        ok = got == exp
        bad += 0 if ok else 1
        print("  %-38s %8d %8d  %-16s %s"
              % (name, got, exp, "ok" if ok else "*** MISMATCH ***", note))
    print("\nPRIMITIVES MATCHING PRIOR RUN: %d/%d" % (len(PRIMS) - bad, len(PRIMS)))

    sig = bytes.fromhex(r["dump.sig"])
    print("\nBANK SWITCH ACTUALLY HAPPENED (signature byte read at $8000)")
    print("  before any switch      : $%02X  (expect $B0)" % sig[0])
    print("  after lda#$81/sta $5114: $%02X  (expect $B1)" % sig[2])
    print("  after sta $5114 (A=$82): $%02X  (expect $B2)" % sig[3])
    print("  at end of run          : $%02X  (expect $B2)" % sig[1])
    sig_ok = (sig[0], sig[2], sig[3], sig[1]) == (0xB0, 0xB1, 0xB2, 0xB2)
    print("  -> %s" % ("OK" if sig_ok else "*** BANKS DID NOT SWITCH ***"))

    rb = bytes.fromhex(r["dump.rambank"])
    print("\nPRG-RAM BANKS: header DECLARES 8 x 8 KB = 64 KB.")
    print("  wrote $A0+b to bank b for b=0..7, then read every bank back:")
    print("  " + " ".join("b%d=$%02X" % (i, v) for i, v in enumerate(rb)))
    distinct = len(set(rb))
    print("  distinct values -> %d real 8 KB banks = %d KB of PRG-RAM"
          % (distinct, distinct * 8))
    return 1 if (bad or not sig_ok) else 0


if __name__ == "__main__":
    sys.exit(main())
