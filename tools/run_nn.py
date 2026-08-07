#!/usr/bin/env python3
"""Run the transformer ROM, compare its generated tokens against the host
reference token by token, and report exact cycles per token.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nes_bench import run_rom, cycles, CPU_HZ

MARK_COST = 6


def main():
    rom = sys.argv[1] if len(sys.argv) > 1 else "out/nn.nes"
    exp = json.load(open(sys.argv[2] if len(sys.argv) > 2
                        else "out/model/expected.json"))
    secs = int(sys.argv[3]) if len(sys.argv) > 3 else 120

    dumps = [("tokens", 0x0200, 32), ("ntok", 0x0220, 1),
             ("xvec", 0x0600, 64), ("outb", 0x0500, 128),
             ("actb", 0x0400, 128), ("attv", 0x0680, 64)]
    r = run_rom(rom, seconds=secs, dump=dumps, timeout=7200)
    print("status:", r["status"])
    ev = r["events"]
    print("marker events:", len(ev))

    got = list(bytes.fromhex(r["dump.tokens"]))
    want = exp["tokens"][1:]          # tokens[0] is the seed token
    n = len(want)
    print("\nROM vs HOST, token by token")
    print("  %-4s %-8s %-8s %s" % ("pos", "rom", "host", ""))
    bad = 0
    for i in range(n):
        g = got[i] if i < len(got) else None
        w = want[i]
        ok = g == w
        if not ok:
            bad += 1
        print("  %-4d %-8s %-8s %s" % (i, g, w, "ok" if ok else "*** MISMATCH ***"))
    print("\nTOKENS MATCHING: %d/%d  -> %s"
          % (n - bad, n, "EXACT" if bad == 0 else "MISMATCH"))

    # cycles per token from the BEGIN/END pairs
    deltas, start = [], None
    for v, t in ev:
        if v == 1:
            start = t
        elif v == 2 and start is not None:
            deltas.append(t - start)
            start = None
    if deltas:
        print("\nCYCLES PER TOKEN (exact, from the write tap)")
        tot = 0
        for i, d in enumerate(deltas):
            c = cycles(d)
            tot += c
            print("  pos %-3d %10d cycles   %8.4f s @ %.0f Hz"
                  % (i, c, c / CPU_HZ, CPU_HZ))
        print("  %-7s %10d cycles   %8.4f s   (mean %d cycles/token)"
              % ("TOTAL", tot, tot / CPU_HZ, tot // len(deltas)))
    if bad:
        print("\nfirst-mismatch debug dumps:")
        for k in ("xvec", "attv", "outb"):
            if "dump." + k in r:
                print("  %s = %s" % (k, r["dump." + k]))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
