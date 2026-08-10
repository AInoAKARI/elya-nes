#!/usr/bin/env python3
"""What a weight-stream bank switch costs INSIDE the token loop.

The MMC5 datasheet number for `sta $5114` is 6 cycles.  That is the store, not
the switch: the ROM's sentinel path also has to bump the bank counter, rebuild
the OR mask, reset the self-modified chain pointer, step the header pointer and
re-read the header row it was about to read.  The BANKPROF build brackets that
whole body with markers 50 and 51 in situ, so this measures the real thing on
real tokens rather than a hand count.

Each BMARK costs exactly 12 cycles, and the pair's own cost lands inside the
interval, so the reported body cost has one BMARK_COST removed.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nes_bench import run_rom, cycles, CPU_HZ

MARK_COST = 12


def main():
    rom = sys.argv[1] if len(sys.argv) > 1 else "out/nnbank.nes"
    secs = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    r = run_rom(rom, seconds=secs, timeout=7200)
    print("rom:", rom)
    print("status:", r["status"])
    ev = r["events"]

    per_tok = []          # list of [switch_cycles...] per token
    tot = []
    cur, tstart, open_at = None, None, None
    router, router_at = [], []
    for v, t in ev:
        if v == 1:
            cur = []
            per_tok.append(cur)
            tstart = t
        elif v == 2 and cur is not None:
            tot.append(cycles(t - tstart))
            cur = None
        elif v == 50 and cur is not None:
            open_at = t
        elif v == 51 and cur is not None and open_at is not None:
            cur.append(cycles(t - open_at))
            open_at = None
        elif v == 60 and cur is not None:
            router_at.append(t)
        elif v == 61 and cur is not None and router_at:
            router.append(cycles(t - router_at.pop()))

    all_sw = [c for tk in per_tok for c in tk]
    n = len(all_sw)
    print("\ntokens measured           %d" % len(tot))
    print("bank switches per token   %s"
          % " ".join(str(len(tk)) for tk in per_tok[:8]))
    if not n:
        print("NO BANK SWITCHES OBSERVED")
        return 1
    raw = sorted(all_sw)
    body = [c - MARK_COST for c in all_sw]
    print("\nbank-switch body, marker 50 -> 51, %d samples" % n)
    print("  raw interval        min %d  max %d  mean %.2f"
          % (raw[0], raw[-1], sum(raw) / float(n)))
    print("  minus one BMARK     min %d  max %d  mean %.2f cycles"
          % (min(body), max(body), sum(body) / float(n)))
    hist = {}
    for c in body:
        hist[c] = hist.get(c, 0) + 1
    print("  histogram           %s"
          % "  ".join("%d:%d" % (k, hist[k]) for k in sorted(hist)))

    if router:
        rb = [c - MARK_COST for c in router]
        print("\nrouter body, marker 60 -> 61, %d samples" % len(router))
        print("  minus one BMARK     min %d  max %d  mean %.2f cycles"
              % (min(rb), max(rb), sum(rb) / float(len(rb))))

    sw_per_tok = n / float(len(tot))
    mean_tok = sum(tot) / float(len(tot))
    inflated = 2 * MARK_COST * sw_per_tok
    print("\ntoken cost with the markers in   %.0f cycles/token" % mean_tok)
    print("markers add                      %.1f cycles/token "
          "(2 x %d x %.2f switches)" % (inflated, MARK_COST, sw_per_tok))
    print("token cost without them          %.0f cycles/token" % (mean_tok - inflated))
    tot_sw = sum(body) / float(len(tot))
    print("\nbank switching is               %.1f cycles/token = %.4f%% of a token"
          % (tot_sw, 100.0 * tot_sw / (mean_tok - inflated)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
