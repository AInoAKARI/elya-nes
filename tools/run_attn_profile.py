#!/usr/bin/env python3
"""Cycle breakdown INSIDE attention, from the ATTNPROF build.

The PROFILE build brackets attention as a whole (markers 30/31).  This build
adds nested pairs so the ~302 k cycles at full context can be split into the
QK kernel, the per-t score-loop overhead, softmax, the AV kernel and the AV
bookkeeping.

Every AMARK costs exactly 12 cycles.  A nested pair inflates the region that
encloses it, so each enclosing region has 2 x 12 x (nested call count)
subtracted.  Regions with nothing nested inside them need no correction.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nes_bench import run_rom, cycles

MARK_COST = 12

# label -> (begin marker, end marker, [labels nested strictly inside])
PAIRS = [
    ("score loop (all t)", 34, 35, ["QK kernel"]),
    ("QK kernel",          32, 33, []),
    ("softmax",            36, 37, []),
    ("AV section",         40, 41, ["AV kernel"]),
    ("AV kernel",          38, 39, []),
]


def collect(ev):
    """-> list per token of {label: [raw_cycles, calls]}, and token totals."""
    begins = {b: lab for lab, b, e, _ in PAIRS}
    ends = {e: lab for lab, b, e, _ in PAIRS}
    per_tok = []
    totals = []
    open_at = {}
    tstart = None
    cur = None
    for v, t in ev:
        if v == 1:
            cur = {lab: [0.0, 0] for lab, _, _, _ in PAIRS}
            per_tok.append(cur)
            tstart = t
        elif v == 2 and cur is not None:
            totals.append(t - tstart)
        elif cur is not None and v in begins:
            open_at[begins[v]] = t
        elif cur is not None and v in ends:
            lab = ends[v]
            if lab in open_at:
                cur[lab][0] += t - open_at[lab]
                cur[lab][1] += 1
                del open_at[lab]
    return per_tok, totals


def report(rom, label_only=None):
    r = run_rom(rom, seconds=400, timeout=7200)
    if r["status"] != "OK":
        raise SystemExit("status %s" % r["status"])
    per_tok, totals = collect(r["events"])
    n = len(totals)
    out = []
    for label, i in (("first token (pos 0)", 0),
                     ("last token (pos %d)" % (n - 1), n - 1)):
        cur = per_tok[i]
        out.append("")
        out.append("%s   token total %d cycles" % (label, cycles(totals[i])))
        corrected = {}
        for lab, b, e, inner in PAIRS:
            raw, cnt = cur[lab]
            c = cycles(raw)
            for il in inner:
                c -= 2 * MARK_COST * cur[il][1]
            corrected[lab] = (c, cnt)
        for lab, b, e, inner in PAIRS:
            c, cnt = corrected[lab]
            per = (float(c) / cnt) if cnt else 0.0
            out.append("  %-20s %9d cycles  %5d calls  %9.1f cyc/call"
                       % (lab, c, cnt, per))
        sl = corrected["score loop (all t)"][0]
        qk = corrected["QK kernel"][0]
        av = corrected["AV section"][0]
        avk = corrected["AV kernel"][0]
        sm = corrected["softmax"][0]
        out.append("  %-20s %9d cycles" % ("  score-loop other", sl - qk))
        out.append("  %-20s %9d cycles" % ("  AV other", av - avk))
        out.append("  %-20s %9d cycles" % ("attention accounted", sl + sm + av))
        # QK is called once per (layer, head, t) and does d_head MACs;
        # AV is called once per (layer, head, d) and does curpos+1 MACs.
        nm = corrected["QK kernel"][1] * 32
        if nm:
            out.append("  QK: %.2f cycles/MAC over %d MACs" % (qk / float(nm), nm))
        nm2 = corrected["AV kernel"][1] * (i + 1)
        if nm2:
            out.append("  AV: %.2f cycles/MAC over %d MACs" % (avk / float(nm2), nm2))
    return "\n".join(out)


if __name__ == "__main__":
    print(report(sys.argv[1] if len(sys.argv) > 1 else "out/nnattn.nes"))
