#!/usr/bin/env python3
"""Host reference for the NES ternary transformer.

This file is the SPECIFICATION.  Everything is exact integer arithmetic with
no floating point anywhere in the forward pass, so the 6502 implementation can
be compared to it bit for bit rather than approximately.

It also emits every binary the ROM consumes: the sign-separated weight stream,
the row header table, the embedding / positional tables and the lookup tables.
"""
import json
import os
import struct
import sys

# ---------------------------------------------------------------------------
# model shape (see DESIGN.md for how these were chosen from the RAM budget)
# ---------------------------------------------------------------------------
V  = 64          # vocab
D  = 64          # d_model
L  = 3           # layers
H  = 2           # heads
DH = 32          # d_head  (H * DH == D)
F  = 128         # d_ff
T  = int(os.environ.get("NES_T", "20"))   # context positions

# T is the ONLY shape knob that varies.  It is bounded by the KV cache, which
# is L * T * 2 * D bytes of PRG-RAM at one byte per 4-bit activation:
#   T = 20 ->  7,680 B, 23.4% of the 32,768 B window (one 8 KB bank)
#   T = 85 -> 32,640 B, 99.6% of it - the largest context this cartridge holds
# rom/nn.s takes the same value through -DNCTX and asserts the cache fits.
assert 1 <= T <= 85, "T must be 1..85; L*T*2*D must fit 32 KB of PRG-RAM"

BLOCK    = 16    # gather block size; 16*14 = 224 < 256 keeps the carry clear
BIAS     = 7     # activations stored as value+7 in 0..14

K_SHIFT  = int(os.environ.get("NES_K_SHIFT", "2"))   # Wq/Wk/Wv/Wo/W1
W2_SHIFT = int(os.environ.get("NES_W2_SHIFT", "3"))  # W2 (128 inputs)
AV_SHIFT = int(os.environ.get("NES_AV_SHIFT", "2"))  # attention value sum
SM_SHIFT = int(os.environ.get("NES_SM_SHIFT", "3"))  # score difference -> exp

# AV_SHIFT was 4 in the first cut of this port.  It is now 2, and the reason
# it is 2 rather than the 1 the range argument demands is a measured negative -
# see FINDINGS, "Fixing the attention shift made it WORSE".
# The attention accumulator is provably bounded: the quantised softmax
# normalises so that sum(p_t) <= 8 and every value nibble is in -7..7, so
# sum_t floor(p_t*v_t/4) cannot exceed 7*8/4 = 14.  Shifting that by 4 leaves
# floor(a/16) in {-1, 0} - the entire attention path of the model was carrying
# ONE BIT per element.  Measured over a real 19-token trajectory the raw
# accumulator ran -13..12 with std 7.41 and AV_SHIFT=4 used exactly 2 of the
# 15 available levels.  At AV_SHIFT=1 it uses 14 of 15 and saturates 0.00% of
# the time.  See FINDINGS.

BANK = 8192      # PRG bank size
STREAM_BANKS = int(os.environ.get("NES_STREAM_BANKS", "7"))
                 # rom/nn.cfg WS0..WS6; the stream image is always this long
SENTINEL = 0xFF  # header n_pos value meaning "advance to the next bank"

# ---------------------------------------------------------------------------
# deterministic RNG (a plain LCG so the stream is reproducible anywhere)
# ---------------------------------------------------------------------------
class LCG:
    """xorshift32.

    This started out as a textbook LCG and that was a genuine bug: with a
    power-of-two modulus the low bits of an LCG are periodic, so `s % 4` just
    cycled 0,1,2,3 and every 64-input row came out with EXACTLY 32 nonzeros.
    The tell was that the packer reported zero bank crossings - 51,200 stream
    bytes made of rows that were all exactly 32 or 64 long tile 8,192 with no
    remainder. Structured weights would also have made the exactness test far
    weaker than it looks. xorshift32 has usable low bits.
    """

    def __init__(self, seed):
        self.s = (seed & 0xFFFFFFFF) or 1

    def next(self):
        s = self.s
        s ^= (s << 13) & 0xFFFFFFFF
        s ^= s >> 17
        s ^= (s << 5) & 0xFFFFFFFF
        self.s = s
        return s

    def below(self, n):
        return self.next() % n


# ---------------------------------------------------------------------------
# exact integer primitives - each one has a 1:1 6502 counterpart
# ---------------------------------------------------------------------------
def quant(acc, k):
    """16-bit accumulator -> signed 4-bit activation.

    q = acc >> k  (arithmetic, floor) whenever that lands inside -7..7,
    saturating otherwise.  The 6502 does the same thing with a 16-bit range
    test followed by a 256-byte table lookup on the low byte.
    """
    hi = (8 << k) - 1
    lo = -(7 << k)
    if acc > hi:
        return 7
    if acc < lo:
        return -7
    return acc >> k


def clamp7(x):
    return 7 if x > 7 else (-7 if x < -7 else x)


def nib(v):
    """signed -7..7 -> 4-bit two's complement nibble 0..15"""
    return v & 0x0F


def build_mul_table():
    """mul[(nq<<4)|nk] = ((q*k) >> 2) + 13, biased into 0..25 so that eight of
    them can be summed in one 8-bit register without ever setting carry."""
    t = [0] * 256
    for a in range(16):
        av = a - 16 if a >= 8 else a
        for b in range(16):
            bv = b - 16 if b >= 8 else b
            # nibble 8 (= -8) is unreachable for real activations, which are
            # clamped to -7..7, so clamping it here cannot affect any result.
            t[(a << 4) | b] = max(0, min(255, ((av * bv) >> 2) + 13))
    return t


MUL = build_mul_table()
MUL_BIAS = 13


def build_qtbl(k):
    """qtbl_k[b] = clamp(signed(b) >> k, -7, 7), stored as a byte."""
    t = []
    for b in range(256):
        s = b - 256 if b >= 128 else b
        t.append(clamp7(s >> k) & 0xFF)
    return t


def build_exptab():
    """15 entries for score-difference bucket -14..0, ~64*exp(d/2)."""
    import math
    return [max(0, min(64, int(round(64.0 * math.exp((i - 14) / 2.0)))))
            for i in range(15)]


EXPTAB = build_exptab()

# tbl_p geometry: one row per shift count, SM_PROW entries covering every
# value the exp table can produce.
SM_PROW = max(EXPTAB) + 1
SM_KROWS = 9            # sum(e) <= T*max(EXPTAB), so kk can never exceed 8


def softmax_q(scores):
    """Quantised softmax: max-shift, 15-entry exp table, power-of-two
    normalisation, clamp to 0..7.  Exactly what the ROM does."""
    m = max(scores)
    e = []
    for s in scores:
        d = (s - m) >> SM_SHIFT          # <= 0, arithmetic shift
        if d < -14:
            d = -14
        e.append(EXPTAB[d + 14])
    S = sum(e)
    kk = 0
    while (S >> kk) > 8:
        kk += 1
    return [min(x >> kk, 7) for x in e]


# ---------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------
def gen_ternary(rng, rows, cols):
    """P(0)=0.5, P(+1)=P(-1)=0.25 - a realistic ternary-quantised density."""
    m = []
    for _ in range(rows):
        r = []
        for _ in range(cols):
            u = rng.below(4)
            r.append(0 if u < 2 else (1 if u == 2 else -1))
        m.append(r)
    return m


class Model:
    # Mixture of experts.  nexp == 1 is the dense model and every code path
    # below then behaves exactly as it did before MoE existed.
    nexp = 1
    nexp_head = 1
    route = [0] * V

    @property
    def moe(self):
        return self.nexp > 1 or self.nexp_head > 1

    def __init__(self, seed=20260807):
        rng = LCG(seed)
        self.emb = [[rng.below(15) - 7 for _ in range(D)] for _ in range(V)]
        self.pos = [[rng.below(15) - 7 for _ in range(D)] for _ in range(T)]
        self.layers = []
        for _ in range(L):
            self.layers.append({
                "Wq": gen_ternary(rng, D, D),
                "Wk": gen_ternary(rng, D, D),
                "Wv": gen_ternary(rng, D, D),
                "Wo": gen_ternary(rng, D, D),
                # one-element lists: the dense model is the E = 1 case of the
                # mixture, not a separate shape
                "W1": [gen_ternary(rng, F, D)],
                "W2": [gen_ternary(rng, D, F)],
            })
        self.head = [gen_ternary(rng, V, D)]

    @classmethod
    def from_npz(cls, path):
        """Load trained integer weights.  The npz is the SAME array the trainer
        exported and the same one the max|dW| check reads, so the stream the
        packer writes and the numbers the reference runs are provably one
        quantisation, not two that happen to agree."""
        import numpy as np
        z = np.load(path)
        if "_ctx" in z:
            got = int(z["_ctx"][0])
            if got != T:
                raise SystemExit(
                    "%s was trained at T = %d but this reference is configured "
                    "for T = %d.  Set NES_T=%d, or retrain.  (The positional "
                    "table is the only array whose shape shows it, and a "
                    "silent truncation there is exactly the class of bug this "
                    "repo keeps getting caught by.)" % (path, got, T, got))
        if "_shifts" in z:
            want = [K_SHIFT, W2_SHIFT, AV_SHIFT, SM_SHIFT]
            got = [int(v) for v in z["_shifts"]]
            if got != want:
                raise SystemExit(
                    "%s was trained at shifts K/W2/AV/SM = %s but this "
                    "reference is configured for %s.  Set NES_K_SHIFT / "
                    "NES_W2_SHIFT / NES_AV_SHIFT / NES_SM_SHIFT to match, or "
                    "retrain.  (Loading it anyway gives a plausible-looking "
                    "model that generates rubbish - it happened.)"
                    % (path, got, want))
        m = cls.__new__(cls)
        m.emb = [[int(v) for v in row] for row in z["emb"]]
        m.pos = [[int(v) for v in row] for row in z["pos"]]
        assert len(m.emb) == V and len(m.emb[0]) == D, (len(m.emb), len(m.emb[0]))
        assert len(m.pos) == T and len(m.pos[0]) == D
        # Mixture of experts: the trainer stamps the expert count and the
        # routing table into the npz for the same reason it stamps the shifts
        # and the context length - a model packed against the wrong routing
        # table runs perfectly and says the wrong thing.
        if "_moe" in z:
            m.nexp, m.nexp_head = (int(v) for v in z["_moe"])
            m.route = [int(v) for v in z["_route"]]
            assert len(m.route) == V
            assert max(m.route) < max(m.nexp, m.nexp_head)
        else:
            m.nexp, m.nexp_head, m.route = 1, 1, [0] * V

        def tern(key, rows, cols):
            a = z[key]
            g = [[int(v) for v in row] for row in a]
            assert len(g) == rows and len(g[0]) == cols, (key, len(g), len(g[0]))
            assert set(v for row in g for v in row) <= {-1, 0, 1}
            return g

        m.layers = []
        for l in range(L):
            d = {}
            for nm in ("Wq", "Wk", "Wv", "Wo"):
                d[nm] = tern("L%d_%s" % (l, nm), D, D)
            for nm, rows, cols in (("W1", F, D), ("W2", D, F)):
                if m.nexp == 1:
                    d[nm] = [tern("L%d_%s" % (l, nm), rows, cols)]
                else:
                    d[nm] = [tern("L%d_%s_e%d" % (l, nm, e), rows, cols)
                             for e in range(m.nexp)]
            m.layers.append(d)
        if m.nexp_head == 1:
            m.head = [tern("head", V, D)]
        else:
            m.head = [tern("head_e%d" % e, V, D) for e in range(m.nexp_head)]
        return m

    # -- the ordered list of matrices exactly as the stream stores them -----
    # SHARED chunks and EXPERT chunks alternate, because that is the order
    # rom/nn.s consumes them: attention, then feed-forward, per layer.  A chunk
    # is the unit the packer bank-aligns.
    def chunks(self, e=0):
        out = []
        for l in range(L):
            out.append(("S", [("L%d.%s" % (l, n), self.layers[l][n])
                              for n in ("Wq", "Wk", "Wv", "Wo")]))
            out.append(("E", [("L%d.W1" % l, self.layers[l]["W1"][e]),
                              ("L%d.W2" % l, self.layers[l]["W2"][e])]))
        out.append(("E" if self.nexp_head > 1 else "S",
                    [("head", self.head[e if self.nexp_head > 1 else 0])]))
        return out

    def matrices(self, e=0):
        return [nm for kind, ms in self.chunks(e) for nm in ms]


# ---------------------------------------------------------------------------
# the ternary matmul, written the way the 6502 does it
# ---------------------------------------------------------------------------
def ternary_row(pos_idx, neg_idx, actb):
    """actb is the BIASED activation page (value+7).  Sum the biased values
    for each list, then remove the bias once per list.  This is exactly the
    ROM's arithmetic including the fact that it never subtracts per element."""
    sp = 0
    for i in pos_idx:
        sp += actb[i]
    sn = 0
    for i in neg_idx:
        sn += actb[i]
    return (sp - BIAS * len(pos_idx)) - (sn - BIAS * len(neg_idx))


def split_row(row):
    p = [i for i, w in enumerate(row) if w == 1]
    n = [i for i, w in enumerate(row) if w == -1]
    return p, n


def matmul(mat_split, x, k, relu=False, raw=False):
    """mat_split is [(pos_idx, neg_idx), ...] per output row."""
    actb = [v + BIAS for v in x]
    out = []
    for p, n in mat_split:
        acc = ternary_row(p, n, actb)
        if raw:
            out.append(acc)
            continue
        q = quant(acc, k)
        if relu and q < 0:
            q = 0
        out.append(q)
    return out


# ---------------------------------------------------------------------------
# forward pass
# ---------------------------------------------------------------------------
class Runner:
    def __init__(self, model):
        self.m = model
        # split[e] holds every matrix expert e uses.  The shared ones are the
        # SAME list object in every entry, so this costs nothing and lets
        # step() do one dictionary lookup instead of branching on whether the
        # matrix it wants is routed.
        ne = max(model.nexp, model.nexp_head)
        shared = {}
        self.split = []
        for e in range(ne):
            d = {}
            for kind, ms in model.chunks(e):
                for name, mat in ms:
                    if kind == "S":
                        if name not in shared:
                            shared[name] = [split_row(r) for r in mat]
                        d[name] = shared[name]
                    else:
                        d[name] = [split_row(r) for r in mat]
            self.split.append(d)
        # KV cache holds NIBBLES, exactly as the $6000 window does
        self.K = [[[0] * D for _ in range(T)] for _ in range(L)]
        self.Vc = [[[0] * D for _ in range(T)] for _ in range(L)]
        self.trace = []
        # Opt-in observation only.  Setting this appends (layer, head, pos,
        # probability vector) for every attention head evaluated; it changes no
        # arithmetic and defaults to off so the specification path is unaltered.
        self.record_attn = False
        self.attn_log = []

    def step(self, tok, p):
        m = self.m
        # The router: one table lookup on the CURRENT token, which is all the
        # 6502 can afford and - see FINDINGS - all any pre-layer router could
        # see anyway.
        sp = self.split[m.route[tok]]
        x = [clamp7(m.emb[tok][j] + m.pos[p][j]) for j in range(D)]
        stage = {"tok": tok, "pos": p, "x0": list(x), "exp": m.route[tok]}

        for l in range(L):
            q = matmul(sp["L%d.Wq" % l], x, K_SHIFT)
            kv = matmul(sp["L%d.Wk" % l], x, K_SHIFT)
            vv = matmul(sp["L%d.Wv" % l], x, K_SHIFT)
            for j in range(D):
                self.K[l][p][j] = nib(kv[j])
                self.Vc[l][p][j] = nib(vv[j])

            att = [0] * D
            dbg = {}
            for h in range(H):
                base = h * DH
                scores = []
                for t in range(p + 1):
                    s = 0
                    for j in range(DH):
                        s += MUL[(nib(q[base + j]) << 4) | self.K[l][t][base + j]]
                    scores.append(s - MUL_BIAS * DH)
                pr = softmax_q(scores)
                if self.record_attn:
                    self.attn_log.append((l, h, p, list(pr)))
                dbg["scores"] = list(scores)
                dbg["p"] = list(pr)
                for j in range(DH):
                    s = 0
                    for t in range(p + 1):
                        s += MUL[(nib(pr[t]) << 4) | self.Vc[l][t][base + j]]
                    s -= MUL_BIAS * (p + 1)
                    att[base + j] = quant(s, AV_SHIFT)

            if l == 0:
                stage["att"] = list(att)
                stage["q"] = list(q)
                stage["scores"] = dbg["scores"]
                stage["p"] = dbg["p"]
            o = matmul(sp["L%d.Wo" % l], att, K_SHIFT)
            x = [clamp7(x[j] + o[j]) for j in range(D)]
            hdn = matmul(sp["L%d.W1" % l], x, K_SHIFT, relu=True)
            f = matmul(sp["L%d.W2" % l], hdn, W2_SHIFT)
            x = [clamp7(x[j] + f[j]) for j in range(D)]
            stage["L%d.x" % l] = list(x)

        logits = matmul(sp["head"], x, 0, raw=True)
        best, bi = logits[0], 0
        for i in range(1, V):
            if logits[i] > best:
                best, bi = logits[i], i
        stage["logits0_8"] = logits[:8]
        stage["next"] = bi
        self.trace.append(stage)
        return bi


def generate(model, start_tok, n):
    r = Runner(model)
    toks = [start_tok]
    cur = start_tok
    for p in range(n):
        cur = r.step(cur, p)
        toks.append(cur)
    return toks, r


# ---------------------------------------------------------------------------
# packer: stream + headers, bank aware
# ---------------------------------------------------------------------------
ROWS_PER_TOKEN = L * (4 * D + F + D) + V     # 1,408: every row the ROM walks


def nnz_of(rows):
    return sum(len(p) + len(n) for p, n in rows)


def pack_region(chunks, first_bank):
    """Lay a list of row-chunks into 8 KB banks, one chunk per fresh bank.

    Every chunk starts at a bank boundary because the ROM's stream offset is
    IMPLICIT: `chain_reset` puts the gather index back to 0 on every bank
    switch, so the only place a walk can resume after leaving a region and
    coming back is the start of a bank.  A chunk is therefore the unit that
    can be interleaved with another region, and the shared attention rows and
    an expert's feed-forward rows are exactly such chunks.

    Each bank opens with BLOCK bytes of padding.  The ROM's gather chains
    address the stream as (offset - BLOCK) so that the chain's base address is
    page-aligned; without the pad that quantity goes negative at the start of
    a bank and the chain pointer runs off the front of the chain table.  With
    the base page-aligned the indexed load is 4 cycles instead of 5 - a
    straight 1 cycle per MAC.

    Returns (banks, events_per_chunk).  An event is ('bank', absolute_bank) or
    ('row', pos_idx, neg_idx), in the order the ROM consumes them.
    """
    banks = []
    per_chunk = []
    for rows in chunks:
        banks.append(bytearray(b"\x00" * BLOCK))     # fresh bank for the chunk
        ev = [("bank", first_bank + len(banks) - 1)]
        for p, n in rows:
            need = len(p) + len(n)
            # tracked EXPLICITLY rather than as len(stream) % BANK: a row that
            # ends exactly on a bank boundary makes the modulo read 0, the pad
            # is then never emitted, and the ROM's pointer underflows into
            # chain -1 and executes garbage.  That is what happened the first
            # time.
            if len(banks[-1]) + need > BANK:
                banks.append(bytearray(b"\x00" * BLOCK))
                ev.append(("bank", first_bank + len(banks) - 1))
            ev.append(("row", p, n))
            banks[-1] += bytes(p) + bytes(n)
        per_chunk.append(ev)
    for b in banks:
        b += b"\x00" * (BANK - len(b))
    return banks, per_chunk


def encode_headers(events):
    """Events -> the ROM's 4-byte header rows.

    A bank event is `$FF, bank, 0, 0`: the sentinel now carries the ABSOLUTE
    bank number rather than meaning "increment".  That is what lets one header
    table describe a walk that leaves the shared region for an expert's banks
    and comes back, and it costs two cycles a switch over the increment form.
    """
    out = bytearray()
    for ev in events:
        if ev[0] == "bank":
            assert 0 <= ev[1] < 128, ev
            out += bytes([SENTINEL, ev[1], 0, 0])
        else:
            _, p, n = ev
            d7 = -BIAS * (len(p) - len(n))
            out += bytes([len(p), len(n), d7 & 0xFF, (d7 >> 8) & 0xFF])
    return out


def rows_of(mats):
    return [split_row(r) for _, mat in mats for r in mat]


def pack_streams(model, outdir):
    """Emit stream.bin and the header table(s).  Returns a layout dict."""
    ch = model.chunks(0)
    if not model.moe:
        rows = rows_of([nm for _, ms in ch for nm in ms])
        banks, per_chunk = pack_region([rows], 0)
        nsh = len(banks)
        want = max(STREAM_BANKS, nsh)
        if nsh > STREAM_BANKS:
            raise SystemExit(
                "stream needs %d banks but rom/nn.cfg provides %d.  nnz=%d"
                % (nsh, STREAM_BANKS, nnz_of(rows)))
        stream = bytearray()
        for b in banks:
            stream += b
        stream += b"\x00" * ((want - nsh) * BANK)
        hdr = encode_headers(per_chunk[0])
        with open(os.path.join(outdir, "stream.bin"), "wb") as f:
            f.write(bytes(stream))
        with open(os.path.join(outdir, "headers.bin"), "wb") as f:
            f.write(bytes(hdr))
        return {"moe": 0, "banks": want, "header_bytes": len(hdr),
                "nnz": nnz_of(rows),
                "crossings": sum(1 for e in per_chunk[0] if e[0] == "bank") - 1}

    # ---- mixture of experts ------------------------------------------------
    ne = max(model.nexp, model.nexp_head)
    shared_chunks = [rows_of(ms) for kind, ms in ch if kind == "S"]
    sh_banks, sh_ev = pack_region(shared_chunks, 0)
    nsh = len(sh_banks)

    ex_banks, ex_ev = [], []
    for e in range(ne):
        chunks = [rows_of(ms) for kind, ms in model.chunks(e) if kind == "E"]
        b, v = pack_region(chunks, 0)
        ex_banks.append(b)
        ex_ev.append(v)
    per_exp = max(len(b) for b in ex_banks)

    # re-base each expert's bank numbers now that the stride is known
    for e in range(ne):
        base = nsh + e * per_exp
        ex_ev[e] = [[("bank", ev[1] + base) if ev[0] == "bank" else ev
                     for ev in chunk] for chunk in ex_ev[e]]

    stream = bytearray()
    for b in sh_banks:
        stream += b
    for e in range(ne):
        for b in ex_banks[e]:
            stream += b
        stream += b"\x00" * ((per_exp - len(ex_banks[e])) * BANK)

    hdr_bytes = []
    for e in range(ne):
        ev, si, ei = [], 0, 0
        for kind, _ in model.chunks(e):
            if kind == "S":
                ev += sh_ev[si]; si += 1
            else:
                ev += ex_ev[e][ei]; ei += 1
        hdr_bytes.append(encode_headers(ev))

    with open(os.path.join(outdir, "stream.bin"), "wb") as f:
        f.write(bytes(stream))
    for e in range(ne):
        with open(os.path.join(outdir, "headers_e%d.bin" % e), "wb") as f:
            f.write(bytes(hdr_bytes[e]))
    nnz = sum(nnz_of(c) for c in shared_chunks) + sum(
        sum(nnz_of(rows_of(ms)) for kind, ms in model.chunks(e) if kind == "E")
        for e in range(ne))
    nnz_sh = sum(nnz_of(c) for c in shared_chunks)
    nnz_ex = [sum(nnz_of(rows_of(ms)) for kind, ms in model.chunks(e)
                  if kind == "E") for e in range(ne)]
    return {"moe": 1, "nexp": ne, "shared_banks": nsh, "expert_banks": per_exp,
            "banks": len(stream) // BANK, "nnz": nnz,
            "nnz_shared": nnz_sh, "nnz_expert": nnz_ex,
            # index bytes actually STREAMED for one token: the shared rows plus
            # the one expert the router picked.  The cartridge holds far more.
            "nnz_per_token": [nnz_sh + x for x in nnz_ex],
            "header_bytes": max(len(h) for h in hdr_bytes),
            "crossings": max(len(h) for h in hdr_bytes) // 4 - ROWS_PER_TOKEN}


def pack(model, outdir):
    lay = pack_streams(model, outdir)

    def w(name, data):
        with open(os.path.join(outdir, name), "wb") as f:
            f.write(bytes(data))

    emb = bytearray()
    for t in range(V):
        emb += bytes((v & 0xFF) for v in model.emb[t])
    w("embed.bin", emb)
    pos = bytearray()
    for t in range(T):
        pos += bytes((v & 0xFF) for v in model.pos[t])
    w("pos.bin", pos)
    w("tbl_mul.bin", MUL)
    w("tbl_q2.bin", build_qtbl(K_SHIFT))
    w("tbl_q3.bin", build_qtbl(W2_SHIFT))
    w("tbl_q4.bin", build_qtbl(AV_SHIFT))
    w("tbl_exp.bin", EXPTAB)
    # softmax score difference -> exp bucket, in one lookup instead of a
    # SM_SHIFT-long arithmetic shift loop.  Indexed by the LOW byte of the
    # 16-bit difference; only valid when the high byte is $FF, i.e. the
    # difference is in [-256,-1].  The ROM handles diff == 0 and
    # diff <= -257 as constants, which is why they are emitted below.
    sm = []
    for b in range(256):
        d = (b - 256) >> SM_SHIFT
        sm.append(EXPTAB[max(d, -14) + 14])
    w("tbl_sm.bin", sm)
    # min(e >> kk, 7) << 4, one row per kk.  kk is fixed for a whole softmax,
    # so the ROM picks the row once and the per-position shift loop goes away.
    # e never exceeds max(EXPTAB), so SM_PROW entries per row is enough, and
    # every row from kk = 7 up is all zero - which is why clamping the row
    # index at SM_KROWS-1 is exact rather than approximate.
    pt = []
    for kk in range(SM_KROWS):
        for e in range(SM_PROW):
            pt.append(min(e >> kk, 7) << 4)
    w("tbl_p.bin", pt)
    # entry offset into a 16-entry chain for a list of length n, and the
    # number of blocks that list needs
    w("tbl_entoff.bin", [((BLOCK - (n % BLOCK)) % BLOCK) * 6 for n in range(256)])
    w("tbl_blkcnt.bin", [min(255, (n + BLOCK - 1) // BLOCK) for n in range(256)])
    # how far X advances after the FIRST (possibly partial) block
    w("tbl_step.bin", [(n % BLOCK) or BLOCK for n in range(256)])
    # clamp a signed byte to -7..7 (used for the residual adds)
    w("tbl_clamp.bin", [clamp7(b - 256 if b >= 128 else b) & 0xFF
                        for b in range(256)])
    # emit the shifts as an assembler include so rom/nn.s and this file cannot
    # drift apart.  A silent disagreement here is invisible in the ROM until
    # the token ids differ, which is far too late.
    with open(os.path.join(outdir, "shifts.inc"), "w") as f:
        f.write("; generated by host/ref.py - do not edit\n")
        f.write("KSHIFT   = %d\n" % K_SHIFT)
        f.write("W2SHIFT  = %d\n" % W2_SHIFT)
        f.write("AVSHIFT  = %d\n" % AV_SHIFT)
        f.write("SMSHIFT  = %d\n" % SM_SHIFT)
        f.write("EXP_TOP  = %d\n" % EXPTAB[14])
        f.write("EXP_BOT  = %d\n" % EXPTAB[0])
        f.write("SM_PROW  = %d\n" % SM_PROW)
        f.write("SM_KROWS = %d\n" % SM_KROWS)

    if model.moe:
        emit_moe_layout(model, lay, outdir)

    ne = max(model.nexp, model.nexp_head)
    weights = sum(len(m) * len(m[0]) for _, m in model.matrices(0))
    total = weights + (ne - 1) * sum(
        len(m) * len(m[0]) for kind, ms in model.chunks(0) if kind == "E"
        for _, m in ms)
    return {
        "rows": ROWS_PER_TOKEN,
        "nnz": lay["nnz"],
        "nnz_per_token": lay.get("nnz_per_token", lay["nnz"]),
        "weights_per_token": weights,
        "weights_on_cart": total,
        "nexp": ne,
        "stream_banks": lay["banks"],
        "header_bytes": lay["header_bytes"],
        "banks": lay["banks"],
        "bank_crossings_per_token": lay["crossings"],
    }


# ---------------------------------------------------------------------------
# cartridge layout for a mixture build
#
# The bank map, the assembler constants, the .incbin lines and the linker
# config all have to agree about where every expert lives.  They are generated
# from ONE computation here rather than written out four times, because four
# hand-maintained copies of a bank map is precisely the kind of thing that
# produces a ROM which runs and lies.
# ---------------------------------------------------------------------------
TABLE_FILES = ["tbl_mul", "tbl_q2", "tbl_q3", "tbl_q4", "tbl_clamp",
               "tbl_entoff", "tbl_blkcnt", "tbl_step"]
# Eight page-aligned 256-byte tables plus the 15-byte exp table need
# 8 * 256 + 15 = 2,063 bytes, so they start at $D700 and end at $DF0F.  At
# $D800 the exp table overflowed the bank by exactly 15 bytes, which the
# linker caught; it is pinned rather than aligned so that every expert's copy
# lands at the same address whatever its header table's length.
TABLES_START = 0xD700
TABLES_BYTES = 8 * 256 + 15


def emit_moe_layout(model, lay, outdir):
    ne = lay["nexp"]
    nsh, per_exp = lay["shared_banks"], lay["expert_banks"]
    sb = nsh + ne * per_exp                      # first non-stream bank
    emb, pos, hdr0 = sb, sb + 1, sb + 2
    total = hdr0 + ne + 1                        # + the fixed code bank
    if total % 2:                                # iNES counts 16 KB units
        total += 1
    code = total - 1
    if total > 128:
        raise SystemExit("layout needs %d banks; the MMC5 addresses 128" % total)
    if lay["header_bytes"] > TABLES_START - 0xC000:
        raise SystemExit("header table is %d bytes, the bank holds %d before "
                         "the lookup tables at $%04X"
                         % (lay["header_bytes"], TABLES_START - 0xC000,
                            TABLES_START))
    if TABLES_START + TABLES_BYTES > 0xE000:
        raise SystemExit("the lookup tables do not fit above $%04X" % TABLES_START)

    with open(os.path.join(outdir, "moe.inc"), "w") as f:
        f.write("; generated by host/ref.py - do not edit\n")
        f.write("NEXPERT   = %d\n" % ne)
        f.write("NSHAREDB  = %d\n" % nsh)
        f.write("NEXPBANK  = %d\n" % per_exp)
        f.write("NSTREAM   = %d\n" % sb)
        f.write("EMBBANK   = $%02X\n" % (0x80 + emb))
        f.write("POSBANK   = $%02X\n" % (0x80 + pos))
        f.write("HDRBANK0  = $%02X\n" % (0x80 + hdr0))
        f.write("TBLBANK   = HDRBANK0\n")
        f.write("CODEBANK  = $%02X\n" % (0x80 + code))
        f.write("NPRGUNITS = %d\n" % (total // 2))

    # route -> header bank, indexed by token id.  This IS the router.
    with open(os.path.join(outdir, "routebank.bin"), "wb") as f:
        f.write(bytes(0x80 + hdr0 + model.route[t] for t in range(V)))

    with open(os.path.join(outdir, "moebanks.inc"), "w") as f:
        f.write("; generated by host/ref.py - do not edit\n")
        for b in range(sb):
            f.write('.segment "STREAM%d"\n' % b)
            f.write('    .incbin "out/model/stream.bin", $%X, $2000\n'
                    % (b * BANK))
        for e in range(ne):
            f.write('.segment "HEADERS%d"\n' % e)
            f.write('    .incbin "out/model/headers_e%d.bin"\n' % e)
            f.write('.segment "TABLES%d"\n' % e)
            sfx = "" if e == 0 else "_e%d" % e
            for t in TABLE_FILES:
                f.write("    .align $100\n")
                f.write('%s%s: .incbin "out/model/%s.bin"\n' % (t, sfx, t))
            f.write('tbl_exp%s: .incbin "out/model/tbl_exp.bin"\n' % sfx)
            if e:
                # Every expert's copy must land at the SAME address or a token
                # routed to expert e reads its multiply table out of thin air.
                # Asserted at assembly time, not hoped for.
                for t in TABLE_FILES + ["tbl_exp"]:
                    f.write('    .assert %s_e%d = %s, error, '
                            '"expert %d table copy moved"\n' % (t, e, t, e))

    with open(os.path.join(outdir, "nnmoe.cfg"), "w") as f:
        f.write("# generated by host/ref.py - do not edit\n")
        f.write("MEMORY {\n")
        f.write("    ZP:     start = $0000, size = $0100, type = rw, define = yes;\n")
        f.write("    RAM:    start = $0200, size = $0100, type = rw, define = yes;\n")
        f.write("    HDR:    start = $0000, size = $0010, type = ro, file = %O, fill = yes;\n")
        for b in range(sb):
            f.write("    WS%d: start = $8000, size = $2000, type = ro, "
                    "file = %%O, fill = yes, fillval = $00;\n" % b)
        f.write("    EMB:    start = $A000, size = $2000, type = ro, file = %O, fill = yes, fillval = $00;\n")
        f.write("    POS:    start = $A000, size = $2000, type = ro, file = %O, fill = yes, fillval = $00;\n")
        for e in range(ne):
            f.write("    HB%d: start = $C000, size = $2000, type = ro, "
                    "file = %%O, fill = yes, fillval = $00;\n" % e)
        for i in range(total - hdr0 - ne - 1):
            f.write("    SPARE%d: start = $C000, size = $2000, type = ro, "
                    "file = %%O, fill = yes, fillval = $00;\n" % i)
        f.write("    PRGF:   start = $E000, size = $2000, type = ro, file = %O, fill = yes, fillval = $FF;\n")
        f.write("    PRAM:   start = $8000, size = $1000, type = rw;\n")
        f.write("    CHRROM: start = $0000, size = $2000, type = ro, file = %O, fill = yes, fillval = $00;\n")
        f.write("}\n\nSEGMENTS {\n")
        f.write("    HEADER:   load = HDR,  type = ro;\n")
        for b in range(sb):
            f.write("    STREAM%d:  load = WS%d, type = ro;\n" % (b, b))
        f.write("    EMBED:    load = EMB,  type = ro,  start = $A000;\n")
        f.write("    POS:      load = POS,  type = ro,  start = $A000;\n")
        for e in range(ne):
            f.write("    HEADERS%d: load = HB%d, type = ro,  start = $C000;\n" % (e, e))
            f.write("    TABLES%d:  load = HB%d, type = ro,  start = $%04X;\n"
                    % (e, e, TABLES_START))
        f.write("    SMTABLES: load = EMB,  type = ro,  align = $100;\n")
        f.write("    RAMKERN:  load = EMB,  run = PRAM, type = ro, define = yes;\n")
        f.write("    CHAINS:   load = PRGF, type = ro,  start = $E000;\n")
        f.write("    CODE:     load = PRGF, type = ro;\n")
        f.write("    VECTORS:  load = PRGF, type = ro,  start = $FFFA;\n")
        f.write("    ZEROPAGE: load = ZP,   type = zp;\n")
        f.write("    BSS:      load = RAM,  type = bss, define = yes;\n")
        f.write("}\n")
    lay["cart_banks"] = total


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "out/model"
    ntok = int(sys.argv[2]) if len(sys.argv) > 2 else T - 1
    npz = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("NES_WEIGHTS")
    seed_tok = int(os.environ.get("NES_SEED_TOK", "1"))
    os.makedirs(outdir, exist_ok=True)
    m = Model.from_npz(npz) if npz else Model()
    info = pack(m, outdir)
    toks, r = generate(m, seed_tok, ntok)
    info["tokens"] = toks
    info["ntok"] = ntok
    info["weights_npz"] = npz or "(random init)"
    info["seed_tok"] = seed_tok
    with open(os.path.join(outdir, "expected.json"), "w") as f:
        json.dump({"info": info, "tokens": toks,
                   "trace": [{k: v for k, v in s.items()
                              if k in ("tok", "pos", "next", "x0",
                                       "L0.x", "L1.x", "L2.x", "logits0_8",
                                       "att", "q", "scores", "p")}
                             for s in r.trace]}, f, indent=1)
    for k, v in info.items():
        print("%-28s %s" % (k, v))


if __name__ == "__main__":
    main()
