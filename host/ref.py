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
T  = 20          # context positions

BLOCK    = 16    # gather block size; 16*14 = 224 < 256 keeps the carry clear
BIAS     = 7     # activations stored as value+7 in 0..14

K_SHIFT  = int(os.environ.get("NES_K_SHIFT", "2"))   # Wq/Wk/Wv/Wo/W1
W2_SHIFT = int(os.environ.get("NES_W2_SHIFT", "3"))  # W2 (128 inputs)
AV_SHIFT = int(os.environ.get("NES_AV_SHIFT", "1"))  # attention value sum
SM_SHIFT = int(os.environ.get("NES_SM_SHIFT", "3"))  # score difference -> exp

# AV_SHIFT was 4 in the first cut of this port and that was measurably wrong.
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
                "W1": gen_ternary(rng, F, D),
                "W2": gen_ternary(rng, D, F),
            })
        self.head = gen_ternary(rng, V, D)

    @classmethod
    def from_npz(cls, path):
        """Load trained integer weights.  The npz is the SAME array the trainer
        exported and the same one the max|dW| check reads, so the stream the
        packer writes and the numbers the reference runs are provably one
        quantisation, not two that happen to agree."""
        import numpy as np
        z = np.load(path)
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
        m.layers = []
        for l in range(L):
            d = {}
            for nm in ("Wq", "Wk", "Wv", "Wo", "W1", "W2"):
                a = z["L%d_%s" % (l, nm)]
                d[nm] = [[int(v) for v in row] for row in a]
                assert set(v for row in d[nm] for v in row) <= {-1, 0, 1}
            m.layers.append(d)
        m.head = [[int(v) for v in row] for row in z["head"]]
        assert len(m.head) == V and len(m.head[0]) == D
        return m

    # -- the ordered list of matrices exactly as the stream stores them -----
    def matrices(self):
        out = []
        for l in range(L):
            for name in ("Wq", "Wk", "Wv", "Wo", "W1", "W2"):
                out.append(("L%d.%s" % (l, name), self.layers[l][name]))
        out.append(("head", self.head))
        return out


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
        self.split = {}
        for name, mat in model.matrices():
            self.split[name] = [split_row(r) for r in mat]
        # KV cache holds NIBBLES, exactly as the $6000 window does
        self.K = [[[0] * D for _ in range(T)] for _ in range(L)]
        self.Vc = [[[0] * D for _ in range(T)] for _ in range(L)]
        self.trace = []

    def step(self, tok, p):
        m = self.m
        x = [clamp7(m.emb[tok][j] + m.pos[p][j]) for j in range(D)]
        stage = {"tok": tok, "pos": p, "x0": list(x)}

        for l in range(L):
            q = matmul(self.split["L%d.Wq" % l], x, K_SHIFT)
            kv = matmul(self.split["L%d.Wk" % l], x, K_SHIFT)
            vv = matmul(self.split["L%d.Wv" % l], x, K_SHIFT)
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
            o = matmul(self.split["L%d.Wo" % l], att, K_SHIFT)
            x = [clamp7(x[j] + o[j]) for j in range(D)]
            hdn = matmul(self.split["L%d.W1" % l], x, K_SHIFT, relu=True)
            f = matmul(self.split["L%d.W2" % l], hdn, W2_SHIFT)
            x = [clamp7(x[j] + f[j]) for j in range(D)]
            stage["L%d.x" % l] = list(x)

        logits = matmul(self.split["head"], x, 0, raw=True)
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
def nnz_of(rows):
    return sum(len(p) + len(n) for p, n in rows)


def pack(model, outdir):
    rows = []                       # (pos_idx, neg_idx) in stream order
    for name, mat in model.matrices():
        for r in mat:
            rows.append(split_row(r))

    # Each bank opens with BLOCK bytes of padding.  The ROM's gather chains
    # address the stream as (offset - BLOCK) so that the chain's base address
    # is page-aligned; without the pad that quantity goes negative at the start
    # of a bank and the chain pointer runs off the front of the chain table.
    # With the base page-aligned the indexed load is 4 cycles instead of 5 -
    # a straight 1 cycle per MAC.
    #
    # bank_off is tracked EXPLICITLY rather than as len(stream) % BANK: a row
    # that ends exactly on a bank boundary makes the modulo read 0, the pad is
    # then never emitted, and the ROM's pointer underflows into chain -1 and
    # executes garbage.  That is exactly what happened the first time.
    stream = bytearray(b"\x00" * BLOCK)
    bank_off = BLOCK
    headers = bytearray()
    crossings = 0
    for p, n in rows:
        need = len(p) + len(n)
        if bank_off + need > BANK:
            headers += bytes([SENTINEL, 0, 0, 0])
            stream += b"\x00" * (BANK - bank_off)     # fill out this bank
            stream += b"\x00" * BLOCK                 # open the next one
            bank_off = BLOCK
            crossings += 1
        d7 = -BIAS * (len(p) - len(n))
        headers += bytes([len(p), len(n), d7 & 0xFF, (d7 >> 8) & 0xFF])
        stream += bytes(p) + bytes(n)
        bank_off += need
    if len(stream) % BANK:
        stream += b"\x00" * (BANK - len(stream) % BANK)

    # rom/nn.s incbins the stream at FIXED offsets 0, $2000 ... so the image
    # has to be exactly STREAM_BANKS banks long whatever the model's sparsity.
    # A trained model is sparser than the 50%-dense random init, so without
    # this the last incbin reads past the end of the file; a denser one would
    # silently lose its tail rows, which is far worse.
    want = STREAM_BANKS * BANK
    if len(stream) > want:
        raise SystemExit(
            "stream needs %d banks but rom/nn.cfg provides %d.  nnz=%d "
            "(density %.4f); the 7-bank window caps density at about %.4f."
            % (len(stream) // BANK, STREAM_BANKS, nnz_of(rows),
               nnz_of(rows) / float(sum(len(m) * len(m[0])
                                        for _, m in model.matrices())),
               (want - STREAM_BANKS * BLOCK) /
               float(sum(len(m) * len(m[0]) for _, m in model.matrices()))))
    stream += b"\x00" * (want - len(stream))

    def w(name, data):
        with open(os.path.join(outdir, name), "wb") as f:
            f.write(bytes(data))

    w("stream.bin", stream)
    w("headers.bin", headers)
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

    nnz = sum(len(p) + len(n) for p, n in rows)
    return {
        "rows": len(rows),
        "nnz": nnz,
        "weights": sum(len(m) * len(m[0]) for _, m in model.matrices()),
        "stream_bytes": len(stream),
        "header_bytes": len(headers),
        "banks": len(stream) // BANK,
        "bank_crossings_per_token": crossings,
    }


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "out/model"
    ntok = int(sys.argv[2]) if len(sys.argv) > 2 else 19
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
