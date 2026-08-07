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

K_SHIFT  = 2     # requantise shift for Wq/Wk/Wv/Wo/W1
W2_SHIFT = 3     # requantise shift for W2 (128 inputs)
AV_SHIFT = 4     # requantise shift for the attention value sum
SM_SHIFT = 3     # score-difference shift feeding the exp table

BANK = 8192      # PRG bank size
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
            for h in range(H):
                base = h * DH
                scores = []
                for t in range(p + 1):
                    s = 0
                    for j in range(DH):
                        s += MUL[(nib(q[base + j]) << 4) | self.K[l][t][base + j]]
                    scores.append(s - MUL_BIAS * DH)
                pr = softmax_q(scores)
                for j in range(DH):
                    s = 0
                    for t in range(p + 1):
                        s += MUL[(nib(pr[t]) << 4) | self.Vc[l][t][base + j]]
                    s -= MUL_BIAS * (p + 1)
                    att[base + j] = quant(s, AV_SHIFT)

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
def pack(model, outdir):
    rows = []                       # (pos_idx, neg_idx) in stream order
    for name, mat in model.matrices():
        for r in mat:
            rows.append(split_row(r))

    stream = bytearray()
    headers = bytearray()
    crossings = 0
    for p, n in rows:
        need = len(p) + len(n)
        off = len(stream) % BANK
        if off + need > BANK:
            # a row must never straddle a bank: pad and emit the sentinel
            headers += bytes([SENTINEL, 0, 0, 0])
            stream += b"\x00" * (BANK - off)
            crossings += 1
        d7 = -BIAS * (len(p) - len(n))
        headers += bytes([len(p), len(n), d7 & 0xFF, (d7 >> 8) & 0xFF])
        stream += bytes(p) + bytes(n)
    if len(stream) % BANK:
        stream += b"\x00" * (BANK - len(stream) % BANK)

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
    os.makedirs(outdir, exist_ok=True)
    m = Model()
    info = pack(m, outdir)
    toks, r = generate(m, 1, ntok)
    info["tokens"] = toks
    info["ntok"] = ntok
    with open(os.path.join(outdir, "expected.json"), "w") as f:
        json.dump({"info": info, "tokens": toks,
                   "trace": [{k: v for k, v in s.items()
                              if k in ("tok", "pos", "next", "x0",
                                       "L0.x", "L1.x", "L2.x", "logits0_8")}
                             for s in r.trace]}, f, indent=1)
    for k, v in info.items():
        print("%-28s %s" % (k, v))


if __name__ == "__main__":
    main()
