#!/usr/bin/env python3
"""Differentiable twin of host/ref.py.

The forward pass here is the NES reference forward pass, op for op, evaluated
on integer-valued float tensors.  Every non-differentiable step (ternarise,
requantise, floored product, quantised softmax) runs the EXACT integer op in
the forward direction and a matched linear surrogate in the backward direction
via a straight-through estimator.

This is why the trainer could not simply be the sibling N64 QAT trainer: that
model has float attention, per-32-block fp16 weight scales and an int8 tied
head over a 256 vocab.  The NES kernel has NO scales anywhere - a weight is
literally -1, 0 or +1 and an activation is literally an integer in -7..7 - so
the output magnitude is set by the fixed requantise shift and the row's
nonzero count, and nothing else.  The STE helpers below are the sibling
trainer's; the model is this port's.

Post-training ternarisation is not an option and was not tried: on the sibling
N64 model PTQ of a trained fp32 artifact scored a fit loss of 6.63 on a
byte-level corpus where uniform random is ln(256) = 5.545 - worse than random -
while QAT of the identical artifact reached 0.1012 against an fp32 control's
0.1007.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- shape: settled, see DESIGN.md -----------------------------------------
V, D, L, H, DH, FF, T = 64, 64, 3, 2, 32, 128, 20
K_SHIFT, W2_SHIFT, AV_SHIFT, SM_SHIFT = 2, 3, 4, 3
ACT_MAX = 7
MUL_SHIFT = 2          # the mul table is floor(q*k / 4)
SM_TEMP = 16.0         # exp table is ~64*exp(d/2) with d = floor(ds/8)
SM_SUM = 8.0           # power-of-two normalisation targets a sum of <= 8


def ste(soft, hard):
    """forward = hard (exact integer), backward = d(soft)."""
    return soft + (hard - soft).detach()


# ---------------------------------------------------------------------------
# exact integer primitives (float tensors holding integers)
# ---------------------------------------------------------------------------
def q_exact(acc, k):
    """clamp(floor(acc / 2^k), -7, 7) - identical to ref.quant(acc, k)."""
    return torch.clamp(torch.floor(acc / (1 << k)), -ACT_MAX, ACT_MAX)


def q_ste(acc, k):
    soft = torch.clamp(acc / (1 << k), -ACT_MAX, ACT_MAX)
    return ste(soft, q_exact(acc, k))


def floor_prod(a, b, k=MUL_SHIFT):
    """The 256-byte mul table: floor(a*b / 4), with a linear surrogate."""
    p = a * b
    return ste(p / (1 << k), torch.floor(p / (1 << k)))


def softmax_q(scores, mask, exptab):
    """Quantised softmax, bit-identical to ref.softmax_q.

    scores : (..., T) integer-valued
    mask   : (..., T) bool, True where the position is attendable
    """
    neg = torch.full_like(scores, -1e6)
    s = torch.where(mask, scores, neg)
    m = s.amax(-1, keepdim=True)
    d = torch.clamp(torch.floor((s - m) / (1 << SM_SHIFT)), min=-14.0)
    idx = (d + 14).long().clamp(0, 14)
    e = exptab[idx]                                   # exact table lookup
    S = e.sum(-1, keepdim=True)
    # kk = smallest k with (S >> k) <= 8.  floor(S / 2^k) <= 8  <=>  2^k >= ...
    kk = torch.zeros_like(S)
    cur = S.clone()
    for _ in range(12):                               # S <= 20*64 = 1280 < 2^11
        step = (torch.floor(cur / 2 ** kk) > SM_SUM).float()
        kk = kk + step
        if step.sum() == 0:
            break
    hard = torch.clamp(torch.floor(e / 2 ** kk), max=float(ACT_MAX))
    hard = torch.where(mask, hard, torch.zeros_like(hard))
    soft = torch.softmax(torch.where(mask, scores, neg) / SM_TEMP, dim=-1) * SM_SUM
    soft = torch.clamp(soft, max=float(ACT_MAX))
    return ste(soft, hard)


# ---------------------------------------------------------------------------
# weight quantisers
# ---------------------------------------------------------------------------
def ternarise(w, tau, mode):
    """No scale is stored anywhere in the ROM, so the quantiser emits bare
    -1/0/+1 and the output magnitude is governed by the row's nonzero count
    against the fixed requantise shift.  tau is therefore both the sparsity
    knob and the gain knob."""
    if mode == "none":
        return w
    a = w.abs()
    if mode == "twn":
        delta = tau * a.mean(dim=-1, keepdim=True)
        q = torch.sign(w) * (a > delta).to(w.dtype)
    elif mode == "bn":                                # BitNet b1.58 absmean
        s = a.mean(dim=-1, keepdim=True).clamp(min=1e-8) * tau
        q = torch.clamp(torch.round(w / s), -1.0, 1.0)
    else:
        raise ValueError(mode)
    return ste(w, q)


def quant_act_tbl(w):
    """embedding / positional entries: integers in -7..7."""
    return ste(torch.clamp(w, -ACT_MAX, ACT_MAX),
               torch.clamp(torch.round(w), -ACT_MAX, ACT_MAX))


# ---------------------------------------------------------------------------
class NesModel(nn.Module):
    def __init__(self, tau=1.0, mode="twn", quant=True, logit_scale=0.05):
        super().__init__()
        self.tau, self.mode, self.quant = tau, mode, quant
        g = torch.Generator().manual_seed(20260807)

        def p(*shape, s=1.0):
            return nn.Parameter(torch.randn(*shape, generator=g) * s)

        self.emb = p(V, D, s=2.0)
        self.pos = p(T, D, s=1.0)
        self.Wq, self.Wk, self.Wv, self.Wo = (nn.ParameterList() for _ in range(4))
        self.W1, self.W2 = nn.ParameterList(), nn.ParameterList()
        for _ in range(L):
            self.Wq.append(p(D, D)); self.Wk.append(p(D, D))
            self.Wv.append(p(D, D)); self.Wo.append(p(D, D))
            self.W1.append(p(FF, D)); self.W2.append(p(D, FF))
        self.head = p(V, D)
        self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale)))

        import numpy as np
        exptab = [max(0, min(64, int(round(64.0 * math.exp((i - 14) / 2.0)))))
                  for i in range(15)]
        self.register_buffer("exptab", torch.tensor(exptab, dtype=torch.float32))
        m = torch.tril(torch.ones(T, T, dtype=torch.bool))
        self.register_buffer("causal", m)

    # -- quantised views ----------------------------------------------------
    def _w(self, p):
        return ternarise(p, self.tau, self.mode if self.quant else "none")

    def _tbl(self, p):
        return quant_act_tbl(p) if self.quant else p

    def _mm(self, x, w, k, relu=False, raw=False):
        acc = x @ w.t()
        if raw:
            return acc
        q = q_ste(acc, k)
        if relu:
            q = torch.clamp(q, min=0.0)
        return q

    # -----------------------------------------------------------------------
    def forward(self, idx):
        """idx: (B, T) int64.  Returns logits (B, T, V)."""
        B, Tn = idx.shape
        emb, pos = self._tbl(self.emb), self._tbl(self.pos)
        x = emb[idx] + pos[:Tn].unsqueeze(0)
        x = torch.clamp(x, -ACT_MAX, ACT_MAX)

        mask = self.causal[:Tn, :Tn].unsqueeze(0).unsqueeze(0)      # (1,1,T,T)

        for l in range(L):
            q = self._mm(x, self._w(self.Wq[l]), K_SHIFT)
            k = self._mm(x, self._w(self.Wk[l]), K_SHIFT)
            v = self._mm(x, self._w(self.Wv[l]), K_SHIFT)

            def heads(t):
                return t.view(B, Tn, H, DH).transpose(1, 2)          # (B,H,T,DH)

            qh, kh, vh = heads(q), heads(k), heads(v)
            # scores[b,h,i,j] = sum_d floor(q[i,d]*k[j,d] / 4)
            sc = floor_prod(qh.unsqueeze(3), kh.unsqueeze(2)).sum(-1)  # (B,H,T,T)
            pr = softmax_q(sc, mask.expand(B, H, Tn, Tn), self.exptab)
            # att[b,h,i,d] = quant(sum_j floor(pr[i,j]*v[j,d] / 4), 4)
            av = floor_prod(pr.unsqueeze(-1), vh.unsqueeze(2))
            av = torch.where(mask.unsqueeze(-1), av, torch.zeros_like(av)).sum(3)
            att = q_ste(av, AV_SHIFT)
            att = att.transpose(1, 2).reshape(B, Tn, D)

            o = self._mm(att, self._w(self.Wo[l]), K_SHIFT)
            x = torch.clamp(x + o, -ACT_MAX, ACT_MAX)
            h = self._mm(x, self._w(self.W1[l]), K_SHIFT, relu=True)
            f = self._mm(h, self._w(self.W2[l]), W2_SHIFT)
            x = torch.clamp(x + f, -ACT_MAX, ACT_MAX)

        # RAW integer logits, exactly what the ROM argmaxes.  The training
        # temperature is applied by the caller so that this function stays a
        # bit-exact twin of ref.Runner.step.
        return self._mm(x, self._w(self.head), 0, raw=True)

    # -----------------------------------------------------------------------
    @torch.no_grad()
    def renorm_(self):
        """Keep every ternary matrix at unit mean-|w|.  The quantiser is scale
        invariant (delta is proportional to mean|w|), so this only stops the
        latent weights from drifting numerically; it changes no forward value."""
        for pl in (self.Wq, self.Wk, self.Wv, self.Wo, self.W1, self.W2):
            for p in pl:
                p.data /= p.data.abs().mean().clamp(min=1e-8)
        self.head.data /= self.head.data.abs().mean().clamp(min=1e-8)

    @torch.no_grad()
    def export_int(self):
        """Integer arrays exactly as host/ref.py wants them."""
        import numpy as np
        out = {}
        out["emb"] = torch.clamp(torch.round(self.emb), -ACT_MAX, ACT_MAX).cpu().numpy().astype(np.int8)
        out["pos"] = torch.clamp(torch.round(self.pos), -ACT_MAX, ACT_MAX).cpu().numpy().astype(np.int8)
        for l in range(L):
            for nm, pl in (("Wq", self.Wq), ("Wk", self.Wk), ("Wv", self.Wv),
                           ("Wo", self.Wo), ("W1", self.W1), ("W2", self.W2)):
                w = pl[l]
                a = w.abs()
                if self.mode == "twn":
                    delta = self.tau * a.mean(dim=-1, keepdim=True)
                    q = torch.sign(w) * (a > delta).to(w.dtype)
                else:
                    s = a.mean(dim=-1, keepdim=True).clamp(min=1e-8) * self.tau
                    q = torch.clamp(torch.round(w / s), -1.0, 1.0)
                out["L%d_%s" % (l, nm)] = q.cpu().numpy().astype(np.int8)
        w = self.head
        a = w.abs()
        if self.mode == "twn":
            delta = self.tau * a.mean(dim=-1, keepdim=True)
            q = torch.sign(w) * (a > delta).to(w.dtype)
        else:
            s = a.mean(dim=-1, keepdim=True).clamp(min=1e-8) * self.tau
            q = torch.clamp(torch.round(w / s), -1.0, 1.0)
        out["head"] = q.cpu().numpy().astype(np.int8)
        return out
