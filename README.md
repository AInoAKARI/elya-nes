# elya-nes

> **Part of [Transformers on Retro Game Consoles](https://hackaday.io/project/206401-transformers-on-retro-game-consoles)** on Hackaday.io — build logs, corrections, and the sibling NES / SNES / Genesis / N64 / Game Boy Color ports.

A cycle-exact measurement instrument for the NES/Famicom 2A03, and a ternary
transformer forward pass built on top of it, verified bit-exact against a host
reference over 19 generated tokens - now with **trained weights**, so it says
something.

```
seed token 'b'      -> 'because he was so happy and sa'
seed token 'd'      -> 'day, they had a big boy na'
seed token 'j'      -> 'jack for a long friend. '
seed token 'l'      -> 'lily was very happy and saw a b'
```

**446,464** ternary weights on the cartridge, **102,400** of them streamed per
token, 4-bit activations, a 64-symbol vocabulary, trained on TinyStories with
quantisation-aware training so the forward pass the trainer sees is the
forward pass the 6502 executes. 0.634 seconds per token on a 1.79 MHz 2A03.

Every number in `FINDINGS.md` is measured. Nothing here is estimated.

### The two wins compose: 8 experts AND an exact softmax normaliser

Two lines of work landed on this tree from separate branches - an exact
softmax normaliser (below) and a mixture of experts (below that) - and the
obvious question is whether their gains add or get in each other's way. All
four cells of the 2 x 2 were retrained here, 60,000 steps, two seeds each, so
that the comparison is one tree and one estimator:

| val nats/char | power-of-two normaliser | **exact normaliser** |
|---|---:|---:|
| dense | 1.4020 / 1.4096 = 1.4058 | 1.3754 / 1.3779 = 1.3766 |
| **8 experts** | 1.2202 / 1.2221 = 1.2211 | 1.1932 / 1.1898 = **1.1915** |

```
A, exact normalisation:  -0.0292 at dense capacity,  -0.0296 at 8 experts
B, 8 experts:            -0.1847 under pow2,         -0.1851 under exact
additive prediction   1.4058 -0.0292 -0.1847 = 1.1920
measured                                        **1.1915**
interaction                                       -0.0005   (seed spread 0.0076)
```

**They add.** The interaction is inside the seed spread under both estimators
this repo uses, and it has the sign of very slightly *more* than additive.
The cycles add too: 1,117,248 -> 1,129,375 (A) -> 1,134,432 (both), against an
additive prediction of 1,135,265.

| | dense + pow2 | **merged cartridge** |
|---|---:|---:|
| val nats/char, seed 1 / seed 2 | 1.4020 / 1.4096 | **1.1932 / 1.1898** |
| ternary weights on the cartridge | 102,400 | **446,464** |
| ternary weights streamed per token | 102,400 | 102,400 |
| mean cycles/token | 1,117,248 | **1,134,432** (+1.54%) |
| seconds/token | 0.6243 | **0.6338** |
| image | 106,512 B, 12 banks | 548,880 B, 66 banks |
| ROM vs host, 64-seed survey | 1,216/1,216 | **1,216/1,216**, both seeds: **2,432/2,432** |

One correction falls out of doing this on one tree. The softmax result below
was published as **-0.0375** nats/char against a dense baseline of 1.4141 that
came from two older arms; against a dense control trained here the same effect
is **-0.0292**, and -0.0296 at 8 experts. The effect is real, non-overlapping
and reproduces at both capacities - it is 0.008 smaller than published, and
the reason is the baseline, not the mechanism. Full arithmetic in
`FINDINGS.md`.

### The integer softmax was throwing away a quarter of its budget

The quantised softmax normalised with a power-of-two shift: `kk` is the
smallest shift with `S >> kk <= 8`, so the **realised** sum landed anywhere in
`(4, 8]` and a quarter of the time the softmax was running on half its
budget. Replacing that with an exact normalisation - `p = min(e*8/S, 7)`,
the same three-bit nibble, the same budget - and re-laddering `AV_SHIFT`,
which moves from 2 to 3 once the accumulator is bigger:

| | shipped | exact | **exact + AV_SHIFT 3** |
|---|---|---|---|
| val nats/char, seed 1 / seed 2 | 1.4133 / 1.4149 | 1.3957 / 1.3848 | **1.3754 / 1.3779** |
| mean | 1.4141 | 1.3902 | **1.3766** (**-0.0375, 2.65%**) |

(The 1.4141 baseline is `runs/final_av2_bpe64_tau0.75` and `runs/t20_final_s2`,
older arms from a different implementation of the same forward pass. Against a
dense + pow2 control trained on the merged tree the effect is -0.0292; see the
2 x 2 above.)
| mean cycles/token | 1,116,979 | 1,121,121 | **1,129,291** (+1.10%) |
| ROM vs host | 1,216/1,216 (64 seeds) | 1,216/1,216 | **1,216/1,216 (64 seeds)** |

60,000 steps, two seeds each, ranges nowhere near overlapping. That is two
thirds of what ternarising the weights costs, recovered from a normaliser and
a shift. (Of the +1.10% cycles, +4,223 at the last position is the kernel;
the rest is that this model trains to 557 more nonzero weights.)

**Widening the probability nibble - which is what the previous experiment's
diagnosis called for - is worth nothing.** Budgets of 8, 16 and 32 measure
1.5279, 1.5335 and 1.5267 nats/char at matched steps: not monotone, and every
gap smaller than the noisier arms' own seed spread. The leak was the shared
exponent, not the bit width. See `FINDINGS.md`.

**And fixing it did not make longer context help.** The `T = 85` minus
`T = 20` penalty is +0.0472 nats/char under the old normaliser and +0.0487
under the new one - the gap moved by 0.0015, inside the seed spread. The
softmax was not what stood between the long-range head and paying for itself.
The ceiling is still capacity.

### Is 29 characters of context the reason it cannot form a sentence? No.

`T` is now a build parameter and the cartridge was rebuilt at the largest
context its 32 KB of PRG-RAM can hold - **85 tokens, ~124 characters**, the KV
cache spread across all four banks. Retrained on the identical recipe:

| | T = 20 | T = 85 |
|---|---|---|
| val nats/char, seed 1 / seed 2 | **1.4133 / 1.4149** | **1.4347 / 1.4318** (worse) |
| ROM vs host | 1,216/1,216 tokens exact (64 seeds) | **252/252 tokens exact** (3 seeds) |
| mean cycles/token | 1,116,979 | **1,697,916** (+52%) |
| attention share, last position | 7.3% | **52.9%** |
| seconds/token, last position | 0.641 | **1.283** |

(`T = 20` runs the attention kernels; `T = 85` is past their addressing
ceiling and runs the legacy attention path, which is why the cycle gap is
wider than the context ratio. Both columns are measured on this tree.)

Two seeds each, and **the two groups do not overlap**: the worst `T = 20` run
beats the best `T = 85` run by 0.0169 nats/char, six times the within-context
seed spread. The attention did learn to reach - layer 2's mean attention
distance went from 0.91 to 7.82 and 13.7% of its mass now lands beyond what
`T = 20` could see - and the loss got **worse** anyway. Held-out loss is flat from about position 10
(~15 characters) in every model that has room to show it, and at matched
positions the short-context model wins everywhere. **The ceiling is capacity,
not context.** Full argument in `FINDINGS.md`.

### So capacity was bought instead - and it worked

A flash cartridge holds far more than an original NES cart, so the mixture
build keeps attention, the embedding and the head shared and gives the
feed-forward block of every layer **N copies**, routed by a 64-byte table on
the current token id.  One expert streams per token, so the weights the 6502
walks stay at 102,400 while the weights on the cartridge do not.

| | dense | 8-expert mixture |
|---|---|---|
| ternary weights on the cartridge | 102,400 | **446,464** |
| ternary weights streamed per token | 102,400 | **102,400** |
| val nats/char, seed 1 / seed 2 | 1.4020 / 1.4096 | **1.2202 / 1.2221** |
| mean cycles/token | 1,117,063 | **1,123,138** (+0.54%) |
| seconds/token | 0.6241 | **0.6275** |
| image | 106,512 B, 12 banks | 548,880 B, 66 banks |
| ROM vs host, 64-seed survey | 1,216/1,216 EXACT | **1,216/1,216 EXACT** |

Two seeds each and the groups are nowhere near overlapping: the worst mixture
seed beats the best dense seed by **0.180 nats/char**, twenty times the 0.009
seed noise.  Quadrupling the context cost 52% more cycles and made the model
0.019 nats/char **worse**; quadrupling the parameters cost 0.55% more cycles
and made it 0.182 nats/char **better**.

The routing table's construction turned out not to matter - balanced,
bigram-clustered and outright random assignments land within 0.0032 nats/char
of each other - so the router is one `lda routebank,x`, measured at **11
cycles**.

And the ceiling was built.  `rom/bankprobe.s` maps a 1 MB image and confirms
**127/127** switchable MMC5 banks answer, which is room for **16 experts**:

| N | weights on cart | image | **val nats/char** | cycles/token | ROM vs host |
|---:|---:|---:|---:|---:|---|
| 1 | 102,400 | 106,512 B | 1.4020 / 1.4096 | 1,117,063 | 1,216/1,216 |
| 8 | 446,464 | 548,880 B | **1.2202 / 1.2221** | 1,123,138 | 1,216/1,216 |
| 16 | 839,680 | **1,007,632 B** | **1.1671** (1 seed) | 1,125,463 | 1,216/1,216 |

**A one-megabyte NES cartridge running an 8.2x model for 0.76% more time.**

## Quick start

```sh
./build.sh                                     # model binaries + every ROM
python3 tools/check_branches.py out/calib.nes out/calib.lbl
python3 tools/run_calib.py   out/calib.nes     # instrument calibration, 28/28
python3 tools/run_prim.py    out/prim.nes      # MMC5 primitives, 19/19
python3 tools/run_nn.py      out/nn.nes out/model/expected.json 200
python3 tools/run_profile.py out/nnprof.nes    # per-stage cycle profile
python3 host/blocksize.py                      # block-size saturation
```

### Train a model and put it on the cartridge

```sh
python3 train/prep_corpus.py                   # charset + 64-slot BPE, coverage
python3 train/test_equiv.py                    # trainer forward == ref.py, exactly
sh      train/sweep.sh                         # the arm table
python3 train/train_nes.py --steps 60000 --tau 0.75 --name mymodel
sh      train/verify_trained.sh runs/mymodel.npz "1 26 40"
python3 train/sidebyside.py out/FINAL_VERIFICATION.txt
python3 train/sample.py runs/mymodel.npz --survey
```

`verify_trained.sh` packs, **proves `max|dW| = 0`** by decoding the ROM's own
weight stream back into matrices, assembles, and only then runs MAME. That
order is deliberate: assembling first would turn a silent exporter bug into a
ROM that runs perfectly and says the wrong thing.

## Headline results

| | |
|---|---|
| instrument | MAME 0.277 `nes` + Lua write tap on `$0300`, 1 cycle = 558.73 ns |
| CPU clock | **derived**, not assumed: 1,789,772 Hz (MAME truncates 21477272/12) |
| datasheet calibration | **28/28, 0 mismatches**, bit-identical over 3 runs |
| primitives vs prior run | **19/19 match** |
| bank crossings per token | **6** dense, **13** for the 8-expert mixture - and a switch costs **73 cycles measured in the loop**, not the 6-cycle datasheet store.  949.8 cycles/token, 0.084% |
| ternary kernel | **10.688 cycles/MAC** asymptotic vs the 8-cycle primitive |
| ROM vs host reference, T = 20 | **19/19 tokens EXACT** at every one of 64 seed tokens: **1,216/1,216** per cartridge |
| ROM vs host reference, T = 85 | **84/84 tokens EXACT**, at three independent seed tokens: **252/252** |
| trained model (T = 20) | val **2.0546 nats/token = 1.4133 nats/char** (uniform 4.1589) |
| trained model (T = 85) | val **2.0856 nats/token = 1.4347 nats/char** - longer context, worse |
| **8-expert mixture, power-of-two normaliser** | val **1.7738 / 1.7766 = 1.2202 / 1.2221 nats/char**, two seeds - **13% better than dense, non-overlapping** |
| **SHIPPING: 8 experts + exact normaliser** | val **1.7346 / 1.7296 = 1.1932 / 1.1898 nats/char**, two seeds |
| **shipping cost** | **+1.54% cycles/token** against dense + pow2, 4.36x the parameters on the cartridge |
| **shipping ROM vs host** | **2,432/2,432 tokens EXACT** - 64 seed tokens on each of the two seeds' cartridges, every expert routed in both |
| **do the two wins add?** | **yes**: interaction **-0.0005 nats/char** against a seed spread of 0.0076, and **-833 cycles** against an additive prediction of 1,135,265 |
| nonzero weights (dense, 2026-08-08 arm) | **52,207** of 102,400 (density 0.5098) |
| nonzero weights (shipping mixture) | **233,408** of 446,464 on the cartridge, 53,266 streamed per token |
| cycles per token (T = 20, shipping mixture) | 1,101,416 (pos 0) .. 1,165,451 (pos 18), mean **1,134,432** |
| cycles per token (T = 20, dense + pow2) | mean **1,117,248** |
| attention at full context (T = 20) | **86,142 cycles, 7.3%** of a token (was 302,624, 21.6%) |
| attention kernels | **8.00 cycles/MAC** measured, self-modified operands in PRG-RAM |
| wall clock at 1,789,772 Hz | **0.6338 s/token** at T = 20, shipping mixture (0.6243 dense + pow2) |
| cycles per token (T = 85, legacy attention) | 1,103,387 (pos 0) .. 2,295,963 (pos 83), mean **1,697,916** |
| attention share at full context (T = 85) | **52.9%**, on the legacy attention path - see below |
| independent emulator | ares 147 gives the **identical** 19 tokens (random-init build) |

## Layout

```
rom/     common.inc  marker protocol + deterministic machine init
         calib.s     28-payload datasheet calibration ROM (NROM)
         prim.s      MMC5 primitives: bank switch, $6000, accumulators, ternary
         mmc1.s      MMC1 bank switch cost, on a real MMC1 cart
         mmc3.s      MMC3 bank switch cost, on a real MMC3 cart
         nn.s        the transformer: 32 gather chains + forward pass
         bankprobe.s how many of the MMC5's 128 PRG banks really answer
         *.cfg       one ld65 config per mapper
host/    ref.py      the exact-integer SPECIFICATION and the weight packer
         blocksize.py  block-size saturation measurement
train/   prep_corpus.py   the 64-symbol charset, the BPE, the story-disjoint split
         model_nes.py     differentiable twin of ref.py (STE on every hard op)
         test_equiv.py    proves that twin IS ref.py, at every layer
         train_nes.py     the QAT trainer
         sweep.sh         the arm table
         verify_pack.py   decodes the ROM's stream back out: max|dW| = 0
         build_trained.sh pack -> prove -> assemble
         verify_trained.sh the same, then MAME at several seed tokens
         sidebyside.py    ROM vs host as decoded text
         sample.py        generate with ref.py and detokenise
         table.py         the results table, normalised per character
         perpos.py        held-out loss POSITION BY POSITION
         attnspan.py      how far back the attention actually reaches
         route.py         the four routing tables, all 11 cycles on the 6502
         factorial.sh     the 2 x 2: {dense, 8 experts} x {pow2, exact}
         factorial_table.py  that 2 x 2, and whether the two effects add
         link_variants.sh every build variant, assembled and linked
         eval_npz.py      one held-out estimator for every arm
         replicate_experts.py  N IDENTICAL experts: the mixture's control arm
         expert_coverage.py    proves the survey routes to every expert
         moe_gate.sh      pack, prove, run, cover, survey, profile
         moe_table.py     the mixture comparison table, per character
tools/   nes_bench.py      the instrument (write tap, no polling, GC-safe)
         run_calib.py      datasheet calibration report
         run_prim.py       primitive report
         run_nn.py         ROM vs host, token by token, plus cycles/token
         run_profile.py    per-stage cycle profile
         run_bank_profile.py  bank-switch and router cost, bracketed in situ
         gen_bankstamp.py  stamps every PRG bank for the bank-budget probe
         check_branches.py branch placement verified from RAW ROM BYTES
DESIGN.md   ROM layout, weight stream format, cost model - written FIRST
FINDINGS.md the journal, appended after every discrete result
```

## Build variants of `nn.s`

| define | ROM | purpose |
|---|---|---|
| (none) | `nn.nes` | the clean build; this is what the timing numbers come from |
| `-DPROFILE` | `nnprof.nes` | X-preserving 12-cycle stage markers |
| `-DBENCH` | `nnbench.nes` | drives the real `gather` over synthetic list lengths |
| `-DATTNPROF` | `nnattn.nes` | nested markers splitting attention into QK / softmax / AV (`NCTX <= 21`) |
| `-DATTNBENCH` | `nnabench.nes` | isolated slope of the attention kernels, self-modified vs pointer (`NCTX <= 21`) |
| `-DRAMEXEC` | `ramexec.nes` | probes whether MMC5 PRG-RAM at `$8000` is writable and executable (`NCTX <= 21`) |
| `-DBANKPROF` | `nnbank.nes` | brackets the weight-stream bank switch and the router in situ - this is where 73 and 11 cycles come from |
| `-DMOE` + `out/model/nnmoe.cfg` | `nn.nes` | mixture-of-experts build.  Both the define and the linker config are decided by `train/build_trained.sh` from whether the packer emitted `out/model/moe.inc`, which it does whenever the npz carries `_moe` |
| `-DDEBUG -DDBGPOS=n` | `nndbg.nes` | dumps intermediate state at position n (`NCTX <= 21` only) |
| `-DSEEDTOK=n` | | the seed token the ROM free-runs from (default 1) |
| `-DNCTX=n` | | context length; **must match `NES_T`** for `host/ref.py`. **21** is the ceiling for the attention kernels (`64 / L`, the key cache row); above it the ROM builds on the legacy attention path, up to **85** (`32768 / (L*2*D)`) |
| `-DNSTREAM=9` + `rom/nn9.cfg` | | 9 weight-stream banks for a model denser than 0.559 |

`NES_T` is passed to `build.sh` / `train/build_trained.sh` and reaches the
packer, the trainer and the assembler from that one place. The trainer stamps
`T` into the npz and `ref.Model.from_npz` refuses to load a mismatch, for the
same reason it refuses a requantise-shift mismatch.

The requantise shifts (`KSHIFT`, `W2SHIFT`, `AVSHIFT`, `SMSHIFT`) are
**generated** by `host/ref.py` into `out/model/shifts.inc` and included by
`rom/nn.s`, so the kernel and the specification cannot hold different values.
