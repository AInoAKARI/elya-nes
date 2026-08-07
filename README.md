# elya-nes

A cycle-exact measurement instrument for the NES/Famicom 2A03, and a ternary
transformer forward pass built on top of it, verified bit-exact against a host
reference over 19 generated tokens - now with **trained weights**, so it says
something.

```
seed token 'b'      -> 'big friends. she was so hap'
seed token ', '     -> ', he saw a big dad and said'
seed token 'en'     -> 'enture. he was so happy and sai'
seed token '!'      -> '! he was so happy. she was so '
```

102,400 ternary weights (52,207 nonzero), 4-bit activations, a 64-symbol
vocabulary, trained on TinyStories with quantisation-aware training so the
forward pass the trainer sees is the forward pass the 6502 executes. 0.689
seconds per token on a 1.79 MHz 2A03.

Every number in `FINDINGS.md` is measured. Nothing here is estimated.

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
| bank crossings per token | **6** (36 cycles, 0.003% of a token) |
| ternary kernel | **10.688 cycles/MAC** asymptotic vs the 8-cycle primitive |
| ROM vs host reference | **19/19 tokens EXACT**, at three independent seed tokens: **57/57** |
| trained model | val **2.0546 nats/token = 1.4133 nats/char** (uniform 4.1589) |
| nonzero weights | **52,207** of 102,400 (density 0.5098) |
| cycles per token (trained) | 1,103,615 (pos 0) .. 1,366,939 (pos 18), mean **1,233,099** |
| cycles per token (random init) | mean **1,221,027** - trained costs +0.99% for +1.8% nnz |
| wall clock at 1,789,772 Hz | **0.6890 s/token** |
| independent emulator | ares 147 gives the **identical** 19 tokens (random-init build) |

## Layout

```
rom/     common.inc  marker protocol + deterministic machine init
         calib.s     28-payload datasheet calibration ROM (NROM)
         prim.s      MMC5 primitives: bank switch, $6000, accumulators, ternary
         mmc1.s      MMC1 bank switch cost, on a real MMC1 cart
         mmc3.s      MMC3 bank switch cost, on a real MMC3 cart
         nn.s        the transformer: 32 gather chains + forward pass
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
tools/   nes_bench.py      the instrument (write tap, no polling, GC-safe)
         run_calib.py      datasheet calibration report
         run_prim.py       primitive report
         run_nn.py         ROM vs host, token by token, plus cycles/token
         run_profile.py    per-stage cycle profile
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
| `-DDEBUG -DDBGPOS=n` | `nndbg.nes` | dumps intermediate state at position n |
| `-DSEEDTOK=n` | | the seed token the ROM free-runs from (default 1) |
| `-DNSTREAM=9` + `rom/nn9.cfg` | | 9 weight-stream banks for a model denser than 0.559 |

The requantise shifts (`KSHIFT`, `W2SHIFT`, `AVSHIFT`, `SMSHIFT`) are
**generated** by `host/ref.py` into `out/model/shifts.inc` and included by
`rom/nn.s`, so the kernel and the specification cannot hold different values.
