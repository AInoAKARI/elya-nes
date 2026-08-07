# elya-nes

A cycle-exact measurement instrument for the NES/Famicom 2A03, and a ternary
transformer forward pass built on top of it, verified bit-exact against a host
reference over 19 generated tokens.

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

## Headline results

| | |
|---|---|
| instrument | MAME 0.277 `nes` + Lua write tap on `$0300`, 1 cycle = 558.73 ns |
| CPU clock | **derived**, not assumed: 1,789,772 Hz (MAME truncates 21477272/12) |
| datasheet calibration | **28/28, 0 mismatches**, bit-identical over 3 runs |
| primitives vs prior run | **19/19 match** |
| bank crossings per token | **6** (36 cycles, 0.003% of a token) |
| ternary kernel | **10.688 cycles/MAC** asymptotic vs the 8-cycle primitive |
| ROM vs host reference | **19/19 tokens EXACT** |
| cycles per token | 1,090,397 (pos 0) .. 1,352,886 (pos 18), mean **1,219,518** |
| wall clock at 1.789773 MHz | **0.681 s/token** |
| independent emulator | ares 147 gives the **identical** 19 tokens |

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
