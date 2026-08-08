# NES/Famicom cycle-measurement instrument + transformer port — FINDINGS

Journal. Appended after every discrete result. Committed alongside.
Rebuild after a scratchpad wipe; measured values from the prior run are
carried in as *expectations to reproduce*, never as results.

---

## 1. Instrument rebuilt and calibrated (2026-08-07)

**Method**: MAME 0.277 `nes` driver, headless (`-video none -sound none
-nothrottle -seconds_to_run N`), Lua `install_write_tap` on `$0300`,
timestamping `manager.machine.time` in attoseconds. No polling anywhere; the
tap handle is held in a global `KEEP` table. `nes` needs no BIOS ROM set.

**Marker protocol**: `MARKX v` = `ldx #v` (2) + `stx $0300` (4) = 6 cycles;
`MARKY v` = `ldy`/`sty`, also 6. Neither touches A, and `ld[xy] #imm` sets
only N/Z so the carry flag survives — that is what makes the branch tests
possible. A window is BEGIN..END, so `raw = payload + 6`; the host subtracts
6, and the empty payload therefore reads **0**.

**Clock, DERIVED not assumed.** Reference payload `lda #imm` (2) + `sta abs`
(4); window = 12 cycles including the closing marker.

| candidate | Hz | measured cycles | error |
|---|---|---|---|
| **1789772 (MAME truncation)** | 1789772.0000 | **11.999999999983** | -1.7e-11 |
| 21477272/12 (true NTSC) | 1789772.6667 | 12.000004469826 | +4.5e-06 |
| 1789773 (rounded) | 1789773.0000 | 12.000006704747 | +6.7e-06 |
| 1789800 (folklore) | 1789800.0000 | 12.000187733392 | +1.9e-04 |

Confirms the prior run: MAME instantiates the NTSC 2A03 at an integer
1789772 Hz, truncating 1789772.667. `cycles = round(delta_as*1789772/1e18)`.

**Resolution**: 1 CPU cycle = 558 730 385 770 as = 558.73 ns. Individual
cycles are resolved exactly — a single `nop` reads 2.

**Reproducibility**: 3 runs, marker streams compared **including absolute
timestamps** — BIT-IDENTICAL. MAME's NES boot is deterministic from reset.

**Datasheet calibration: 28/28, 0 mismatches.** Worst deviation of any window
from an integer cycle count: 2.6e-11. The set deliberately probes where a
plausible-but-wrong emulator diverges, and every trap came out right:

- indexed LOAD `lda abs,x|y` 4 -> **5** on a page cross;
- indexed STORE `sta abs,x|y` **5 whether or not the page is crossed** (the
  dummy read is unconditional) — this is the one a naive emulator gets wrong;
- `sta (zp),y` always 6; `lda (zp),y` 5 -> 6 on cross;
- RMW `inc abs` = **6** (dummy write modelled), `inc abs,x` = 7, `inc zp` = 5;
- branch = **three distinct values**: 2 not taken / 3 taken / 4 taken across a
  page;
- empty payload = **0**.

Branch placement is verified from the **raw ROM bytes**, not labels
(`tools/check_branches.py` decodes the opcode and signed relative operand out
of the .nes image and recomputes `page(PC_after) != page(target)`):
`t_br_taken` bcc @ $8406, PC_after $8408 -> $840B, no cross;
`t_br_cross` bcc @ $85FB, PC_after $85FD -> $8600, cross. Link-time
`.assert`s in the ROM enforce the same thing.

Files: `rom/common.inc`, `rom/calib.s`, `rom/nrom.cfg`,
`tools/nes_bench.py`, `tools/run_calib.py`, `tools/check_branches.py`,
report in `out/calib_report.txt`.

## 2. Primitives reproduced (2026-08-07)

All measured on real cartridge hardware models in MAME (MMC5 / MMC1 / MMC3
iNES images), each bank switch verified by reading a per-bank signature byte
back out of the switched window so a no-op switch cannot pass as a success.
3 runs each, bit-identical.

**19/19 MMC5-ROM primitives match the prior run exactly.**

| primitive | measured | prior | note |
|---|---|---|---|
| MMC1 PRG bank switch | **30** | 30 | `lda#`(2) + 5x`sta $E000`(20) + 4x`lsr a`(8); 5-bit serial port |
| MMC3 full switch | **12** | 12 | `lda#`/`sta $8000` + `lda#`/`sta $8001` |
| MMC3 hot switch | **6** | 6 | register already selected |
| MMC5 bank switch | **6** | 6 | `lda #bank` + `sta $5114` |
| MMC5, value already in A | **4** | 4 | just `sta $5114` |
| `$6000` PRG-RAM `lda`/`sta` abs | **4** | 4 | identical to system RAM absolute; **no cartridge penalty** |
| `$6000` `lda abs,y` aligned / cross | **4 / 5** | 4/5 | the `$6000` window has the same page-cross hazard |
| zero page `lda` | **3** | 3 | |
| PRG-ROM `lda tbl,y` aligned / cross | **4 / 5** | 4/5 | |
| 8-bit accumulate, acc resident in A | **4**/elem | 4 | `adc abs`; x16 = **64**, exactly linear |
| 8-bit accumulate spilled to zero page | **12** | 12 | `lda`/`clc`/`adc`/`sta` |
| 16-bit accumulate | **20** | 20 | |
| 32-bit accumulate | **36** | 36 | |
| ternary sign-separated gather | **8** | 8 | `ldy idx,x` + `adc act,y`; x16 = **128**, exactly linear |
| ternary branchy, zero trit (skip) | **7** | 7 | `lda w,x`(4) + `beq` taken(3) |
| ternary branchy, +1 trit (add) | **20** | 20 | |
| ternary branchy, -1 trit (sub) | **21** | 21 | |

Bank signatures observed: MMC5 $B0 -> $B1 -> $B2; MMC1 $C0 -> $C2;
MMC3 $D0 -> $D1 -> $D3. Every switch really happened.

**Mapper choice, from the numbers**: MMC1 is **5.0x** the cost of MMC5 per
switch and MMC3 2.0x (1.0x hot). MMC5 confirmed.

**Ternary inner loop, from the numbers**: sign separation keeps the
accumulator in A and pays **8** per nonzero and **0** per zero. The branchy
variant pays 7 per zero and 20/21 per nonzero because testing the trit forces
the accumulator out of A. At a realistic 50% zeros the branchy loop costs
0.5*7 + 0.5*20.5 = **13.75 cycles per weight** against 0.5*8 = **4 cycles per
weight** for the sign-separated gather (before index-advance overhead) - a
**3.4x** structural gap. Confirms the design decision.

### PRG-RAM: 32 KB in 4 banks, confirmed, with a detail

Probe: write `$A0+b` to `$6000` with `$5113 = b` for b=0..7, then read every
selector back. Repeated with the iNES PRG-RAM size byte declared as
0/1/2/4/8/16 units of 8 KB:

| declared | read back at $5113 = 0..7 | distinct |
|---|---|---|
| 0 KB / 8 KB | A3 A3 A3 A3 A7 A7 A7 A7 | 2 |
| 16 KB | A3 A3 A3 A3 A6 A7 A6 A7 | 3 |
| **32 KB** | A3 A3 A3 A3 **A4 A5 A6 A7** | 5 |
| **64 KB** | A3 A3 A3 A3 **A4 A5 A6 A7** | 5 |
| **128 KB** | A3 A3 A3 A3 **A4 A5 A6 A7** | 5 |

Reproduces the prior run: **declaring more than 32 KB still yields 4 banks.**
The extra detail this time is *where* they are - `$5113 = 4,5,6,7` are the
four distinct 8 KB banks (32 KB), while `$5113 = 0..3` all alias onto one
further 8 KB region (MMC5's separate first PRG-RAM chip). The port therefore
uses **$5113 = 4..7** for its 32 KB banked window.

Files: `rom/prim.s`, `rom/mmc1.s`, `rom/mmc3.s`, `rom/mmc5.cfg`,
`rom/mmc1.cfg`, `rom/mmc3.cfg`, `tools/run_prim.py`, `out/prim_report.txt`.

## 3. Host reference built - and it caught a generator bug immediately

`host/ref.py` is the exact-integer specification (no floating point anywhere in
the forward pass) and also the packer.

Model shape: V=64, D=64, L=3, H=2, d_head=32, F=128, T=20 ->
**102,400 ternary weights**, 1,408 rows, **51,299 nonzeros**, 57,344 bytes of
weight stream (7 banks), 5,656 bytes of row headers.

**Negative worth recording:** the first version used a textbook LCG. With a
power-of-two modulus an LCG's low bits are periodic, so `s % 4` cycled
0,1,2,3 and *every* 64-input row came out with exactly 32 nonzeros. The tell
was the packer reporting **zero bank crossings** - 51,200 stream bytes made of
rows that are all exactly 32 or 64 bytes tile 8,192 with no remainder. Had
that gone unnoticed the exactness test would have run on perfectly structured
weights and been far weaker than it looked. Replaced with xorshift32; nnz is
now 51,299 and irregular.

**BANK CROSSINGS PER TOKEN = 6** (measured by the packer, not estimated:
6 rows out of 1,408 would have straddled an 8 KB boundary and were pushed to
the next bank behind a sentinel). At the measured MMC5 cost of 6 cycles that
is **36 cycles per token**. Against a predicted ~900,000 cycles/token that is
0.004%. The reason is structural: the weight stream is one forward sweep per
token, never a random walk.

Generated token ids (start token 1, 19 steps, greedy argmax):
`1, 60, 0, 6, 27, 32, 5, 60, 41, 34, 34, 34, 34, 34, 34, 32, 5, 8, 27, 6`
- 10 distinct values, so the ROM-vs-host comparison is not a constant.

## 4. The port: ROM matches the host reference EXACTLY over 19 tokens

`rom/nn.s`, MMC5, 80 KB PRG (ten 8 KB banks), built by `./build.sh`.

### 4.1 Two real bugs, both of which a 3-token check would have passed

**Bug 1 - Duff's device read the wrong end of the list.** The gather chain is
16 entries of `ldy stream+k,x / adc actb,y` with *constant* offsets `k`.
Entering such a chain at entry `16-r` to handle a partial block of `r` covers
offsets `x+16-r .. x+15` - the **last** r bytes of the block, not the first r.
Fixed by carrying a `-BLOCKSZ` shift in the pointer and pre-advancing X, so
entry `k` reads offset `k - BLOCKSZ` and entering at `16-r` with the pointer
one past the block covers exactly the first r.

The important part: with this bug the ROM still produced the **correct token
at positions 0 and 1**, and only diverged at position 2. Dumping intermediate
state showed layer 0 was already wrong at position 0 - the tokens matched by
coincidence. A 3-token check would have reported 2/3 or even 3/3 and passed a
completely broken kernel. This is the second time in this project that has
happened.

**Bug 2 - a row landing exactly on a bank boundary.** The packer decided where
banks start with `len(stream) % BANK`. A row that ends exactly on a boundary
makes that read 0, the per-bank pad is never emitted, the ROM's pointer
underflows to chain -1 and it executes whatever follows the chain table. The
symptom was a **hang**, not wrong output. Fixed by tracking the bank offset
explicitly. (Reported crossings went 6 -> 5 -> 6; the 5 was the tell.)

### 4.2 ROM vs HOST, 19 generated tokens, side by side

| pos | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |10 |11 |12 |13 |14 |15 |16 |17 |18 |
|-----|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ROM |60 | 0 | 6 |27 |32 | 5 |60 |41 |34 |34 |34 |34 |34 |34 |32 | 5 | 8 |27 | 6 |
| host|60 | 0 | 6 |27 |32 | 5 |60 |41 |34 |34 |34 |34 |34 |34 |32 | 5 | 8 |27 | 6 |

**19/19 EXACT.** 10 distinct token values, so this is not a constant match.
Intermediate state was additionally checked element-by-element at position 0:
`x0`, `x` after each of the three layers, `q`, and the attention output all
match the host reference exactly.

3 runs of the full ROM: marker streams **bit-identical including absolute
timestamps**, token dumps identical.

### 4.3 Kernel: measured cycles per MAC

Micro-benchmark (`-DBENCH`) drives the **real** `gather` routine over synthetic
lists and measures with the same instrument:

| list length n | cycles | cycles/MAC |
|---|---|---|
| 1 | 100 | 100.00 |
| 8 | 156 | 19.50 |
| 16 | 220 | 13.75 |
| 17 | 271 | 15.94 |
| 32 | 391 | 12.22 |
| 64 | 733 | 11.45 |
| 96 | 1075 | 11.20 |
| 128 | 1417 | 11.07 |

Least squares over the full-block lengths: **cycles = 49.00 + 10.688 n**.

- **10.688 cycles/MAC asymptotic**, against the measured **8-cycle**
  sign-separated gather primitive. The 2.688 gap is entirely per-block
  bookkeeping: the 16-entry chain ends in `jmp (wfold)` and the fold does a
  16-bit accumulate, an X advance, a block counter and an indirect jump - 43
  cycles per 16 MACs.
- **49-cycle intercept** per list (table lookups, entry-point computation, the
  jsr/rts pair).
- In situ the model's lists average only 18.2 entries, which is 2 blocks, so
  the real figure is worse than asymptotic: **16.34 cycles per MAC** measured
  across a whole token (838,367 cycles / 51,299 nonzeros).

Two measured optimisations along the way, both re-verified at 19/19 exact:

| change | cycles/MAC in situ | mean cycles/token |
|---|---|---|
| first working version | 19.07 | 1,359,212 |
| inline the X advance, immediate block step | 17.28 | 1,267,489 |
| page-align the chain bases (see below) | **16.34** | **1,219,518** |

The last one is worth calling out. Carrying the `-BLOCKSZ` shift in the
instruction operand made every chain's base address `$xxF0`, so **essentially
every indexed load in the kernel paid the page-cross +1** measured back in the
calibration. Moving the shift into the pointer instead - and padding each bank
with 16 bytes so the pointer cannot go negative - made the bases page-aligned
and bought a straight **1 cycle per MAC**, 51,299 cycles per token. The
calibration suite predicted this exactly; without having measured the
page-cross penalty first it would have looked like free abstraction.

### 4.4 Where a token goes

Profiled build (`-DPROFILE`, X-preserving 12-cycle stage markers, overhead
counted and subtracted):

| stage | pos 0 | pos 18 (full context) |
|---|---|---|
| ternary gather (51,299 MACs) | 838,367 (74.5%) | 838,327 (60.4%) |
| attention (QK + AV) | 39,814 (3.5%) | 302,444 (21.8%) |
| everything else | 213,540 (19.0%) | 213,475 (15.4%) |
| bank switches (6) | 36 (0.003%) | 36 (0.003%) |

"Everything else" is header reads, requantisation, the biased activation
copies, residual adds, relu, embedding lookup and the argmax. Attention is
the only stage that grows with position, exactly as expected.

### 4.5 CYCLES PER TOKEN AND WALL CLOCK

| | cycles | seconds @ 1.789773 MHz |
|---|---|---|
| position 0 (context 1) | 1,090,397 | 0.6092 |
| position 18 (context 19) | 1,352,886 | 0.7559 |
| **mean over 19 tokens** | **1,219,518** | **0.6814** |
| all 19 tokens | 23,170,853 | 12.946 |

(The three candidate clocks agree to five decimals at this magnitude.)

**Against the prior extrapolation.** The prior run extrapolated ~0.8 s/token
for 114,688 active weights, i.e. ~12.2 cycles per weight. This measurement is
**8.19 cycles per weight** for the ternary kernel (838,367 / 102,400), and
**11.9 cycles per weight** including attention, requantisation and everything
else (1,219,518 / 102,400). So the extrapolation is **CONFIRMED as an
end-to-end figure** - scaled to 114,688 weights this port would run
1,366,000 cycles = 0.763 s/token, against the predicted 0.8 - and the
sign-separated kernel proper is about **1.5x better than the extrapolation**
because a zero weight costs literally nothing: it is absent from both index
lists. The gap between the two numbers is the non-kernel work, which the
extrapolation from a bare 32x16 layer did not include.

## 5. Independent emulator cross-check (ares 147)

The ROM parks its result in battery-backed PRG-RAM (`"ELYA"`, token count,
then the tokens at `$7FE0`), which gives a way to read a result out of an
emulator that has no scripting hook.

```
xvfb-run -a flatpak run dev.ares.ares --system Famicom elyanes.nes
```

`elyanes.ram` written by ares:

```
size 32768   magic at 8160   ntok 19
tokens [60, 0, 6, 27, 32, 5, 60, 41, 34, 34, 34, 34, 34, 34, 32, 5, 8, 27, 6]
```

**Identical to MAME and to the host reference, on a completely independent
emulator.** The result is not a MAME artifact.

Two things fall out of this for free:

- ares's battery file is **32,768 bytes**, independently confirming the
  measured 32 KB of MMC5 PRG-RAM from a second implementation.
- Because the state lives in battery-backed SRAM, the generated sequence
  survives power-off on real hardware at zero runtime cost - the write is 24
  bytes at the end of the run.

## 6. Block size: measured, and the prior finding does NOT carry over

The brief carried forward that a block of 32 never saturated in 1.148e9 blocks
despite a bound of 224, because ternary zeros and sign cancellation make the
sum grow like sqrt(n). **That argument does not survive sign separation** -
inside a sign-separated block every term comes from the same index list and
therefore has the same sign, so there is nothing left to cancel and the sum
grows like n, not sqrt(n).

Measured over the real 19-token trajectory (`host/blocksize.py`):

| block | blocks observed | max value seen | signed >127 | biased >255 |
|---|---|---|---|---|
| 16 | 84,455 | 191 | 0 | 0 |
| **32** | 56,981 | 344 | 0 | **5,477 (9.6%)** |

Worst case by construction: `block*14` biased, `block*7` signed.
Block 16 -> 224 / 112, both provably in range. Block 32 -> 448 / 224, both
provably out of range for the biased accumulator this kernel uses.

**Block 16 confirmed by measurement, and block 32 refuted for this
formulation** - it would corrupt 9.6% of blocks. The prior run's block-32
observation remains true for the *mixed-sign dense* formulation it was
measured on; it is simply not transferable here.

## 7. KV row alignment

The brief's guidance was to page-align KV rows because `$6000` has the same
page-cross +1 hazard (which this run re-measured: 4 aligned, 5 crossing).
Rows here are `D = 64` bytes at 64-byte aligned addresses, so the row base low
byte is one of 0/64/128/192 and the attention loop's `lda (kptr),y` with
`y <= 63` reaches at most `192 + 63 = 255`. **It provably never crosses a
page**, so 64-byte alignment gives the same guarantee as 256-byte alignment at
a quarter of the memory. This is a refinement of the guidance, not a
contradiction of it.

## 8. Summary and limitations

### What was reproduced or established

| | |
|---|---|
| instrument resolution | 1 CPU cycle = 558 730 385 770 as = 558.73 ns, individual cycles resolved |
| reproducibility | bit-identical over 3 runs **including absolute timestamps**, for every ROM |
| clock | **derived**: 1,789,772 Hz, error -1.7e-11 against a 12-cycle reference |
| datasheet calibration | **28/28, 0 mismatches** |
| branch placement | verified from **raw ROM bytes**, plus link-time asserts |
| primitives | **19/19 match the prior run**, plus MMC1 30 / MMC3 12+6 on real carts |
| PRG-RAM | 32 KB in 4 banks at `$5113 = 4..7`; declaring 64 or 128 KB still yields 4 |
| bank crossings per token | **6** = 36 cycles = 0.003% |
| kernel | **10.688 cycles/MAC** asymptotic, 16.34 in situ, 8.19 per weight |
| exactness | **19/19 tokens exact** vs the host reference; intermediates exact too |
| cycles/token | mean **1,219,518** = **0.681 s** at 1.789773 MHz |
| independent emulator | ares 147: identical tokens, 32,768-byte battery file |

### Negatives, plainly

- **The 8-cycle primitive is not achievable in a real kernel.** Measured
  asymptotic is 10.688 and the model's actual figure is 16.34, because its
  index lists average only 18.2 entries and each list pays a 49-cycle
  intercept plus a 43-cycle fold per 16-entry block. The primitive is a real
  lower bound, not a design target.
- **Block 32 is wrong for this formulation** and would corrupt 9.6% of blocks.
  The prior "block 32 never saturates" result does not transfer to a
  sign-separated kernel.
- **Attention costs ~23-25 cycles/MAC** and is not optimised. It hits the same
  structural wall as the discarded 4-bit-weight LUT: building the multiply
  table index needs `ORA`, there is no `ORA` to X or Y, so the accumulator
  cannot stay in A. At full context it is 21.8% of a token.
- **Two of my own bugs produced plausible output**: the Duff's-device entry bug
  gave the right token at positions 0 and 1 while layer 0 was already wrong at
  position 0, and the bank-boundary bug hung rather than mis-answered. Neither
  would have been caught by a short check.

### What I could not do

- **No second emulator with a usable timing hook.** ares has no scripting
  interface for Famicom, so its cross-check is functional only (via the
  battery file), not cycle-level. No third NES emulator with Lua (fceux,
  Mesen) is installed on this machine. All timing figures rest on MAME alone,
  mitigated by the 28/28 datasheet calibration and the branch-byte check.
- **The model is randomly initialised, not trained.** This port is verified for
  *exactness* and *cycle cost*, not for output quality. The prior run's
  training ablation (4-bit activations being load-bearing, 8-bit activations
  with an int8 accumulator being destroyed) was taken as a design input and
  was not re-measured - retraining was out of scope here.
- **The self-modifying RAM-resident kernel variant was not built.** The 32
  ROM-resident page chains reached 10.688 cycles/MAC without self-modifying
  code; an SMC variant was estimated at roughly the same cost and was not
  worth the risk, but that estimate is *not* a measurement.
- **The `.sav` cross-check does not prove ares's cycle timing**, only that the
  ROM computes the same answer under a different CPU implementation.

---

# Training a real model (2026-08-07)

The port was verified with randomly initialised weights: exactly correct,
completely meaningless. This section is the attempt to give it something to
say. Everything below is measured on this machine; predictions are labelled.

## The 64-symbol charset

`V = 64` is not a design knob - `rom/nn.s` declares `NVOCAB = 64` and the head
is a 64x64 ternary matrix - and 64 symbols is short of printable ASCII, so the
charset had to be chosen from the corpus rather than assumed.

Measured character histogram of TinyStories-valid (19,432,979 bytes, 21,990
stories), lowercased:

| band | share |
|---|---|
| `a-z` + space | 91.0% cumulative by `k` (the 27th symbol) |
| `.` `,` | 1.86% + 0.90% |
| `"` `'` | 0.48% + 0.22% |
| `!` `?` | 0.17% + 0.06% |
| digits `0-9`, all ten | **0.0035%** |

Digits are three parts in a hundred thousand. They were dropped, not encoded.
That is the whole justification: at V=64 a slot spent on `7` is a slot not
spent on a merge that fires every other token.

**The charset, 33 symbols:**

```
abcdefghijklmnopqrstuvwxyz  space  .  ,  "  '  !  ?
```

Everything else is folded onto a symbol that preserves the *shape* of the
text (newline/tab/dash/slash -> space, colon/semicolon -> comma, curly quotes
-> straight quotes, ellipsis -> full stop, accented letters -> unaccented) or
dropped (digits, `&#$()` and one emoji).

**Coverage after mapping, over all 19,089,665 characters:**

| | count | share |
|---|---|---|
| represented exactly | 18,986,555 | **99.4599%** |
| folded onto a related symbol | 102,354 | 0.5362% |
| dropped | 756 | **0.0040%** |
| **coverage (exact + folded)** | | **99.9960%** |

That leaves **31 of the 64 slots free**, which is 31 dead rows in the output
head - 1,984 ternary weights, 1.9% of the model, doing nothing. So a second
vocabulary was built on the same mapped text: the 33 base symbols plus **31
greedy BPE merges**, filling all 64 slots.

The 31 merges learned, in order:

```
'e ' ' t' 'd ' 'he ' ' a' ' s' 'he' ' the ' 't ' 'in' 'wa' 'nd ' '. ' 'y '
' to' 's ' 'ou' 'on' 'ha' 'er' 'ed ' 'it' ' and ' 'ed' 'her' ', ' 'en' 'ar'
'ing' ' to ' 'om'
```

They are worth **1.454 characters per token**, so at the fixed `T = 20` the
model sees ~29 characters of context instead of 20. That matters more than it
sounds: 20 characters is under four words.

**Split**: 1,500 whole stories held out, 20,490 kept, chosen by story id, never
by line. The sibling N64 run was bitten by a "validation" split that was a
reshuffle of the fit lines, which makes val identical to fit by construction.

| vocabulary | fit tokens | val tokens |
|---|---|---|
| charset (33 used of 64) | 17,798,157 | 1,290,667 |
| bpe64 (64 used of 64) | 12,243,030 | 887,743 |

## Random-init baseline, re-measured this session

Before touching anything, the committed ROM was re-run to confirm the
instrument still reproduces its own transcript:

```
TOKENS MATCHING: 19/19  -> EXACT
TOTAL     23199514 cycles    12.9623 s   (mean 1221027 cycles/token)
```

That is bit-identical to `out/FULL_VERIFICATION.txt`.

**A discrepancy inside this repo, worth recording.** FINDINGS section 4.5
above quotes the mean as **1,219,518** cycles (0.6814 s) and position 0 as
1,090,397. The current ROM measures **1,221,027** and 1,091,722. The 4.5 table
was written at commit `0f5df22`, *before* `8722a49` fixed the packer's
exact-bank-boundary pad; that fix changed the stream by a few bytes and cost
1,509 cycles per token. The clean-rebuild transcript is the truth and 4.5 is
stale by that amount. Everything below uses **1,221,027 cycles/token at
nnz = 51,299 (density 0.5010)** as the random-init baseline.

## Block 32 is refuted harder on trained weights than on random ones

The block-size argument depends on the *activation* range, not the weights,
but the number of blocks and therefore the overflow count depends on the
sparsity training chooses. Re-measured over the real 19-token trajectory with
the trained tau=0.50 weights (density 0.674):

| block | blocks | max value | signed >127 | biased >255 |
|---|---|---|---|---|
| 16 | 110,409 | **212** | 0 | **0** |
| 32 | 60,838 | 382 | 269 | **7,308 (12.0%)** |

Against the random-init figure of 5,477 of 56,981 (9.6%). Denser rows mean
longer index lists, longer lists mean more full 32-blocks rather than short
tails, and every term in a sign-separated block has the same sign, so there is
nothing to cancel. Block 16's worst observed value is 212 against the
provable bound of 224 - close, and it is *provably* closed, which is the point.

## Training table (12,000 steps each, identical shape, 102,400 weights)

All arms are the same 102,400-weight model. Loss is cross entropy in nats.
**Nats per token are not comparable across the two vocabularies** - a bpe64
token is worth 1.454 characters and a charset token exactly one - so the
column that decides anything is nats per **character**.

| arm | vocab | quant | tau | fit | val | val/char | density | nnz | banks | fits 7? |
|---|---|---|---|---|---|---|---|---|---|---|
| bpe64_twn_tau1.00_q1 | bpe64 | float W | 1.00 | 2.2044 | 2.2075 | **1.5185** | 0.4256 | 43,583 | 6 | yes |
| bpe64_twn_tau0.75_q2 | bpe64 | QAT | 0.75 | 2.2401 | 2.2435 | **1.5432** | 0.5441 | 55,711 | 7 | yes |
| bpe64_bn_tau1.00_q2 | bpe64 | QAT | 1.00 | 2.2566 | 2.2586 | **1.5537** | 0.6738 | 69,002 | 9 | **NO** |
| bpe64_twn_tau0.50_q2 | bpe64 | QAT | 0.50 | 2.2566 | 2.2586 | **1.5537** | 0.6738 | 69,002 | 9 | **NO** |
| bpe64_twn_tau1.00_q2 | bpe64 | QAT | 1.00 | 2.2824 | 2.2874 | **1.5734** | 0.4137 | 42,360 | 6 | yes |
| bpe64_twn_tau1.00_q2 (seed 2) | bpe64 | QAT | 1.00 | 2.2984 | 2.3008 | **1.5827** | 0.4148 | 42,475 | 6 | yes |
| bpe64_twn_tau1.25_q2 | bpe64 | QAT | 1.25 | 2.3823 | 2.3855 | **1.6410** | 0.3080 | 31,538 | 4 | yes |
| charset_twn_tau1.00_q2 | charset | QAT | 1.00 | 1.7041 | 1.6999 | **1.6999** | 0.4224 | 43,257 | 6 | yes |
| bpe64_twn_tau1.50_q2 | bpe64 | QAT | 1.50 | 2.5246 | 2.5290 | **1.7396** | 0.2171 | 22,230 | 3 | yes |
| bpe64_twn_tau1.00_q0 | bpe64 | fp32 | 1.00 | 2.6446 | 2.6472 | **1.8209** | 0.4249 | 43,506 | 6 | yes |

uniform is ln(64) = 4.1589 nats/token, i.e. 2.861 nats/char on bpe64.

Five things fall out of that table, three of them surprises.

**1. BitNet b1.58 and TWN are the same arm.** `bn` at tau=1.00 and `twn` at
tau=0.50 produced byte-identical results - same loss to four decimals, same
69,002 nonzeros. That is not a coincidence and not a bug: BitNet's absmean
rule keeps a weight when `|w| >= 0.5*tau*mean|w|` and TWN keeps it when
`|w| > tau*mean|w|`, so `bn(tau) == twn(tau/2)` exactly. "BitNet vs TWN" is
not a real experiment for a scale-free ternary kernel; it is the same map with
the threshold reparametrised. Reported rather than quietly dropped.

**2. tau has an interior optimum and it sits almost exactly on the ROM's
capacity.** Density falls monotonically with tau (0.674 -> 0.217) but loss is
U-shaped with a minimum at tau = 0.75, density 0.544. The 7-bank weight-stream
window holds at most 57,232 index bytes = density 0.5589. The best arm needs
55,711. It fits with 1,521 bytes to spare, which is luck, not design.

**3. Ternarising the weights costs almost nothing here.** Float weights with
everything else identical (`q1`) scored 1.5185 nats/char against QAT's 1.5734
at the same tau - **0.055 nats/char, about 3.6%**. The weights are not the
bottleneck; the 4-bit activations and the 20-token context are.

**4. The fp32 control is WORSE than the quantised model** (1.8209 vs 1.5734).
This model has no LayerNorm and no RMSNorm anywhere - the only thing holding
the residual stream in range is the `clamp(-7, 7)` after each residual add.
Remove it and the fixed requantise shifts become arbitrary rescalings of an
unbounded stream. The 4-bit clamp is **load-bearing as the model's only
normaliser**. (Fairness check: the control was rerun at two more learning
rates before this claim was allowed to stand - see below.)

**5. bpe64 beats plain characters by 0.157 nats/char**, 1.5432 vs 1.6999, i.e.
9.2%. Spending the 31 otherwise-dead vocabulary slots on merges is worth it,
and the reason is almost certainly the context: 20 bpe64 tokens is ~29
characters, 20 charset tokens is 20.

Seed noise: the same arm at seed 2 scored 1.5827 against seed 1's 1.5734, so
**~0.009 nats/char**. The tau 0.75-vs-1.00 gap of 0.030 is about three times
that, and the vocabulary gap of 0.157 is seventeen times it.

## The attention path was carrying ONE BIT per element

Training turned up a defect in the port's own constants that the exactness
work could never have found, because the ROM and the reference agreed on it
perfectly.

`AV_SHIFT` was 4. The attention output is

```
att[j] = quant( sum_t floor(p_t * v_t / 4) , AV_SHIFT )
```

and the quantised softmax normalises so that `sum_t p_t <= 8` (the
power-of-two step picks the smallest `kk` with `S >> kk <= 8`, and a sum of
floors is at most the floor of the sum). Every value nibble is in `-7..7`.
So the accumulator is **provably bounded by `7 * 8 / 4 = 14`**, and shifting a
quantity that never exceeds 14 by 4 bits leaves `floor(a/16)` in `{-1, 0}`.

Measured over a real 19-token trajectory, all three layers:

```
raw AV accumulator over 3648 values:  min -13  max 12  mean -0.30  std 7.41

  AV_SHIFT=0 -> 15 output levels, saturating 29.28%
  AV_SHIFT=1 -> 14 output levels, saturating  0.00%   <-- correct
  AV_SHIFT=2 ->  8 output levels, saturating  0.00%
  AV_SHIFT=3 ->  4 output levels, saturating  0.00%
  AV_SHIFT=4 ->  2 output levels, saturating  0.00%   <-- what the port shipped
```

Layer-0 attention output histogram, at AV_SHIFT=4, over 1,216 values:

| value | -7..-2 | -1 | 0 | 1..7 |
|---|---|---|---|---|
| trained | 0.0% | 49.1% | 50.9% | 0.0% |
| random init | 0.0% | 53.4% | 46.6% | 0.0% |

Identical for trained and random weights, so this is architecture, not
training. Three attention heads' worth of machinery, 7,680 multiply-adds per
token, resolving to one bit per channel.

The other two shifts are fine: `K_SHIFT = 2` saturates 27.24% of the time
(14.32% high, 12.92% low - a healthy two-sided rate for a 4-bit activation),
`W2_SHIFT = 3` saturates 8.39%.

**Fix**: `AV_SHIFT` 4 -> 1. To stop the ROM and the specification ever
disagreeing about a constant like this again, `host/ref.py` now *generates*
`out/model/shifts.inc` and `rom/nn.s` includes it, so the shift exists in
exactly one place. Regression-checked: rebuilding at `AV_SHIFT=4` produces a
ROM **byte-identical** to the committed one, and the trainer/reference
equivalence test still reports EXACT at both 1 and 4.

## A near miss worth recording: a model is not just its weights

After changing the default `AV_SHIFT` from 4 to 1, the previously trained
60,000-step model was sampled with the new default and produced this:

```
seed  1 'b'      -> 'btle a"s. ymisnonwae ye r'
seed 26 ' '      -> ' yywymeaaay.l sl'. 'n"'
```

Nothing raised. The npz loaded, the shapes matched, every weight was a legal
ternary value, the forward pass ran, tokens came out. The model had simply
been *trained* at `AV_SHIFT = 4` and was being *run* at 1. Sampled at the
shift it was trained for, the identical file gives:

```
seed  1 'b'      -> 'big because a bird and s'
seed 26 ' '      -> ' brank for his friends'
```

This is the same failure mode as the sibling N64 exporter - a silent
disagreement between two halves of a pipeline that each look fine on their
own. Fixed the same way: the four requantise shifts are now written **into
the npz** as `_shifts`, and `ref.Model.from_npz` refuses to load a file whose
stamp disagrees with the reference it is being loaded into. The 11 npz files
that predate the stamp were backfilled with `[2, 3, 4, 3]`.

## Fixing the attention shift made it WORSE - the ladder

The bound above says `AV_SHIFT = 1` is the shift that uses the full 4-bit
range. Trained at the same tau, same budget, same seed, it is **worse**:

| AV_SHIFT | output levels reachable | val (nats/token) | val/char |
|---|---|---|---|
| 1 | 14 of 15 | 2.3020 | 1.5832 |
| **2** | **8 of 15** | **2.2217** | **1.5280** |
| 3 | 4 of 15 | 2.2358 | 1.5377 |
| 4 (as shipped) | 2 of 15 | 2.2435 | 1.5432 |
| 5 | 1-2 of 15 | 2.2823 | 1.5698 |

(bpe64, tau = 0.75, 12,000 steps, seed 1. Seed noise on this axis is ~0.009
nats/char, so 1-vs-2 is real and 3-vs-4 is not.)

The ladder is U-shaped with the optimum at **2**, and the "correct" shift by
the range argument is the **worst** of the five. Widening the attention output
does not help; it hurts.

The explanation is the same one as the fp32 control. This model has no
normalisation layer at all. The attention output goes through `Wo` and into
`x = clamp(x + o, -7, 7)`, and the residual stream is only 15 levels wide. A
wide attention update spends most of its magnitude being clipped off by that
clamp, and clipping is where gradient goes to die. A narrow one behaves as a
small gated correction the residual can actually carry. At `AV_SHIFT = 1` the
attention path has the most information and the least ability to deliver it.

So the "attention carries one bit" finding stands as a **description** of the
architecture and the obvious fix is **refuted by measurement**. AV_SHIFT is
moved from 4 to 2, worth 0.015 nats/char - real, but a twentieth of what the
vocabulary choice was worth.

Also verified: the trained cartridge reproduces the host reference exactly
from three independent starting tokens, not one.

```
seed token  1 : max|dW| = 0,  TOKENS MATCHING: 19/19 -> EXACT
seed token 26 : max|dW| = 0,  TOKENS MATCHING: 19/19 -> EXACT
seed token 40 : max|dW| = 0,  TOKENS MATCHING: 19/19 -> EXACT
```

57 generated tokens, every one bit-identical between the 6502 and the
exact-integer specification.

## THE FINAL CARTRIDGE

`runs/final_av2_bpe64_tau0.75.npz` - bpe64, TWN tau = 0.75, `AV_SHIFT = 2`,
60,000 steps at batch 192 (about 230 million training tokens, ~19 epochs of
the 12.24 M-token fit split), 16 minutes on an RTX 4070 Laptop.

| | |
|---|---|
| fit / val | 2.0431 / **2.0546** nats/token = **1.4133 nats/char** |
| uniform baseline | 4.1589 nats/token = 2.861 nats/char |
| weights | 102,400 ternary |
| **nonzero weights** | **52,207 (density 0.5098)** |
| stream image | 57,344 bytes, 7 banks, 6 bank crossings/token |
| ROM | 90,128 bytes (80 KB PRG + header + CHR) |

### max\|dW\| = 0

The npz the trainer wrote, decoded back out of `stream.bin` + `headers.bin`
using only the documented stream format (not the packer's own code path):

```
matrix       shape        nnz        max|dW|
L0_Wq        64x64        2108       0
L0_Wk        64x64        2117       0
L0_Wv        64x64        2114       0
...          (all 19 matrices)       0
L2_W2        64x128       3912       0
head         64x64        2306       0

embed.bin  max|d| = 0   over 4096 values
pos.bin    max|d| = 0   over 1280 values
weights    102400   nonzero 52207   density 0.5098

max|dW| = 0   over 102400 ternary weights -> EXACT
```

(Full transcript: `out/FINAL_MAXDW.txt`. The two per-matrix nnz figures first
written into this table were recalled rather than read, and were wrong by ~4%
and ~12%; they are now copied from the transcript. Recording that because it
is precisely the failure this repo keeps guarding against.)

The check also validates the derived `d7 = -7*(n_pos - n_neg)` header constant
on every one of the 1,408 rows, that no row straddles a bank, and that each
row's `+1` and `-1` index lists are disjoint.

### ROM == host, 57 tokens over three seeds

MAME 0.277 `nes` driver, cycle-exact write tap. Full transcripts in
`out/FINAL_VERIFICATION.txt` and `out/FINAL_SIDEBYSIDE.txt`.

```
seed token 1 = 'b'
  pos  rom   rom sym  host  host sym
  0    8     'i'      8     'i'      ok
  1    6     'g'      6     'g'      ok
  2    26    ' '      26    ' '      ok
  3    5     'f'      5     'f'      ok
  4    17    'r'      17    'r'      ok
  5    8     'i'      8     'i'      ok
  6    59    'en'     59    'en'     ok
  7    3     'd'      3     'd'      ok
  8    18    's'      18    's'      ok
  9    27    '.'      27    '.'      ok
  10   38    ' s'     38    ' s'     ok
  11   36    'he '    36    'he '    ok
  12   43    'wa'     43    'wa'     ok
  13   18    's'      18    's'      ok
  14   38    ' s'     38    ' s'     ok
  15   14    'o'      14    'o'      ok
  16   26    ' '      26    ' '      ok
  17   51    'ha'     51    'ha'     ok
  18   15    'p'      15    'p'      ok
  ROM  text: 'big friends. she was so hap'
  HOST text: 'big friends. she was so hap'
  identical: True   over 19 tokens
```

| seed token | tokens | result | ROM text |
|---|---|---|---|
| 1 `'b'` | 19/19 | EXACT | `big friends. she was so hap` |
| 26 `' '` | 19/19 | EXACT | `' came to playing with hi'` |
| 40 `' the '` | 19/19 | EXACT | `' the stree to friends. he was s'` |

**57 generated tokens, every one bit-identical.** 19 per seed clears the 16+
bar; a 3-token check would have passed three genuinely broken changes in this
project, including this ROM's own Duff's-device bug, which produced the right
token at positions 0 and 1 while layer 0 was already wrong at position 0.

### WHAT IT ACTUALLY SAYS

Greedy argmax with ties to the lowest index, which is exactly what `rom/nn.s`
does, so these are the strings the cartridge produces. The ROM free-runs from
a single seed token - that token is the entire prompt. Full 64-seed survey in
`out/FINAL_SURVEY.txt`; a fair sample:

```
seed  1 'b'      -> 'big friends. she was so hap'
seed  3 'd'      -> 'day, but the strange. he was a'
seed  4 'e'      -> 'elly. she was so excited. one '
seed  7 'h'      -> 'him came and said, "i hav'
seed  9 'j'      -> 'just. he said, "i have to '
seed 12 'm'      -> 'mall with her mommy. she sa'
seed 14 'o'      -> 'or notice upon a time, ther'
seed 20 'u'      -> 'up with him friends. he wa'
seed 22 'w'      -> 'with her mom friends. he sai'
seed 26 ' '      -> ' came to playing with hi'
seed 28 ','      -> ', she said, "that's a proud '
seed 30 "'"      -> ''t mommy said, "i didn't '
seed 31 '!'      -> '! he was so happy. she was so '
seed 34 ' t'     -> ' they were saw from the sky. she '
seed 36 'he '    -> 'he said, "i was so happy. s'
seed 42 'in'     -> 'in the streess. she was so exc'
seed 43 'wa'     -> 'was so happy and said, "i hav'
seed 51 'ha'     -> 'hat it was so excited. she was s'
seed 57 'her'    -> 'her friends. she was so happy'
seed 58 ', '     -> ', he saw a big dad and said'
seed 59 'en'     -> 'enture. he was so happy and sai'
```

And with a multi-token prompt (**host only** - the ROM has no prompt path):

```
'once upon a time, ' -> 'once upon a time, but she sai'
'the little girl '   -> 'the little girl friends. on'
'he said, '          -> 'he said, "i was so happy. s'
'lily was very '     -> 'lily was very playing with her'
```

**Assessment, plainly.** This is coherent short English at the phrase level and
not at the sentence level. Spelling is essentially correct - `friends`,
`excited`, `mommy`, `happy`, `playing`, `proud` - punctuation and quoting are
used correctly, `he`/`she` agreement holds within a clause, and several
outputs are grammatical all the way through: `' he was so happy. she was so '`,
`', he saw a big dad and said'`, `'hat it was so excited. she was s'`. There
are also clear failures: `streess`, `stoppped`, `riess` are non-words,
`'or notice upon a time'` is a mangled `once upon a time`, and the model leans
heavily on a handful of TinyStories cliches (`he said, "i have to`, `was so
happy`, `friends`). It does not plan past about a clause, which is exactly
what a 20-token context and three 64-wide layers should be expected to do.

It is not a language model in any useful sense. It is 102,400 ternary weights
producing recognisable, mostly-correctly-spelled English on a 1.79 MHz 6502,
and that is the claim being made.

### CYCLES PER TOKEN, TRAINED vs RANDOM

| | random init | trained (final) | change |
|---|---|---|---|
| nonzero weights | 51,299 | **52,207** | +908 (+1.8%) |
| density | 0.5010 | 0.5098 | +0.0088 |
| position 0 | 1,091,722 | 1,103,615 | |
| position 18 | 1,354,566 | 1,366,939 | |
| **mean over 19 tokens** | **1,221,027** | **1,233,099** | **+12,072 (+0.99%)** |
| seconds per token @ 1789772 Hz | 0.6822 | **0.6889** | |
| ternary kernel | 16.34 cycles/MAC | **16.30 cycles/MAC** | |
| ternary kernel | 8.19 cycles/weight | **8.31 cycles/weight** | |

A zero weight costs literally nothing in this kernel - it is absent from both
index lists - so cost tracks **nnz**, not weight count, and training moved nnz
by 1.8%. The marginal cost works out at `12,072 / 908 = 13.3` cycles per extra
nonzero, against the 16.30 cycles/MAC average, which is what you would expect:
the average includes per-row fold and setup amortised over the row, and an
extra weight inside an existing block pays only the gather.

The tau knob is a direct speed/quality dial and it was measured, not guessed:

| tau | nnz | mean cycles/token (extrapolated at 13.3/nnz) | val/char |
|---|---|---|---|
| 1.50 | 22,230 | ~832,000 | 1.7396 |
| 1.25 | 31,538 | ~956,000 | 1.6410 |
| 1.00 | 42,360 | ~1,100,000 | 1.5734 |
| **0.75** | **55,711** | **1,233,099 (measured)** | **1.5432** |
| 0.50 | 69,002 | ~1,410,000 - does not fit the ROM | 1.5537 |

(The 12,000-step column; the shipped model is the 60,000-step version of the
tau = 0.75 row.) Those cycle figures except the measured one are the honest
label: extrapolations from one measured marginal rate, not measurements.

### Block sizes on the shipped weights

| block | blocks | max value | signed >127 | biased >255 |
|---|---|---|---|---|
| 16 | 86,507 | **199** | 0 | **0** |
| 32 | 56,715 | 383 | 26 | **5,458 (9.6%)** |

Block 16 remains provably and observably safe. Block 32 remains refuted.

### Attention range, before and after

| | AV_SHIFT=4 (was) | AV_SHIFT=2 (shipped) |
|---|---|---|
| layer-0 attention levels used | 2 (`-1`, `0`) | **8** (`-4..3`) |
| distribution | 49.1% / 50.9% | 3.3 / 2.8 / 11.6 / 19.9 / 24.6 / 20.1 / 6.9 / 10.8 % |
| `K_SHIFT=2` saturation | 27.24% | 31.67% |
| `W2_SHIFT=3` saturation | 8.39% | 13.35% |

### What could not be done

* **Generation is capped at 19 tokens (~28 characters).** `T = 20` is set by
  the KV cache fitting one 8 KB PRG-RAM bank, and the positional table has
  exactly 20 rows. There is no sliding window, on the ROM or in the host
  reference, so nothing here shows the model over a paragraph.
* **The ROM cannot be prompted.** It free-runs from one seed token. The
  multi-token prompts above are host-only and are labelled as such everywhere
  they appear. Adding a prompt path is a small ROM change that was not made.
* **Greedy argmax only.** No temperature, no sampling. That is what the
  cartridge does, so it is what is reported; sampled text would read better
  and would not be what the hardware produces.
* **No search over `K_SHIFT` or `W2_SHIFT`.** Only `AV_SHIFT` was laddered.
  `K_SHIFT = 2` saturates 31.67% of the time, which is high enough to be worth
  a ladder of its own, and that was not run.
* **The float-weight control does not have a stable sign.** At `AV_SHIFT = 4`
  float weights beat ternary by 0.055 nats/char; at `AV_SHIFT = 1` they *lost*
  by 0.061. So "ternarisation costs about 3.5%" is only safe to say at the
  shift it was measured at, and weight precision is not the dominant term at
  this scale either way.
* **One corpus.** TinyStories only. No check that the charset choice
  generalises to a different register of English.

## Note on reproducing the older transcripts

`./build.sh` now emits a random-init ROM at the new default `AV_SHIFT = 2`, so
it no longer reproduces `out/FULL_VERIFICATION.txt`, which was taken at
`AV_SHIFT = 4`. To reproduce that transcript byte for byte:

```sh
NES_AV_SHIFT=4 ./build.sh
python3 tools/run_nn.py out/nn.nes out/model/expected.json 60
```

Checked: at `NES_AV_SHIFT=4` the rebuilt `nn.nes` is **byte-identical** to the
ROM committed before this session, so the shifts.inc refactor, the `SEEDTOK`
define and the `NSTREAM` define are all provably neutral.

The committed `out/nn.nes` is the **trained** cartridge
(`runs/final_av2_bpe64_tau0.75.npz`, seed token 1), not the random-init one.

---

# Context extension: T = 20 -> 85 (2026-08-08)

The shipped cartridge writes phrase-level English with correct spelling and
punctuation and no sentence-level coherence. It also has **29 characters of
context** (T = 20 tokens at 1.454 chars/token). Those two facts are not
separable from the outside: the model may be short of weights, or it may
simply never see a whole sentence. One retrain at a longer context separates
them.

`T` is the only thing that changes. V = 64, D = 64, L = 3, H = 2, d_head = 32,
F = 128, block 16, 4-bit activations, ternary sign-separated weights,
`AV_SHIFT = 2`, the same TinyStories corpus and the same 64-symbol bpe64
vocabulary (`data/vocab.json` in this clone is byte-identical to the one the
T = 20 model was trained on - checked with `cmp`).

The KV arithmetic that sets the target:

```
KV bytes = L x T x 2 x D
T = 20 -> 3*20*2*64 =  7,680 B = 23.4% of the 32,768 B PRG-RAM window
T = 85 -> 3*85*2*64 = 32,640 B = 99.6% of it, ~124 characters of context
```

85 is not a round number, it is `32768 / (3*2*64)` floored - the largest
context this cartridge can hold without touching system RAM.

## Baseline re-measured before anything was touched

The committed cartridge, rebuilt from `runs/final_av2_bpe64_tau0.75.npz` in
this clone, against the host reference:

```
max|dW| = 0   over 102400 ternary weights -> EXACT
TOKENS MATCHING: 19/19  -> EXACT
TOTAL     23428899 cycles    13.0904 s   (mean 1233099 cycles/token)
```

Bit-identical to `out/FINAL_VERIFICATION.txt`. The instrument, the toolchain
and the corpus in this clone all reproduce the shipped result, so anything
that changes from here is the change under test and not the environment.

## What had to change in the ROM, and what it cost

`T` is now a build parameter (`-DNCTX`, matching `NES_T` for `host/ref.py`).
Four things in the port were sized for 20 and had to be re-derived.

**1. The KV cache no longer fits one PRG-RAM bank.** At `T = 85` it is 32,640
bytes across all four. Rows are still `D = 64` bytes and 64-byte aligned, and
`8192 / 64 = 128` rows fill a bank exactly, so a row still never straddles a
bank and `(kptr),y` with `y < 64` still cannot cross a page - the alignment
argument from the one-bank version carries over unchanged. `kv_ptr` now builds
a 16-bit row index and writes `$5113`:

```
row  = (curlay*2 + kvsel) * NCTX + kvt
bank = row >> 7        ->  $5113 = 4 + bank
kptr = $6000 + (row & 127) * 64
```

The layout was changed from `[layer][t][k|v]` to `[layer][k|v][t]` so that the
score loop and the AV loop walk **contiguous** rows in `t` and cross a bank
boundary at most once per loop rather than ping-ponging.

**2. The positional table no longer fits the `$A000` window.** Embedding is
`64*64 = 4,096` bytes and the positional table is `85*64 = 5,440`; together
9,536 > 8,192. They now get a bank each and `embed_pos` copies the embedding
row into `XVEC`, switches `$5115`, and adds the positional row in place. Two
bank writes per token.

**3. The four per-position arrays outgrew their page.** `SCORL`, `SCORH`,
`EXPE` and `P4HI` are `NCTX` entries each: 128 bytes at `T = 20`, **340** at
`T = 85`. The system-RAM map was re-laid so that three of them fill page 7 and
`P4HI` sits in BSS under the linker's `$0200-$02FF` cap. Every base is chosen
so `base_lo + max_index <= 255`, i.e. no indexed access in the map crosses a
page - which is a cycle, not a correctness issue, but it is measured cost and
was avoidable.

**4. Reset must clear four banks, not one,** and the battery-backed result
block moved into the 128 free bytes the cache leaves in the last bank. The
`-DDEBUG` snapshots have nowhere to live at `T = 85` and the build now
**refuses to assemble** rather than quietly overwriting the cache.

### The refactor is neutral at T = 20, and it is not free

Same weights (`runs/final_av2_bpe64_tau0.75.npz`), same seed token, rebuilt
with the new ROM at `NES_T=20`:

```
max|dW| = 0   over 102400 ternary weights -> EXACT
TOKENS MATCHING: 19/19  -> EXACT
TOTAL     23525053 cycles   13.1442 s   (mean 1238160 cycles/token)
```

against the pre-refactor `1,233,099`. **+5,061 cycles per token, +0.41%** -
the `$5113` write inside `kv_ptr` (240 calls per token at `T = 20`) plus the
extra 64-byte embedding copy. Recorded rather than rounded away: the
generalised ROM is very slightly slower at the old context, and that is the
price of the banking.

### T = 85 verified on random weights BEFORE any training

The whole point of having an exact host reference is that the ROM change can
be proved correct without waiting for a model. Random init at `T = 85`,
`nnz = 51,219`:

```
TOKENS MATCHING: 84/84  -> EXACT
TOTAL   144241504 cycles   80.5921 s   (mean 1717160 cycles/token)
```

**84 generated tokens, every one bit-identical to `host/ref.py`**, exercising
all four PRG-RAM banks, both `$A000` banks and the whole 340-byte score map.
Transcript: `out/T85_RANDOM_VERIFICATION.txt`.

## Two measurements taken BEFORE the retrain, which predict its outcome

Both are on the shipped `T = 20` cartridge, and both were run before any
`T = 85` model existed, so neither can be a post-hoc rationalisation.

### 1. The loss stops improving with context after about five tokens

`train/perpos.py` scores the held-out split position by position with the same
quantised forward pass the ROM runs:

```
model      runs/final_av2_bpe64_tau0.75.npz     T = 20
positions    nats/token     nats/char
0-4          2.3085         1.5879
5-9          1.9470         1.3393
10-19        1.9774         1.3602
```

The curve falls sharply from position 0 to about position 5 - roughly **seven
characters** of context - and is then **flat, and very slightly worse**, all
the way to position 19. Going from 5 tokens of context to 19 buys nothing
measurable. A model that cannot use the 20-token window it already has is not
obviously going to use an 85-token one.

### 2. The attention can only reach one or two positions, by construction

`train/attnspan.py` records the quantised softmax vector for every head at
every position over four real generations:

```
T = 20   heads logged = 456
layer  nonzero    mean dist  p95 dist   max dist   mass >19 back
0      1.38       1.00       2          7          0.00%
1      1.22       0.96       3          9          0.00%
2      1.14       0.91       4          11         0.00%
```

**On average 1.1 to 1.4 positions receive any weight at all**, and the mean
attention distance is 1.0 - the previous token. That is not a training
accident; it is the kernel. The quantised softmax normalises so that
`sum_t p_t <= 8` with every `p_t` an integer in `0..7`, so at most 8 positions
can carry weight whatever `T` is, and the exp table plus the power-of-two
normalisation drive it far below that ceiling in practice.

**The prediction this sets up:** `T = 85` will cost a lot of cycles and buy
little or no loss. If that is what the retrain shows, the ceiling is capacity
(or the softmax's resolution), not the window. Recorded here, in advance, so
the retrain is a test and not an illustration.

## The tau ladder at T = 85, against the published T = 20 ladder

Same 12,000 steps, same batch 192, same lr, same seed, same corpus and
vocabulary as the `T = 20` table above. The only difference is the context.

| arm | T | val nats/token | **val/char** | density | nnz | banks | fits 7? |
|---|---|---|---|---|---|---|---|
| twn tau 0.75 | 20 | 2.2435 | **1.5432** | 0.5441 | 55,711 | 7 | yes |
| twn tau 0.75 | **85** | 2.2933 | **1.5772** | 0.5346 | 54,743 | 7 | yes |

At matched training, the 85-token model is **0.034 nats/char worse** than the
20-token one - and it saw 4.25x as many tokens per step to get there. That is
the first hard number against the context hypothesis.

### The whole T = 85 pipeline verified on this 12,000-step arm

Before spending an hour on a longer run, the entire path was checked end to
end on it:

```
weights    102400   nonzero 54743   density 0.5346
max|dW| = 0   over 102400 ternary weights -> EXACT
   (pos.bin max|d| = 0 over 5440 values - the 85-row positional table)
TOKENS MATCHING: 84/84  -> EXACT
TOTAL    148158149 cycles   82.7805 s   (mean 1763787 cycles/token)
```

Transcripts: `out/T85_12K_VERIFICATION.txt`, `out/T85_PROFILE.txt`.

### Where a token goes at T = 85

`run_profile.py`, PROFILE build, marker overhead counted and subtracted:

| stage | T = 20, pos 18 | T = 85, pos 83 |
|---|---|---|
| total | 1,410,393 | **2,430,470** |
| ternary `gather_row` | 851,040 (60.3%) | 883,637 (**36.4%**) |
| attention | 310,602 (**22.0%**) | 1,298,175 (**53.4%**) |
| everything else | 214,935 (15.2%) | 214,842 (8.8%) |
| ternary kernel | 16.30 cycles/MAC | 16.14 cycles/MAC |

The ternary kernel does not move - it is `nnz` work per token whatever `T` is,
and 16.14 vs 16.30 cycles/MAC is the different `nnz` of a different model, not
a different kernel. **Attention goes from a fifth of a token to more than
half**, exactly as `O(T^2)` says it must.

### The attention DOES reach further - and it does not help

`train/attnspan.py` on the same arm:

```
T = 85   heads logged = 2016
layer  nonzero    mean dist  p95 dist   max dist   mass >19 back
0      1.69       1.41       4          19         0.00%
1      1.55       2.19       8          29         0.31%
2      1.51       6.12       22         43         8.42%

ATTENTION MASS LANDING MORE THAN 19 POSITIONS BACK: 2.98%
```

So the longer window is not inert: layer 2's mean attention distance goes from
0.91 to **6.12**, its p95 from 4 to **22**, and 8.42% of its mass now lands
where the `T = 20` cartridge could not reach at all. The model built a
long-range head when it was given somewhere to put it.

And the loss does not care. `train/perpos.py` on the same arm:

```
positions    nats/token     nats/char
0-4          2.5819         1.7761
5-9          2.3052         1.5857
10-19        2.2828         1.5703
20-39        2.2648         1.5579
40-59        2.2634         1.5569
60-84        2.2788         1.5675
```

From position 10 onward the curve is **flat to within 0.7%**, and positions
60-84 - with four times the context of position 19 - are 0.18% better than
positions 10-19. That is noise. The 3.95% figure for "positions 20+ versus
positions 0-19" is entirely the first five positions, which have almost no
context by definition and would be hard for any model.

### A dilution effect that makes the negative result stronger, not weaker

The training and eval batches are random windows of length `T` and the loss is
the mean over all `T` positions. Position 0 has no context and is expensive;
position 80 has plenty. So a **longer window mechanically flattens the average
downwards** - at `T = 10` the four hardest positions are 40% of the reported
loss, at `T = 85` they are 5%.

`T = 85` is worse than `T = 20` at matched training *despite* that advantage.
And the per-position table shows why: the `T = 85` model is worse **at matched
positions**. Predicting token 15 from tokens 0..14 - a task both models can see
completely - costs 1.9774 nats/token for the `T = 20` model and 2.2828 for the
`T = 85` one. The longer model spends capacity on 85 rows of positional table
and on a wider attention pattern, and it is the same 102,400 ternary weights
either way.

## The context length has an INTERIOR OPTIMUM, and 20 is already at it

The decisive experiment is not `T = 20` versus `T = 85`, it is the curve.
Four identical runs - `tau = 0.75`, 12,000 steps, batch 192, lr 3e-3, seed 1,
same corpus, same vocabulary, same shifts - with **nothing changed but `T`**:

| T | chars of context | val nats/token | **val nats/char** | density | nnz |
|---|---|---|---|---|---|
| 10 | ~15 | 2.3051 | **1.5854** | 0.5404 | 55,335 |
| **20** | **~29** | **2.2231** | **1.5289** | 0.5373 | 55,018 |
| 40 | ~58 | 2.2310 | **1.5344** | 0.5343 | 54,714 |
| 85 | ~124 | 2.2933 | **1.5772** | 0.5346 | 54,743 |

It is a U, and the minimum is at **20** - the value the cartridge already had.
Going to 40 costs 0.0055 nats/char, going to 85 costs **0.0483**, which is more
than five times the 0.009 nats/char seed noise measured earlier in this
journal. Halving the context to 10 costs 0.0565, about the same as
quadrupling it to 85.

And the longer arms are being *flattered* by the averaging window (see above):
at `T = 85` the four context-starved leading positions are 5% of the reported
loss where at `T = 10` they are 40%. Correcting for that would make the T = 85
column worse, not better.

(The `T = 20` control here reads 2.2231 where the published 12,000-step arm
read 2.2435. Same steps, batch, lr, seed and tau; the difference is fp32
summation order in the rewritten attention contraction - the gradients agree
to ~1e-6 per step and 12,000 steps of that compounds to 0.014 nats/char. It is
above the 0.009 seed noise, so the comparison above uses the *rerun* control,
which shares its code with every other row. The published number is quoted
here rather than quietly replaced.)

### tau still has its interior optimum at 0.75

| arm | T | val nats/token | val/char | density | nnz | fits 7 banks? |
|---|---|---|---|---|---|---|
| tau 0.50 | 85 | 2.2904 | **1.5752** | 0.6574 | 67,313 | **NO** |
| **tau 0.75** | 85 | 2.2933 | **1.5772** | 0.5346 | 54,743 | **yes** |
| tau 1.00 | 85 | 2.3081 | **1.5874** | 0.4105 | 42,033 | yes |

Same shape as at `T = 20`. tau 0.50 edges tau 0.75 by 0.002 nats/char, which
is a fifth of the measured 0.009 seed noise and therefore nothing, and it needs
67,313 index bytes against the seven-bank window's 57,232 - it does not fit the
cartridge, exactly as at `T = 20` where 0.50 gave density 0.674. tau 1.00 is
clearly worse at both context lengths.

**The tau finding carries over unchanged: 0.75 is the best arm that fits, and
it still lands just inside the ROM's capacity** (54,743 of 57,232 index bytes,
2,489 to spare - at `T = 20` it was 55,711 with 1,521 to spare).

## The decisive table: loss at MATCHED positions

Every model above can be asked exactly the same question - "predict token 15
given tokens 0..14" - and all of them have the full context for it. So the
per-position curves can be laid side by side, and the confound of the
averaging window disappears entirely. All four arms, 12,000 steps, tau 0.75,
identical everything except `T` (`out/PERPOS_T*.txt`):

**nats per token, by position band, lower is better**

| positions | T = 10 | T = 20 | T = 40 | T = 85 |
|---|---|---|---|---|
| 0-4 | 2.4054 | **2.4241** | 2.4994 | 2.5819 |
| 5-9 | 2.2188 | **2.1332** | 2.2064 | 2.3052 |
| 10-19 | - | **2.1677** | 2.1807 | 2.2828 |
| 20-39 | - | - | **2.1974** | 2.2648 |
| 40-59 | - | - | - | 2.2634 |
| 60-84 | - | - | - | 2.2788 |

Two things fall out of it and they answer the question.

**1. The `T = 20` model is the best model at every band it can be compared on.**
At positions 5-9 it beats `T = 40` by 0.073 nats/token and `T = 85` by 0.172.
At positions 10-19 it beats `T = 85` by 0.115. These are positions where all
three models see identical inputs. The longer-context models are worse at the
task they share, which is what spending a fixed 102,400 weights on a bigger
positional table and a wider attention pattern looks like.

**2. Inside each long-context model, the extra context is not paying.** The
`T = 40` model is *worse* at positions 20-39 (2.1974) than at 10-19 (2.1807).
The `T = 85` model is 2.2828 at 10-19 and 2.2788 at 60-84 - **0.18% better with
four times the context**. The curve is flat from about position 10, i.e. from
about **fifteen characters**, in every model that has the room to show it.

The earlier "3.95% improvement from positions 20+" and the matching 3.06% for
`T = 40` are entirely the first five positions dragging the 0-19 average up.
Positions 0-4 are hard because they have no context, in every model, and no
amount of window fixes that.

---

# THE T = 85 CARTRIDGE

`runs/t85_final_tau0.75.npz` - bpe64, TWN tau = 0.75, `AV_SHIFT = 2`, 60,000
steps at batch 192, seed 1, lr 3e-3. **Exactly the recipe the shipped T = 20
model was trained with; `T` is the only difference.**

| | T = 20 (shipped) | T = 85 (this) |
|---|---|---|
| fit / val | 2.0431 / 2.0546 | 2.0795 / **2.0856** |
| **val nats per character** | **1.4133** | **1.4344** |
| uniform baseline | 2.861 nats/char | 2.861 nats/char |
| nonzero weights | 52,207 (0.5098) | **52,186 (0.5096)** |
| stream image | 57,344 B, 7 banks | 57,344 B, 7 banks |
| KV cache | 7,680 B, 1 PRG-RAM bank | **32,640 B, 4 banks** |
| characters of context | ~29 | **~124** |
| ROM image | 90,128 B | 106,512 B |

**Quadrupling the context made the model 0.0211 nats/char WORSE** - 1.5%,
against a measured seed noise of 0.009 nats/char. It saw 4.25x as many tokens
per optimisation step to get there, and the averaging window flatters it.

## max|dW| = 0

```
matrix       shape        nnz        max|dW|
...          (all 19 matrices)       0
embed.bin  max|d| = 0   over 4096 values
pos.bin    max|d| = 0   over 5440 values     <- the 85-row positional table
weights    102400   nonzero 52186   density 0.5096
max|dW| = 0   over 102400 ternary weights -> EXACT
```

## ROM == host, 252 tokens over three seeds

MAME 0.277 `nes`, cycle-exact write tap, full transcript in
`out/T85_FINAL_VERIFICATION.txt` and `out/T85_FINAL_SIDEBYSIDE.txt`.

| seed token | tokens | result | mean cycles/token |
|---|---|---|---|
| 1 `'b'` | **84/84** | EXACT | 1,729,505 |
| 26 `' '` | **84/84** | EXACT | 1,729,563 |
| 40 `' the '` | **84/84** | EXACT | 1,729,529 |

**252 generated tokens, every one bit-identical between the 6502 and the
exact-integer specification**, exercising all four PRG-RAM banks, both
`$A000` banks and the 340-byte score map. That is 84 per seed against a 16-token
bar - and the run before it, on random weights, was also 84/84, so the ROM
change was proved correct before a trained model existed.

## CYCLES PER TOKEN AND THE ATTENTION SHARE

| | T = 20 | T = 85 | change |
|---|---|---|---|
| position 0 | 1,103,615 | 1,104,175 | +0.05% |
| position 18 / 42 | 1,366,939 | 1,738,908 (pos 42) | |
| last position | 1,366,939 (pos 18) | **2,360,222 (pos 83)** | **+72.7%** |
| **mean over the run** | **1,233,099** | **1,729,505** | **+40.3%** |
| mean, same ROM code | 1,238,160 | 1,729,505 | +39.7% |
| seconds per token, mean | 0.6890 | **0.9663** | |
| | | | |
| seconds, last position | 0.7638 | **1.3187** | |
| whole generation | 13.09 s (19 tokens) | **81.17 s** (84 tokens) | |

(The `T = 20` column is the shipped cartridge, built before the ROM was
generalised. The generalised ROM measures 1,238,160 at `T = 20` - the +0.41%
banking overhead recorded earlier - so the like-for-like figure is +39.7%.
Both are given rather than picking whichever is more flattering.)

Per-stage, PROFILE build, marker overhead counted and subtracted
(`out/T85_FINAL_PROFILE.txt`):

| stage | T = 20, pos 18 | T = 85, pos 83 |
|---|---|---|
| total | 1,410,393 | **2,395,078** |
| ternary `gather_row` | 851,040 (60.3%) | 850,198 (**35.5%**) |
| **attention** | 310,602 (**22.0%**) | **1,296,088 (54.1%)** |
| everything else | 214,935 (15.2%) | 214,976 (9.0%) |
| ternary kernel | 16.30 cycles/MAC | **16.29 cycles/MAC** |

The two models have almost identical `nnz` (52,207 vs 52,186), so this is as
close to a controlled comparison of the cost of `T` as the port allows:
`gather_row` is 851,040 against 850,198 cycles - **0.1% apart** - and
everything else is within 41 cycles. **The entire 985,490-cycle difference is
attention.**

Attention at position `p` is `L*H*(p+1)*DH*2` MACs - **linear in the position**,
quadratic only when summed over a whole generation. So the predicted ratio at
the last position is `84/19 = 4.42`, and the measured one is
`1,296,088 / 310,602 = ` **4.17**. The 5.6% shortfall is the part of the
attention block that does not scale with `p`: the per-head accumulator zeroing,
the `av_quant` setup and the final requantise, which are paid `L*H = 6` times
per token whatever the context is. The DESIGN section 4 prediction holds.

Attention goes from **a fifth of a token to more than half.** The mean cost
rises much less than the peak (40.3% against 72.6%) because the mean sees
`(p+1)` averaged over the run, which is about half of full context either way.

## Block 16 is still provably and observably safe at T = 85

Over the real 84-token trajectory, 4.4x as many blocks as the T = 20 check:

| block | blocks | max value | signed >127 | biased >255 |
|---|---|---|---|---|
| 16 | 381,444 | **202** | 0 | **0** |
| 32 | 250,908 | 370 | 114 | **19,594 (7.8%)** |

Worst observed 202 against the provable bound of 224. Block 32 remains
refuted. Nothing about the longer context changes that argument, as expected -
it depends on the activation range, not on `T`.

## WHAT IT ACTUALLY SAYS

Greedy argmax, ties to the lowest index, which is what `rom/nn.s` does. The
ROM free-runs from one seed token; that token is the whole prompt. Full survey
in `out/T85_FINAL_SURVEY.txt`.

```
seed  1 'b'   -> 'ban who was beauticked and saw a big for with her mommy said, "you can your can your can you can you fo'
seed  4 'e'   -> 'e. he said, "what is mom!" lily. she said, "yes, made to for for his mommy." man was so happy to be a big fo'
seed 26 ' '   -> ' with his flower him. he was so hapy saw a big for for for his mommy and saw a big for for his mommy. she was '
seed 28 ','   -> ', she was very happpy. she said, "i worry, but is mommy!" lily said, "you can you can your can you can wory'
seed 31 '!'   -> '! lily looked and said, "what is mom!" mom is made to mommy said, "you can you can you can youre to for you'
seed 43 'wa'  -> 'was made to made to made and said, "i mom!" let's made to made to made to made and said, "thank you can '
seed 58 ', '  -> ', lily. she was very broom and said, "thank you for you can you for your for you for the boy for for you for fo'
```

**It does not form sentences. It forms the same phrases and then jams.**
Every one of these starts as recognisable English at the phrase level and
collapses into a repeated bigram - `for for for`, `you can you can`, `made to
made to` - somewhere between token 20 and token 40.

### Side by side at MATCHED length, which is the fair comparison

The looping above is partly a property of greedy decoding over 84 steps, and
the `T = 20` cartridge was never asked to run that long. So here is the same
`T = 85` model truncated to the 19 tokens the `T = 20` cartridge emits, against
the shipped model, same seeds:

```
seed      T = 20 (shipped, 1.4133)              T = 85 (new, 1.4344)
'b'    -> 'big friends. she was so hap'          'ban who was beauticked a'
' '    -> ' came to playing with hi'             ' with his flower him. he wa'
','    -> ', she said, "that's a proud '         ', she was very happpy. she said'
'!'    -> '! he was so happy. she was so '       '! lily looked and said, "w'
'he '  -> 'he said, "i was so happy. s'          'he said, "i we loook, but '
'wa'   -> 'was so happy and said, "i hav'        'was made to made to made '
'her'  -> 'her friends. she was so happy'        'her for his mom and said, "wha'
', '   -> ', he saw a big dad and said'          ', lily. she was very broom and sa'
'en'   -> 'enture. he was so happy and sai'      'ent. he said, "i wet!" ma'
```

At matched length the two are **the same kind of text**: correct punctuation
and quoting, `he`/`she` agreement within a clause, phrase-level fluency, no
sentence-level plan. The `T = 85` column is slightly more varied in content
(`lily looked and said`, `she was very happpy`, `i wet!`) and slightly worse
at spelling (`happpy`, `loook`, `beauticked`, `broom` for `brave`); the
`T = 20` column leans harder on the TinyStories cliches (`was so happy`,
`friends`). Neither is a sentence.

**Nothing that four times the context bought is visible in the text.**

## THE VERDICT: CAPACITY, NOT CONTEXT

Five independent measurements, and they agree.

**1. The aggregate loss got worse.** 1.4344 nats/char at `T = 85` against
1.4133 at `T = 20`, same recipe, same weights budget, same corpus. The longer
model saw 4.25x more tokens per step and is flattered by the averaging window,
and it still lost by 2.3 times the seed noise.

**2. The loss-versus-context curve has an interior optimum at 20.** At matched
12,000 steps: `T = 10` 1.5854, `T = 20` **1.5289**, `T = 40` 1.5344, `T = 85`
1.5772 nats/char. Twenty is not a compromise the ROM forced; it is where this
model is best.

**3. At matched positions the short-context model wins everywhere.** Both
60,000-step models, asked exactly the same question:

| positions | T = 20 | T = 85 |
|---|---|---|
| 0-4 | **2.3170** | 2.4184 |
| 5-9 | **1.9573** | 2.0958 |
| 10-19 | **1.9660** | 2.0697 |
| 20-39 | - | 2.0552 |
| 40-59 | - | 2.0559 |
| 60-84 | - | 2.0636 |

The `T = 85` model at position 84, with **124 characters** of context, scores
2.0636 - still worse than the `T = 20` model at positions 10-19 with **15 to 28
characters**. Context did not buy back what the extra 65 rows of positional
table and the wider attention cost.

**4. Inside the long model, context past ~10 tokens does nothing.** Positions
10-19 are 2.0697 and positions 60-84 are 2.0636: **0.3% better for four times
the context**, and 20-39 through 60-84 vary by 0.4% with no trend. The curve is
flat from about position 10, i.e. from about **fifteen characters**. That is
the same flattening the `T = 20` model already showed, at the same place, and
it was measured and written down *before* this model existed.

**5. The attention DID learn to reach - and it did not help.** Layer 2's mean
attention distance went from 0.91 to **7.82**, its p95 from 4 to **30**, and
**13.70%** of its mass now lands more than 19 positions back, where the
`T = 20` cartridge could not reach at all:

```
T = 20                                    T = 85
layer nonzero mean p95 max  >19back      layer nonzero mean p95 max  >19back
0     1.38    1.00 2   7    0.00%        0     1.75    1.39 3   80   0.24%
1     1.22    0.96 3   9    0.00%        1     1.44    1.62 5   80   0.15%
2     1.14    0.91 4   11   0.00%        2     1.43    7.82 30  63   13.70%
```

This is the strongest form of the negative result. The failure is **not** that
the model ignored the window. Given somewhere to look, it built a long-range
head and used it, and the loss got worse anyway.

### Why, mechanically

Two things bound it and neither is the window.

*The softmax can only address about 1.5 positions.* The quantised softmax
normalises so `sum_t p_t <= 8` with every `p_t` an integer in `0..7`, so at
most 8 of the `T` positions can carry any weight - and measured, the number
that actually do is **1.14 to 1.75**. Widening `T` widens *where* the model may
look, not *how much* it may look at. A 4-bit probability nibble cannot
represent a distribution over 85 things.

*The weights are the binding budget.* 102,400 ternary weights have to encode
the positional table too, and it went from 20 rows to 85. Ternarising the
weights costs about 3.6% (measured earlier in this journal); quadrupling the
context costs 1.5% and buys nothing. Both are small compared with the gap to
fluent English, which is what "three 64-wide layers" buys you.

**The ceiling is capacity.** More weights, a wider `D`, more layers, or a
mixture-of-experts across the 32 spare PRG banks - not a longer window. And if
context were revisited, the softmax resolution would have to be fixed first:
a `sum <= 8` integer softmax over 85 positions is the constraint that makes the
long-range head the model *did* learn unable to pay for itself.

## What could not be done, and what is thinner than it looks

* **The headline 60,000-step pair is one seed each.** The `T = 20` vs `T = 85`
  aggregate gap is 0.0211 nats/char against a measured seed noise of 0.009 -
  **2.3x**, which is a real margin but not a comfortable one. The supporting
  evidence is much stronger than the headline: the matched-position gap at
  positions 10-19 is 0.070 nats/char (7.8x seed noise) and the 12,000-step
  sweep gap is 0.048 (5.4x). A seed-2 replicate of both 60,000-step arms is
  running as this is written and its result is recorded below.
* **`AV_SHIFT`, `K_SHIFT` and `W2_SHIFT` were not re-laddered at `T = 85`.**
  `AV_SHIFT = 2` was the measured optimum at `T = 20` and the accumulator bound
  that motivated it (`sum_t p_t <= 8`, values in `-7..7`, so `|acc| <= 14`) is
  independent of `T`, so it should carry - but "should" is not "measured", and
  the one time this repo assumed a shift it produced fluent-looking rubbish
  with nothing raising. `K_SHIFT = 2` saturating 31.67% of the time was already
  flagged as worth a ladder at `T = 20` and still has not had one.
* **The softmax resolution was not changed, only measured.** The `sum <= 8`
  integer softmax is identified above as the mechanism that stops the
  long-range head from paying, and nothing was done about it. Widening the
  probability nibble, or changing the normalisation target, is the obvious
  follow-up and was out of scope for a context experiment.
* **`-DDEBUG` does not build at `T = 85`.** The snapshot pages have nowhere to
  live once the KV cache fills all four PRG-RAM banks, so the build refuses to
  assemble rather than corrupting the cache. If a `T = 85` ROM ever *did*
  disagree with the host, the first tool anyone would reach for is missing.
  Routing the dumps through the marker port would fix it and was not done.
* **Greedy argmax only, and the 84-token outputs loop.** The repetition in the
  `T = 85` text is partly greedy decoding over four times as many steps, which
  the `T = 20` cartridge was never asked to do. The matched-length comparison
  above is the honest one and it is the one the verdict rests on; the 84-token
  strings are quoted because they are what the cartridge produces, not because
  they are a fair quality comparison.
* **One corpus, one vocabulary, one model shape.** TinyStories, bpe64, and
  `3 x 64` layers. The conclusion "the ceiling is capacity" is a statement about
  *this* 102,400-weight model. A larger model might well have a context
  optimum past 20 - that is the point of calling it a capacity limit.
* **No sliding window.** The ROM still cannot generate past `NCTX - 1` tokens
  and there is still no prompt path; the multi-token prompts remain host-only.
* **The ares cross-check was not repeated.** The `.sav` block moved to the tail
  of the last KV bank and the offset changed; nothing was run against ares 147
  at `T = 85`.
