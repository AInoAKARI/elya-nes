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
