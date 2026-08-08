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
  *(Superseded - see "Attention kernel optimisation" at the end of this file.
  The wall is real but it is avoidable: only ONE of attention's two kernels
  has two varying operands, and neither of them does once the loops are
  transposed. Attention is now 7.3% of a token at full context.)*
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

# Attention kernel optimisation (session: attention)

## Baseline, reproduced from the committed cartridge

`train/build_trained.sh runs/final_av2_bpe64_tau0.75.npz 1` rebuilds
`out/nn.nes` **byte-identically** to the committed cartridge (git reports the
working tree clean after the build), so everything below is measured against
the shipped ROM, not a lookalike.

| measurement | value |
| --- | --- |
| tokens | 19/19 exact vs host |
| mean cycles/token | **1,233,099** |
| pos 0 | 1,103,687 |
| pos 18 (full context) | **1,366,939** |
| profiled pos-18 attention | **302,624 cycles, 21.6%** |
| profiled pos-0 attention | 39,812 cycles, 3.5% |
| ternary gather in situ | 16.30 cycles/MAC, 8.31 cycles/weight (nnz 52,207) |

The brief's 1,233,099 / 302,444 / 21.8% / 16.34 / 8.19 figures come from
`out/nn_profile.txt`, which was taken at nnz 51,299 — a different training run.
The cycles/token figure agrees to the digit; the attention figure differs by
0.06%. All deltas below are against **302,624 / 1,366,939 / 1,233,099**, the
numbers this session measured on the ROM it is actually editing.

## Where the 302 k cycles actually go

`tools/run_attn_profile.py` on an `-DATTNPROF` build (five nested 12-cycle
marker pairs, each enclosing region corrected for the markers inside it):

| region | pos 0 | pos 18 (full context) |
| --- | ---: | ---: |
| QK kernel (`dot_qk`) | 5,452 | **103,568** |
| score-loop other (`kv_ptr`, score store, loop) | 726 | 12,162 |
| softmax | 2,316 | **33,046** |
| AV kernel (`acc_av`) | 5,160 | **98,040** |
| AV other (`av_fold`, `av_quant`, zeroing, `kv_ptr`) | 26,074 | 56,204 |
| accounted | 39,728 | 303,020 |

Measured per-MAC, **in situ**:

| kernel | cycles/call | MACs/call | cycles/MAC |
| --- | ---: | ---: | ---: |
| `dot_qk` | 908.5 | 32 | **28.39** |
| `acc_av` | 860.0 | 32 | **26.88** |

The 23 and 25 in the brief are the straight-line inner loops; the extra 3-5 is
the per-block carry fold and the call frame. Two things this measurement
changes about the plan:

* **`softmax` is 11% of attention** and was not on the list at all. 5,508
  cycles for 19 positions is 290 cycles per position, for what is a table
  lookup and a shift.
* **"AV other" is 18.5%** - `av_fold` and `av_quant` and the per-t `kv_ptr`
  together cost more than the score loop's entire non-kernel overhead. A
  rewrite that keeps `acc_av` but leaves the bookkeeping alone can win at most
  a third of the AV cost.

## The self-modifying-code plan needs three things from MMC5; all three measured

`ca65 -DRAMEXEC` builds a ROM whose only job is to answer this, because the
whole optimisation depends on it and "MMC5 can probably do that" is not a
measurement. Markers emitted: `111 112 113 114 255` - all four checks pass.

| check | marker | result |
| --- | --- | --- |
| `$5114 = 5` (bit 7 clear) maps PRG-**RAM** at `$8000`, writable | 111 | yes |
| a subroutine assembled into that RAM **executes** | 112 | yes |
| the `$6000` window (bank 4) keeps its own contents meanwhile | 113 | yes |
| RAM banks 4,5,6,7 at `$6000` are four **independent** 8 KB banks | 114 | yes |

So the plan is legal: self-modified kernels live in RAM bank 5 mapped at
`$8000` (the weight-stream window, which attention never reads), the KV cache
stays in the `$6000` window, and there is a spare bank for a second cache
layout.

## Stage A: AV at 8.00 cycles/MAC - stop building the index, move it into the instruction

The wall the brief describes is real but it is not the whole story. Building
`(q<<4)|k` needs `ora`, `ora` has no X/Y form, so the accumulator is evicted
from A. **But only one of attention's two kernels actually has two varying
operands.** In AV,

    att[d] = sum_t mul[(p_t << 4) | v_t[d]]

`p_t` does not depend on `d`. So if `d` is the OUTER loop and `t` the inner
one, the multiply table ROW is fixed for the whole of an inner iteration, and
a fixed row is an *address*, not a register operand. There is no index to
build, so nothing evicts the accumulator:

    ldx VBASE + t*256, y        ; 4   y = curlay*64 + d
    adc tbl_mul + (p_t<<4), x   ; 4   low byte self-modified

That transposition is the whole trick, and it needs three things arranged
around it:

* **`d` outer, `t` inner** - the opposite of the old loop, which had to be
  `t` outer because it accumulated 32 separate sums in `ACC8`.
* **`t` in the instruction, not a register**, so the multiply row can be
  patched per t. That means the t loop is unrolled: 20 units, laid out in
  DESCENDING t, entered at unit `19 - curpos` so it covers exactly
  t = curpos..0. Addition is commutative, so the reversed order cannot
  change a sum.
* **the kernel in RAM** - PRG-RAM bank 5 mapped at `$8000`, the
  weight-stream window, which attention never reads. `reset` copies the
  kernel there from the `$A000` bank.

The value cache moved to its own PRG-RAM bank in the layout the kernel wants,
`V[t][l][d] = $6000 + t*256 + l*64 + d`. The 256-byte t stride is not
padding for its own sake: it makes the base a build-time constant AND makes
`ldx base,y` page-cross free for all 192 (l,d) offsets, which is worth a
whole cycle per MAC.

### Isolated (ATTNBENCH build, real chain, t-count swept 1..20)

```
  MACs   cyc/call     kernel    cyc/MAC
  1         58.08      38.08     38.078
  10       130.08     110.08     11.008   d=+8.00
  11       151.08     131.08     11.916   d=+21.00   <- carry fold
  19       215.08     195.08     10.267   d=+8.00
  20       227.08     207.08     10.354   d=+12.00   <- carry fold
```

**Every step is exactly +8.00 cycles.** The two visible jumps are the two
16-bit carry folds; the folds are ten units apart because 10 x 25 = 250 < 256
is the largest block that provably cannot set carry, the same argument that
licenses the ternary gather chain. Intercept ~30 cycles = `jsr` + `jmp
(avp)` + `rts` + the closing fold.

### Measured, before and after

| | before | after |
| --- | ---: | ---: |
| AV kernel body, isolated | 26.88 cyc/MAC | **8.00 cyc/MAC** (slope) |
| AV kernel, in situ (pos 18) | 98,040 cyc | **38,720 cyc** |
| whole AV section, in situ | 154,244 cyc | **56,344 cyc** |
| whole AV section per MAC | 42.28 cyc/MAC | **15.44 cyc/MAC** |
| attention (PROFILE build, pos 18) | 302,624 (21.6%) | **206,932 (15.9%)** |
| cycles/token, mean | 1,233,099 | **1,184,165** |
| cycles/token, pos 18 | 1,366,939 | **1,270,349** |

**57/57 tokens exact** against the host reference at seed tokens 1, 26 and 40
(19 tokens each, well past the 16-token bar), with `max|dW| = 0` on the
packed weights each time.

One bug worth recording because it passed positions 0 and 1: the AV bias
`MULBIAS*(curpos+1)` was hoisted out of the per-head loop into `attention`,
where it was written to `sumL/sumH` - which `softmax` then overwrites on its
way past. Positions 0 and 1 still came out right. Position 2 did not. That
is the third time in this project that a change has produced the correct
token at position 0 and 1 while being wrong; a 3-token check would have
shipped it.

## Stage B: QK at 8-cycle units, on a key cache turned inside out

QK is the mirror image of AV:

    score_t = sum_d mul[(q_d << 4) | k_t[d]]

Here it is `q_d` that is constant along the summation axis, so the same trick
applies with the roles of the two axes swapped: **d is the unrolled/patched
axis and t is the register axis.** Which means the key cache has to be
TRANSPOSED, because the address must now be linear in t:

    KT[d][l][t] = $6000 + d*64 + (l*20 + t)        (RAM bank 4)
    V [t][l][d] = $6000 + t*256 + (l*64 + d)       (RAM bank 6)

K and V want opposite layouts. That is the real reason attention was worse
per operation than the gather beside it: **one cache cannot serve both
kernels**, and the old code made both of them pay for the compromise. Two
PRG-RAM banks, one layout each, and both kernels get a page-cross-free
`abs,y` load.

The patching is free of extra passes. `post_q` already walked the 64 query
nibbles; it now writes each one straight into its chain unit's operand as
well (`tbl_mul` is page aligned, so the operand's low byte *is* `q<<4`).
Unrolled, that costs 22 cycles an element against the 26 the old loop cost -
the patch is cheaper than the loop it replaced. Same for the key scatter:
unrolled `sta KTBASE+d*64,x` is 11 cycles an element against the 19 the old
`sta (kptr),y` pointer walk cost, and it writes the transposed layout for
free.

### Measured

| | before | after |
| --- | ---: | ---: |
| QK kernel, isolated (32 MACs) | 908.5 cyc = 28.39 cyc/MAC | **357.1 cyc = 11.16 cyc/MAC** |
| QK kernel, in situ (pos 18) | 103,568 cyc | **41,438 cyc** |
| score-loop other (`kv_ptr` etc.) | 12,054 cyc | **4,536 cyc** |
| attention (PROFILE, pos 18) | 302,624 (21.6%) | **136,954 (11.1%)** |
| cycles/token, mean | 1,233,099 | **1,145,564** |

The 11.16 is the whole call: 32 units at exactly 8.00 = 256, four 16-bit
carry folds, the prologue/epilogue, and 17 cycles of `jsr` + `jmp (qkp)` +
`rts`. The 8.00 per unit is the same figure the AV sweep measured
element by element.

**57/57 tokens exact** at seed tokens 1, 26, 40. The random-init `./build.sh`
cartridge also verifies 19/19 exact, so the change is not specific to the
trained weights' sparsity. All seven build variants (default, PROFILE, BENCH,
DEBUG, ATTNPROF, ATTNBENCH, RAMEXEC) assemble and link.

## Stage C: 86.9% of the AV work was multiplying by zero

`mul[(0<<4)|v] = floor(0*v/4) + 13 = 13` for **every** v. So a position whose
softmax nibble is zero contributes exactly `MULBIAS` to every one of the 32
sums and nothing else. Counting them on the trained model (host reference,
19 tokens, all 3 layers x 2 heads):

| context length | positions | with p != 0 | pure bias |
| ---: | ---: | ---: | ---: |
| 2 | 6 calls | 1.33 | 33% |
| 8 | 6 calls | 1.00 | 88% |
| 15 | 6 calls | 1.00 | 93% |
| 19 | 6 calls | 1.33 | **93%** |
| all | 1,140 slots | 149 | **86.9%** |

The quantised softmax is far more peaked than the float one it approximates:
`p = min(e >> kk, 7)` with `kk` chosen so `sum(e) >> kk <= 8` drives almost
everything to zero. That is the same effect the AV_SHIFT finding earlier in
this file describes from the other end - attention here really does look at
one or two positions.

So the AV chain is now **built per head**, not just patched: `av_patch` walks
`P4HI`, drops every zero, and packs the survivors into the top of the 20-unit
chain. Both operands of a live unit are written (the V page, because the t
stride is a whole page, and the multiply row). Entry is at unit
`NCTX - avn`, and the bias becomes `MULBIAS * avn` instead of
`MULBIAS * (curpos+1)` - which is exactly the contribution the dropped units
would have made, so the arithmetic is unchanged. `avn = 0` is possible
(if `kk >= 7` even the argmax quantises to zero) and lands on a bare `rts`.

This is the same shape as the ternary gather kernel next door, which has
always skipped zero weights. Attention was the only kernel in the ROM still
paying for its zeros.

| | before Stage C | after |
| --- | ---: | ---: |
| AV kernel, in situ (pos 18) | 38,720 cyc (201.7/call) | **8,960 cyc (46.7/call)** |
| AV section, in situ | 56,524 | **28,166** |
| attention (PROFILE, pos 18) | 136,954 (11.1%) | **107,267 (8.9%)** |
| cycles/token, mean | 1,145,564 | **1,130,955** |

**57/57 tokens exact** at seed tokens 1, 26, 40.

Caveat worth stating plainly: **this speedup is data dependent.** The
arithmetic is exact for any model, but a model with a flatter attention
distribution would keep more units and save less. The 8.00 cycles/MAC of
Stages A and B are not data dependent; this is.

## Stage D: the AV dimension loop, once the kernel stopped being the cost

With the chain down to ~1.3 units, AV's cost was no longer the multiply-adds -
it was the 98 cycles of frame around each of the 32 dimensions. Three changes,
all measured together:

* **Seed the accumulator with the bias instead of subtracting it.** `nb =
  -(MULBIAS * avn)` is computed once per head and written into `totL/totH`
  before the chain runs; two's-complement addition does the rest. That
  deletes a 20-cycle 16-bit subtract per dimension for the price of 6.
* **Patch the V base's low byte too**, so `y` is just `d` rather than
  `curlay*64 + d`. That removes the second loop counter and its `inc`.
  (The base low byte is `curlay*64` <= 128 and `d` <= 63, so the load still
  never crosses a page.)
* **Inline `requant_k4`** - the `jsr`/`rts` alone was 12 cycles a dimension.

| | before Stage D | after |
| --- | ---: | ---: |
| AV section, in situ (pos 18) | 28,166 (146.7/dim) | **22,089 (115.1/dim)** |
| "AV other" | 19,206 | **12,645** |
| attention (PROFILE, pos 18) | 107,267 (8.9%) | **101,229 (8.5%)** |
| cycles/token, mean | 1,130,955 | **1,124,698** |

57/57 exact at seed tokens 1, 26, 40.

### A negative: the same zero-skip does NOT pay for QK

`mul[(0<<4)|k]` is the constant 13 just as `mul[(p<<4)|0]`... is not - the
symmetric saving for QK would be query nibbles equal to zero. Counted on the
trained model over 19 tokens: **66 of 1,216 layer-0 query nibbles are zero,
5.4%.** The distribution is bimodal at the saturation points (`-7`: 246,
`+7`: 278) and nearly flat in between. Skipping 5.4% of 32 units would save
about 14 cycles a call and cost more than that to detect, so QK keeps its
fixed 32-unit chain. Attention's sparsity lives entirely in the softmax, not
in the queries.

## Stage E: softmax was 33% of what was left, and it was two shift loops

Nothing here is clever - softmax simply had two runtime shift loops that are
table lookups:

* **`(s - max) >> SM_SHIFT` with a clamp** was a 3-iteration
  `lda/cmp/ror/ror/dex/bne` (60 cycles) plus a 15-cycle range test. The
  difference is always <= 0, so its HIGH byte selects one of three cases and
  its LOW byte indexes a 256-byte table: high `$FF` -> `tbl_sm[low]`, high
  `$00` -> the difference is exactly zero -> `EXPTAB[14]`, anything else ->
  past the bottom of the table -> `EXPTAB[0]`. 113 cycles a position becomes
  32.
* **`min(e >> kk, 7) << 4`** was a `kk`-long `lsr` loop *per position*, but
  `kk` is fixed for the whole softmax. One row of a 9 x 65 table is selected
  once per call and the per-position cost is `lda (smp),y`. Rows from
  `kk = 7` up are entirely zero because `e <= max(EXPTAB) = 64`, which is why
  clamping the row index at 8 is exact rather than a fudge - and `sum(e)`
  cannot exceed `T * 64 = 1280`, so `kk <= 8` anyway.

Both tables are generated by `host/ref.py` from `SM_SHIFT` and `EXPTAB`, and
their geometry constants go into the same generated `shifts.inc` the shifts
already travel in, so the ROM still cannot disagree with the specification
about them. They live in the `$A000` bank; the `$C000` table bank had ~470
bytes left and needed 841.

| | before Stage E | after |
| --- | ---: | ---: |
| softmax, in situ (pos 18) | 33,052 (5,508.7/call) | **19,462 (3,243.7/call)** |
| attention (PROFILE, pos 18) | 101,229 (8.5%) | **87,594 (7.4%)** |
| cycles/token, mean | 1,124,698 | **1,117,741** |

57/57 exact at seed tokens 1, 26, 40, and the random-init `./build.sh`
cartridge verifies 19/19 exact too (the packer changed, so that one matters).
