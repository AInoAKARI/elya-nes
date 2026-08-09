# NES transformer port - ROM layout and cost model

Written **before** the kernel, as the brief requires. Every number here is
either a measured primitive from `FINDINGS.md` or arithmetic on top of one.
Where it is a prediction, it is labelled as such and is checked against a
measurement later in `FINDINGS.md`.

## 0. The constraint that shapes everything

The 6502 has one accumulator and two index registers, and **no `ORA`/`ADC` to
X or Y**. A ternary dot product with the accumulator resident in A therefore
has exactly one shape:

```
ldy <stream>,x      ; 4   fetch the next input index from the weight stream
adc <activations>,y ; 4   accumulate that activation
```

Both operands must be reached through an index register, so **the weight
stream must be addressed by an index register too**, which means its base has
to be an *assembled absolute address*. Anything else - `lda (ptr),y` to walk
the stream - puts the stream byte in A and evicts the accumulator, which is
what turns an 8-cycle inner op into a 19-cycle one.

This is the same structural fact that killed the 4-bit-weight LUT design in
the prior run, and it is the single reason for the "page chain" layout below.

## 1. Carry: why activations are stored biased

A chain of `adc` without `clc` is 8 cycles per element (measured) but is
*arithmetically wrong* - each `adc` adds the previous unsigned carry. Adding
`clc` costs 2 and puts us back at 10.

Instead the activations are stored **biased**: `actb[i] = act[i] + 7`, so
every stored byte is in `0..14` and every partial sum is non-negative. With a
block of **16** the largest possible block sum is `16*14 = 224 < 256`, so the
carry flag **provably never sets** and no `clc` is needed anywhere inside a
block. The bias is removed once per list:

```
sum = (biased_sum - 7 * n)
```

This is the same headroom argument as the prior run's `16*7 = 112` bound, just
shifted into unsigned space, and it is the reason block size is **16 and not
32**: 32 would give 448 > 255 and the carry-free guarantee would be lost.
Block 32 is not "probably fine here" - it is provably wrong for this
formulation.

## 2. Model shape, chosen from the RAM budget

| | |
|---|---|
| vocab `V` | 64 |  <!-- was documented as 128; the ROM (rom/nn.s NVOCAB=64), the host
                          reference (host/ref.py V=64) and the committed weight count all
                          say 64. 98,304 layer weights + 64*64 embedding = 102,400, which
                          is the figure in FINDINGS; V=128 would give 106,496. -->
| d_model `D` | 64 |
| layers `L` | 3 |
| heads `H` / d_head | 2 / 32 |
| d_ff `F` | 128 |
| context `T` | **20 or 85** - a build parameter (`NES_T` / `-DNCTX`) |

`T` is the only shape knob that varies, and the two values are the two ends of
what the cartridge can hold: 20 fills one PRG-RAM bank, 85 fills all four.
Everything else below is independent of it.

Ternary weight count:

```
per layer  Wq,Wk,Wv,Wo  4 x 64x64  = 16,384
           W1 (F x D)   128x64     =  8,192
           W2 (D x F)   64x128     =  8,192
                                    -------
                                     32,768
x 3 layers                          = 98,304
output head (V x D) 64x64           =  4,096
                                    -------
TOTAL ternary weights                102,400
```

(This block previously read `128x64 = 8,192` and `TOTAL 106,496`, left over
from the same V=128 doc error the vocab row above was corrected for. The
committed weight count, `rom/nn.s` `NVOCAB = 64` and `host/ref.py` `V = 64`
all say 102,400.)

(The prior run's extrapolation was quoted for 114,688 active weights; this
shape is within 7% of that, so the two are directly comparable.)

### The $6000 window arithmetic

> **This describes the LEGACY attention path only.** The attention kernels
> replaced it: K and V no longer share one interleaved cache, they want
> opposite layouts and live in separate PRG-RAM banks. See "The attention
> kernels and their two caches" below. The layout described here is still
> assembled, and is selected automatically for any `T` the kernels cannot
> address - which is what keeps the `T = 85` experiment buildable.

KV cache is `L x T x 2 x D` bytes, one byte per 4-bit activation:

```
T = 20:  3 x 20 x 2 x 64 =  7,680 bytes   93.75% of ONE 8 KB PRG-RAM bank
T = 85:  3 x 85 x 2 x 64 = 32,640 bytes   99.61% of ALL FOUR
```

Measured PRG-RAM is **32 KB in 4 banks at `$5113 = 4,5,6,7`** (declaring more
in the iNES header still yields 4 - reconfirmed this run), so

```
T_max = 32768 / (L x 2 x D) = 32768 / 384 = 85 positions
```

is a hard ceiling on the legacy path, not a preference. `T = 20` was chosen
originally so that the cache fitted one bank and the attention loop never had
to switch mid-flight; `T = 85` gives that up and pays for a bank register
write per KV row address.

**The attention kernels have a much lower ceiling, and it is not capacity.**
They reach their caches with assembled absolute constants, which is the whole
reason the accumulator can stay in `A`. That costs two bounds:

```
QK:  ldx KTBASE + d*64, y      y = l*T + t
     y is one byte and the d stride is 64, so   L*T <= 64  ->  T <= 21
AV:  ldx VBASE  + t*256, y     y = l*64 + d
     the base is an address inside the $6000 window, so
     VBASE + (T-1)*256 + 255 <= $7FFF           ->  T <= 32
```

`T <= 21` binds. Raising it means giving the key cache a stride of `L*T`
instead of 64, which at `T = 85` is `64 * 255 = 16,320` bytes - two PRG-RAM
banks - and an assembled absolute address cannot cross a bank. And even if it
could: at `T = 85` the two caches together need all 32,768 bytes of PRG-RAM,
so there is no bank left for the writable kernels. Both bounds are `.assert`-ed
in `rom/nn.s`; a build above them selects the legacy path rather than emitting
something that looks right.

**KV row alignment, and why banking is safe.** Rows are `D = 64` bytes, laid
out at row index

```
row = (l*2 + kv) * T + t          layout [layer][k|v][t]
addr = $6000 + (row & 127) * 64,  $5113 = 4 + (row >> 7)
```

`8192 / 64 = 128` rows fill a bank **exactly**, so a row can never straddle a
bank; and every row is 64-byte aligned, so `(kptr),y` with `y < 64` cannot
cross a page and the measured `$6000` page-cross +1 never fires. Both
guarantees are the same ones the one-bank version relied on, and both survive
the extension unchanged - which is the entire reason 64 bytes was chosen over
full 256-byte alignment.

The layout is `[layer][k|v][t]` rather than the original `[layer][t][k|v]` so
that the score loop and the AV loop walk **contiguous** rows in `t`. Each such
loop therefore crosses a bank boundary at most once instead of alternating
between two banks on every position.

### System RAM map (2 KB)

The four per-position arrays are `T` entries each: 128 bytes at `T = 20`,
**340** at `T = 85`. That does not fit page 7 alongside the 16-bit AV
accumulators, so the map is derived from `T`:

| range | use |
|---|---|
| `$0000-$00FF` | zero page: stream pointers, counters, 16-bit accumulators |
| `$0100-$01FF` | stack |
| `$0200-$02FF` | BSS: `res_tokens` (96), `res_ntok`, `P4HI` (T) - linker-capped |
| `$0300` | marker port (instrument) |
| `$0400-$04FF` | **ACTB** - biased activation page, page-aligned, the `adc ACTB,y` target |
| `$0500-$057F` | OUTB - signed matmul output, up to `F = 128` entries |
| `$0580-$05BF` | XVEC - the residual stream |
| `$05C0-$05FF` | Q4HI - `(q & 15) << 4` |
| `$0600-$063F` | ATTV - attention output |
| `$0640-$06FF` | free under the attention kernels; ACC8 / AVL / AVH in the legacy long-context path |
| `$0700-$07FF` | SCORL, SCORH, EXPE - `T` bytes each, laid end to end |

Every base is chosen so that `base_lo + max_index <= 255`: **no indexed access
in this map crosses a page.** That is a cycle per access, not a correctness
issue, but it is measurable and it was avoidable.

The attention rewrite deleted `ACC8`, `AVL` and `AVH` outright - the new AV
kernel keeps its accumulator in `A` and its 16-bit total in zero page, so the
three 64-byte block-accumulator arrays have no reader. They are still declared
in the legacy path that the long-context builds fall back to, which is why the
map above leaves `$0640-$06FF` for them rather than reusing it.

`$0400` is a hard constant, not a BSS symbol, and the BSS memory area is
capped at `$0200-$02FF` in the linker config so that a result slot growing
into `$0300` or `$0400` is a **link error**, not a silent collision. (A silent
collision of exactly this kind previously made banks read back ROM signatures
and looked like an emulator fault.) `P4HI` is deliberately a BSS symbol rather
than a constant so that the cap covers it too.

The battery-backed result block (`"ELYA"`, count, token ids) lives in the tail
of the **last** KV bank. At `T = 85` the cache leaves 128 bytes there, and the
`-DDEBUG` snapshot pages have nowhere to go, so that build refuses to assemble
rather than quietly overwriting the cache.

### The attention kernels and their two caches

QK sums over `d` for a fixed `t`; AV sums over `t` for a fixed `d`. Written
so the accumulator stays in `A`, the summation axis has to be a register
index and the *other* axis has to be unrolled into the instruction stream -
which means the two kernels want the cache laid out along opposite axes:

```
KT[d][l][t] = $6000 + d*64  + (l*20 + t)     RAM bank 4   (QK: y = l*20+t)
V [t][l][d] = $6000 + t*256 + (l*64 + d)     RAM bank 6   (AV: y = d)
```

Both strides are chosen so the indexed load can never cross a page (`64 + 59`
and `128 + 63` both stay under 256), which is worth a whole cycle per
multiply-add - the same argument that page-aligns the ternary gather chains.

The multiply row `tbl_mul + (operand << 4)` is written into the *operand byte*
of the `adc`, so the kernels live in PRG-RAM bank 5, mapped over the
weight-stream window at `$8000` for the duration of attention (attention never
reads the weight stream) and copied there from the `$A000` bank at reset.
`RAMEXEC` in FINDINGS is the measurement that says MMC5 PRG-RAM at `$8000` is
writable, executable, and independent of the `$6000` window.

## 3. PRG-ROM layout

96 KB PRG in twelve 8 KB banks, MMC5 PRG mode 3.  A **mixture** build has a
computed layout instead - see "Mixture of experts" at the end of this section.

| window | reg | contents |
|---|---|---|
| `$6000-$7FFF` | `$5113 = 4` | PRG-RAM: transposed **key** cache (bank 4) |
| `$6000-$7FFF` | `$5113 = 6` | PRG-RAM: **value** cache (bank 6) |
| `$8000-$9FFF` | `$5114` | **weight stream window**, slides 0..N; during attention `$5114 = 5` maps PRG-RAM bank 5 here, holding the self-modified kernels |
| `$A000-$BFFF` | `$5115` | embedding **and** positional table while both fit one bank (they do at `T = 20`); otherwise one each, switched once per token |
| `$C000-$DFFF` | `$5116` | row headers + lookup tables (fixed) |
| `$E000-$FFFF` | `$5117` | **fixed code bank**: kernel, page chains, tables, vectors |

The embedding is `V*D = 4,096` bytes and the positional table is `T*D`, which
at `T = 85` is 5,440. Together that is 9,536 > 8,192, so they no longer share
a bank: `embed_pos` reads the embedding row, switches `$5115`, and adds the
positional row in place. Two bank writes per token, 12 cycles, against ~1.3
million. One spare bank keeps the image a whole number of 16 KB iNES units.

### Weight stream format

The stream is written once, read strictly **sequentially forward, exactly
once per token, in the same order every token**. Per matrix row:

```
byte  n_pos          real count of +1 weights
byte  n_neg          real count of -1 weights
n_pos bytes          input indices with weight +1
n_neg bytes          input indices with weight -1
```

Indices are one byte because the largest input dimension is `F = 128`.

Rows are emitted in fixed order: for each layer, `Wq, Wk, Wv, Wo, W1, W2`;
then the output head. `3 x (4x64 + 128 + 64) + 64 = 1,408` rows per token.

Expected size at 50% zeros (`P(0)=0.5, P(+1)=P(-1)=0.25`):

```
index bytes   0.5 x 102,400 = 51,200
row headers   4 x 1,408     =  5,632   (n_pos, n_neg, d7 lo, d7 hi)
                              ------
stream         51,200 bytes = 6.25 banks of 8 KB; the image is padded to 7
```

### Row-to-bank mapping and BANK CROSSINGS PER TOKEN

This is the number that decided the Genesis and N64 outcomes.

The packer emits rows contiguously and, when the next row would straddle an
8 KB bank boundary, pads to the boundary and starts the row in the next bank.
So **no row ever straddles a bank**, and the reader only ever needs to switch
when it steps from one bank to the next, in order.

```
bank crossings per token = ceil(stream_bytes / 8192) - 1  ~= 6
switch cost              = 73 cycles  (MEASURED IN THE LOOP, see below)
total                    ~= 511 cycles per token
```

**Correction, from the MoE work.** This section used to say the switch cost
was "6 cycles (measured, MMC5)" for a total of ~42 cycles a token. Six cycles
is the `sta $5114` alone. What the loop actually pays is the whole sentinel
body - bump the bank, rebuild the OR mask, store it, reset the self-modified
chain pointer, step the header pointer, re-read the displaced header row - and
a `-DBANKPROF` build brackets that body in situ and reports **71 cycles**
(increment sentinel) or **73** (absolute sentinel), identically on every one
of 114 and 247 samples respectively. The conclusion does not change: at 73
cycles and 7 switches this is 511 cycles of 1,117,063, i.e. **0.046%**. But
the old figure was wrong by 12x and is corrected here rather than left
standing.

Against a measured 1,117,063 cycles per token that is **0.046%**. Bank
switching is a non-issue for this port, and
the reason is structural, not lucky: the access pattern is a single forward
sweep, never a random walk. (On MMC1 the same 7 switches would cost 210
cycles - still nothing. The mapper choice matters for *code layout*, not for
throughput, and MMC5 is chosen because 6 < 30 and because it gives three
independent windows plus 32 KB of banked PRG-RAM.)

### Mixture of experts: a computed bank map

A mixture build keeps `Wq/Wk/Wv/Wo`, the embedding, the positional table and
the head shared, and gives `W1`/`W2` of every layer N copies.  The weights
STREAMED per token stay at 102,400; the weights ON THE CARTRIDGE become
`53,248 + N x 49,152`.

The bank map is computed by `host/ref.py` and written out as three generated
artifacts - `moe.inc` (assembler constants), `moebanks.inc` (the `.incbin`
lines) and `nnmoe.cfg` (the linker config) - because a bank map maintained by
hand in three places is how a ROM comes to run perfectly and say the wrong
thing.

```
bank 0 .. S-1                shared stream (attention rows + head)
bank S + e*K .. +K-1         expert e's feed-forward rows
bank S + N*K                 embedding (+ positions)
bank S + N*K + 1             positional table
bank S + N*K + 2 + e         expert e's HEADER TABLE + a copy of the lookup
                             tables at $D700 (asserted identical addresses)
last bank                    code, chains, vectors
```

Two things make this work at all:

* **The sentinel carries an absolute bank number.**  `$FF, bank, 0, 0`.  The
  old "increment" form cannot express a walk that leaves the shared region for
  an expert's banks and comes back three times a token.
* **Every region chunk starts at a bank boundary.**  The ROM's stream offset
  is implicit - `chain_reset` puts the gather index back to 0 on every switch -
  so the only place a walk can resume is the start of a bank.  This wastes
  part of a bank per chunk and buys a sentinel with no resume offset in it.

The router is the whole of the routing machinery:

```
    ldx curtok            ; 2
    lda routebank,x       ; 4     which expert's header bank
    sta MMC5_PRGC000      ; 4
```

Measured costs: 13 bank switches a token instead of 7, at 73 cycles, plus the
10-cycle router = **448 cycles a token, 0.040%**.  The switch count is 13 at
N = 4, 8 and 16 alike.  The ceiling is **N = 16** (122 of the MMC5's 128
banks, 839,680 ternary weights); N = 17 needs 129 and the packer refuses.

### The page chains

`ldy <stream>,x` needs an assembled absolute base. The `$8000` window is 8 KB
= 32 pages, so the fixed bank contains **32 specialised gather chains**, one
per page `$8000, $8100, ... $9F00`:

```
chain_p:
    ldy $p00+0,x   ; 4
    adc ACTB,y     ; 4
    ldy $p00+1,x
    adc ACTB,y
    ... 16 entries ...
    jmp gather_fold
```

16 entries x 6 bytes + 3 = 99 bytes per chain, 32 chains = 3,168 bytes, plus a
64-byte pointer table. A row's list is walked by pointing `chain_ptr` at the
chain for the row's page and putting the row's low byte in X. Lists shorter
than a multiple of 16 enter the chain part-way (Duff's device) at
`chain + (16-r)*6`; nothing is padded, so **no padding entry is ever
executed** - padding would have cost 8 cycles each and been the dominant
overhead.

`ldy $p00+k,x` with `k <= 15` and `x <= 255` can reach into the next page.
That is *correct* - the window is contiguous - and only costs the measured
page-cross +1. Rows do not straddle **banks**; they may straddle **pages**.

## 4. Predicted cost model (to be confirmed by measurement)

Per MAC in the ternary kernel:

| | cycles |
|---|---|
| `ldy chain,x` + `adc ACTB,y` | 8 |
| fold + X advance + loop, amortised over 16 | ~1.9 |
| **predicted per MAC** | **~10** |

Per token:

| stage | count | cycles each | total |
|---|---|---|---|
| ternary gathers | 51,200 | ~10 | ~512,000 |
| output requantise (16-bit -> 4-bit) | 1,408 | ~40 | ~56,000 |
| attention MACs (QK + AV, full context, `T = 20`) | 7,680 | ~23 | ~177,000 |
| softmax / residual / misc | - | - | ~40,000 |
| bank switches | 6 | 6 | 42 |
| **predicted total** | | | **~810,000** |

= **~0.45 s per token** at 1,789,772 Hz.

The prior run's extrapolation was ~0.8 s/token for 114,688 active weights,
i.e. ~12.2 cycles per weight *including zeros*. The prediction here is lower
because sign separation means a zero weight costs **nothing at all** - it is
simply absent from both index lists - so the cost scales with **nnz**, not
with the weight count. Whether that survives contact with a real measurement
is exactly what the kernel measurement is for.

**How that scales with `T`.** Only the attention row moves, and it moves as
`T^2`: `L*H*(p+1)*DH*2` MACs at position `p`, so `7,680` at `T = 20`'s last
position and `32,640` at `T = 85`'s. Everything else in the table is constant.
Since the attention term is the only one that grows, the *mean* cost per token
grows much less than the peak does - the mean sees `(p+1)` averaged over the
run, i.e. about half of full context either way. The measured figures are in
FINDINGS; this paragraph is the prediction they are checked against.

## 5. Exactness

Everything is integer and the host reference in `host/ref.py` is the
specification. Both sides implement, bit for bit:

- activations: signed 4-bit, `-7..+7`, one per byte, biased by +7 in ACTB
- ternary matmul: `acc16 = (sum_pos - 7*n_pos) - (sum_neg - 7*n_neg)`
- requantise 16-bit to 4-bit: `q = acc >> k` (arithmetic, floor) clamped so
  that `acc > 8*2^k - 1 -> +7` and `acc < -7*2^k -> -7`
- attention products via a 256-byte ROM table `mul[(nq<<4)|nk] = (q*k) >> 2`
  (measured to be exactly `floor(q*k/4)` over the whole reachable -7..7 range;
  the table's saturation clamp never fires)
- softmax: max-shift, 15-entry exp table, power-of-two normalisation, clamp
  to `0..7`. This is a *quantised* softmax, stated exactly, not an
  approximation of a float one
- residual: 4-bit add then clamp to `-7..+7`
- sampling: argmax over the 64 logits, ties to the lowest index
  (deterministic, so the ROM and the host must agree on the token id at every
  position)

**Requantise shifts.** `K_SHIFT`, `W2_SHIFT`, `AV_SHIFT` and `SM_SHIFT` are
now generated by `host/ref.py` into `out/model/shifts.inc`, which `rom/nn.s`
includes, so the specification and the kernel cannot hold different values.
`AV_SHIFT` was 4 in the first cut and is now **2**: the attention accumulator
is provably bounded by 14, so a shift of 4 left it with two reachable output
levels, and the provably-correct shift of 1 measured *worst* of the ladder.
See FINDINGS. (This paragraph said "is now 1" until 2026-08-08; the shipped
value has been 2 since the ladder was run.) The context length `T` is stamped
into the npz the same way and for the same reason.

**What the quantised softmax can express, and why it bounds long context.**
The normalisation picks the smallest `kk` with `S >> kk <= 8` and then clamps
each entry to `0..7`, so

```
sum_t p_t <= 8    with every p_t an integer in 0..7
```

**at most 8 of the T positions can carry any attention weight at all**, and
that ceiling does not move when `T` does. Extending the window extends what
the model may look at, not how much it may look at. This is a property of the
kernel, stated here so that a context result can be read against it rather
than around it.

The bar is a **bit-exact match of every generated token id over 16+ tokens**,
not a 3-token spot check - a 3-token check passed two genuinely broken changes
elsewhere in this project and only failed at positions 5 and 12.
