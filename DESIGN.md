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
| vocab `V` | 128 |
| d_model `D` | 64 |
| layers `L` | 3 |
| heads `H` / d_head | 2 / 32 |
| d_ff `F` | 128 |
| context `T` | 20 |

Ternary weight count:

```
per layer  Wq,Wk,Wv,Wo  4 x 64x64  = 16,384
           W1 (F x D)   128x64     =  8,192
           W2 (D x F)   64x128     =  8,192
                                    -------
                                     32,768
x 3 layers                          = 98,304
output head (V x D) 128x64          =  8,192
                                    -------
TOTAL ternary weights                106,496
```

(The prior run's extrapolation was quoted for 114,688 active weights; this
shape is within 7% of that, so the two are directly comparable.)

### The $6000 window arithmetic

KV cache is `L x T x 2 x D` bytes, one byte per 4-bit activation:

```
3 x 20 x 2 x 64 = 7,680 bytes
```

That is **93.75% of one 8 KB PRG-RAM bank**, and it is why `T = 20` rather
than 24 (24 would need 9,216 > 8,192 and split the cache across banks in the
middle of the attention loop).

Measured PRG-RAM is **32 KB in 4 banks at `$5113 = 4,5,6,7`** (declaring more
in the iNES header still yields 4 - reconfirmed this run). So the whole 32 KB
would support

```
T_max = 32768 / (L x 2 x D) = 32768 / 384 = 85 positions
```

before touching system RAM. The port uses one bank; the other three are
headroom.

**KV row alignment.** Rows are `D = 64` bytes and are laid out at
`$6000 + (l*T*2 + t*2 + kv) * 64`, i.e. every row is **64-byte aligned**. The
attention inner loop reads `(kptr),y` with `y < 64`, and a 64-byte-aligned
base plus `y < 64` **cannot cross a page**, so the measured `$6000` page-cross
+1 never fires. Full 256-byte alignment would give the same guarantee at 4x
the memory cost, so 64-byte alignment is used deliberately.

### System RAM map (2 KB)

| range | use |
|---|---|
| `$0000-$00FF` | zero page: stream pointers, counters, 16-bit accumulators |
| `$0100-$01FF` | stack |
| `$0200-$02FF` | BSS: results, logits scratch (linker-capped, see below) |
| `$0300` | marker port (instrument) |
| `$0400-$04FF` | **ACTB** - biased activation page, page-aligned, the `adc ACTB,y` target |
| `$0500-$05FF` | output/staging page (pre-bias), page-aligned |
| `$0600-$06FF` | q / k / v / attention scratch |
| `$0700-$07FF` | attention probabilities, misc |

`$0400` is a hard constant, not a BSS symbol, and the BSS memory area is
capped at `$0200-$02FF` in the linker config so that a result slot growing
into `$0300` or `$0400` is a **link error**, not a silent collision. (A silent
collision of exactly this kind previously made banks read back ROM signatures
and looked like an emulator fault.)

## 3. PRG-ROM layout

256 KB PRG = 32 banks of 8 KB, MMC5 PRG mode 3.

| window | reg | contents |
|---|---|---|
| `$6000-$7FFF` | `$5113 = 4` | PRG-RAM: KV cache |
| `$8000-$9FFF` | `$5114` | **weight stream window**, slides 0..N |
| `$A000-$BFFF` | `$5115` | embedding + positional table (fixed) |
| `$C000-$DFFF` | `$5116` | spare (fixed) |
| `$E000-$FFFF` | `$5117` | **fixed code bank**: kernel, page chains, tables, vectors |

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
then the output head. 1,472 rows per token.

Expected size at 50% zeros (`P(0)=0.5, P(+1)=P(-1)=0.25`):

```
index bytes   0.5 x 106,496 = 53,248
row headers   2 x 1,472     =  2,944
                              ------
                              56,192 bytes  = 6.86 banks of 8 KB
```

### Row-to-bank mapping and BANK CROSSINGS PER TOKEN

This is the number that decided the Genesis and N64 outcomes.

The packer emits rows contiguously and, when the next row would straddle an
8 KB bank boundary, pads to the boundary and starts the row in the next bank.
So **no row ever straddles a bank**, and the reader only ever needs to switch
when it steps from one bank to the next, in order.

```
bank crossings per token = ceil(stream_bytes / 8192) - 1  ~= 6
switch cost              = 6 cycles  (measured, MMC5)
total                    ~= 42 cycles per token
```

Against a predicted ~800,000 cycles per token that is **0.005%**, or about
**0.0008 cycles per MAC**. Bank switching is a non-issue for this port, and
the reason is structural, not lucky: the access pattern is a single forward
sweep, never a random walk. (On MMC1 the same 7 switches would cost 210
cycles - still nothing. The mapper choice matters for *code layout*, not for
throughput, and MMC5 is chosen because 6 < 30 and because it gives three
independent windows plus 32 KB of banked PRG-RAM.)

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
| ternary gathers | 53,248 | ~10 | ~532,000 |
| output requantise (16-bit -> 4-bit) | 1,472 | ~40 | ~59,000 |
| attention MACs (QK + AV, full context) | 7,680 | ~23 | ~177,000 |
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

## 5. Exactness

Everything is integer and the host reference in `host/ref.py` is the
specification. Both sides implement, bit for bit:

- activations: signed 4-bit, `-7..+7`, one per byte, biased by +7 in ACTB
- ternary matmul: `acc16 = (sum_pos - 7*n_pos) - (sum_neg - 7*n_neg)`
- requantise 16-bit to 4-bit: `q = acc >> k` (arithmetic, floor) clamped so
  that `acc > 8*2^k - 1 -> +7` and `acc < -7*2^k -> -7`
- attention products via a 256-byte ROM table `mul[(nq<<4)|nk] = (q*k) >> 2`
- softmax: max-shift, 15-entry exp table, power-of-two normalisation, clamp
  to `0..7`. This is a *quantised* softmax, stated exactly, not an
  approximation of a float one
- residual: 4-bit add then clamp to `-7..+7`
- sampling: argmax over the 128 logits (deterministic, so the ROM and the
  host must agree on the token id at every position)

The bar is a **bit-exact match of every generated token id over 16+ tokens**,
not a 3-token spot check - a 3-token check passed two genuinely broken changes
elsewhere in this project and only failed at positions 5 and 12.
