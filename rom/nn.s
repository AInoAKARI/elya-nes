; ---------------------------------------------------------------------------
; nn.s - ternary transformer forward pass on the RP2A03, MMC5, 80 KB PRG.
;
; The shape of the ternary inner loop is forced by the CPU (see DESIGN.md):
; the accumulator can only stay in A if BOTH operands are reached through an
; index register, which means the weight stream must be addressed absolutely.
; Hence the 32 page-specialised gather chains at the top of the fixed bank -
; one per 256-byte page of the $8000 weight window.
;
; Activations are stored biased (+7) so every partial sum is non-negative and
; a block of 16 can never set the carry (16*14 = 224 < 256).  That is what
; makes `ldy stream,x / adc actb,y` legal with no `clc` between elements.
;
; REGISTER CONTRACT: X holds the weight-stream offset for the whole of a
; forward pass and must survive every subroutine.  Anything that needs X saves
; it in `xsave` first.  Counts are passed in zero page, never in X.
; ---------------------------------------------------------------------------

.include "common.inc"
.export reset
.export res_tokens, res_ntok

; ---- model shape (must match host/ref.py) ---------------------------------
NVOCAB   = 64
NDMODEL  = 64
NLAYER   = 3
NHEAD    = 2
NDHEAD   = 32
NFF      = 128

; ---- context length -------------------------------------------------------
; NCTX is the only shape knob that varies, and host/ref.py must be given the
; SAME value through NES_T.
;
; The old interleaved KV cache was bounded only by the 32 KB of PRG-RAM the
; MMC5 windows at $6000, which allowed NCTX up to 85.  The attention kernels
; bound it far harder, and the bound is ADDRESSING, not capacity - see the
; ATTN_TMAX arithmetic below.  A build above the ceiling does not assemble.
.ifndef NCTX
NCTX     = 20
.endif
NTOKGEN  = NCTX - 1         ; >= 16, the verification bar

; ---- the context ceiling the attention kernels impose ---------------------
; Both kernels reach their cache with `ldx <build-time constant>, y`, which is
; what keeps the accumulator in A (see the RAMKERN comments).  That costs two
; hard limits:
;
;   QK   ldx KTBASE + d*64, y      y = curlay*NCTX + t
;        y is ONE BYTE and the d stride is 64, so the whole (layer, t) index
;        must fit in the 64-byte row:   NLAYER * NCTX <= 64  ->  NCTX <= 21
;
;   AV   ldx VBASE + t*256, y      y = curlay*64 + d
;        the base is an assembled absolute address inside the $6000 window,
;        so VBASE + (NCTX-1)*256 + 255 <= $7FFF  ->  NCTX <= 32
;
; The binding one is QK's.  Raising it means giving K a stride of NLAYER*NCTX
; instead of 64, which at NCTX = 85 makes the key cache 64 * 255 = 16,320
; bytes - two PRG-RAM banks - and the kernel's assembled absolute addresses
; cannot cross a bank.  See FINDINGS, "Why the two branches do not both fit".
ATTN_TMAX = 64 / NLAYER
FASTATTN  = NCTX <= ATTN_TMAX

; Above the ceiling the ROM falls back to the LEGACY attention path - the
; pointer-walking `dot_qk` / `acc_av` the kernels replaced, over the
; interleaved KV cache that spreads across all four PRG-RAM banks.  It is
; three to four times slower per attention MAC and it is the only way to build
; a context this cartridge cannot address with assembled constants.  It exists
; because the T = 85 context experiment has to remain reproducible; it is not
; what ships.  See FINDINGS, "Why the two branches do not both fit".
.if FASTATTN
    .assert (NCTX - 1) * 256 + 255 <= $1FFF, error, "NCTX above the AV kernel's ceiling: V[t] pages leave the $6000 window"
.endif
    .assert NTOKGEN >= 16, error, "NTOKGEN below the 16-token verification bar"

; seed token: the ROM free-runs from a single token, so this is the whole
; prompt.  Overridable from the build (-DSEEDTOK=nn) so the same weights can
; be checked from several starting points rather than one lucky one.
.ifndef SEEDTOK
SEEDTOK  = 1
.endif

; number of 8 KB banks the weight stream occupies.  7 is what a 50%-dense
; model needs; a denser trained model needs more, so this is a build define
; and rom/nn9.cfg is the matching linker config.  The fixed windows sit
; immediately after the stream banks, so their bank numbers derive from it.
.ifndef NSTREAM
NSTREAM  = 7
.endif
; The embedding is NVOCAB*NDMODEL = 4,096 bytes and the positional table is
; NCTX*NDMODEL.  While the two fit one 8 KB window together they share the
; $A000 bank and `embed_pos` never switches; when they do not, the positional
; table gets a bank of its own and embed_pos pays two bank writes per token.
; The bank NUMBERS do not move either way, so the linker config, the iNES
; header and the ROM image are the same size in both cases.
POSINEMB = (NVOCAB + NCTX) * NDMODEL <= $2000
EMBBANK  = $80 + NSTREAM          ; $A000 window: embedding (+ positions)
POSBANK  = $80 + NSTREAM + 1      ; $A000 window: positional table
TBLBANK  = $80 + NSTREAM + 2      ; $C000 window: row headers + tables
CODEBANK = $80 + NSTREAM + 4      ; $E000 window: code, chains, vectors
                                  ; ($80 + NSTREAM + 3 is the spare bank)

BIASV    = 7
BLOCKSZ  = 16
MULBIAS  = 13

;; requantise shifts come from host/ref.py via a generated include, so the
;; ROM and the specification cannot disagree about them.
.include "shifts.inc"
HI2 = (8 << KSHIFT) - 1
LO2 = <(-(7 << KSHIFT))
HI3 = (8 << W2SHIFT) - 1
LO3 = <(-(7 << W2SHIFT))
HI4 = (8 << AVSHIFT) - 1
LO4 = <(-(7 << AVSHIFT))

MMC5_PRGMODE = $5100
MMC5_RAMPRO1 = $5102
MMC5_RAMPRO2 = $5103
MMC5_RAMBANK = $5113
MMC5_PRG8000 = $5114
MMC5_PRGA000 = $5115
MMC5_PRGC000 = $5116
MMC5_PRGE000 = $5117

.macro PMARK v
.ifdef PROFILE
    stx xsave2              ; 3
    ldx #v                  ; 2
    stx MARKER              ; 4
    ldx xsave2              ; 3   = 12 cycles, A/Y/carry untouched
.endif
.endmacro
PMARK_COST = 12

; Same shape, gated separately, so the attention breakdown can be measured
; without perturbing the PROFILE build the headline numbers come from.
.macro AMARK v
.ifdef ATTNPROF
    stx xsave2              ; 3
    ldx #v                  ; 2
    stx MARKER              ; 4
    ldx xsave2              ; 3   = 12 cycles, A/Y/carry untouched
.endif
.endmacro

; ---- fixed pages in system RAM (constants, NOT bss - see nn.cfg) ----------
; The per-position arrays (SCORL, SCORH, EXPE, P4HI) are NCTX entries each, so
; they are sized by NCTX rather than by the T = 20 constants the first cut
; used.  Three of them fit page 7 alongside nothing else; P4HI goes in BSS,
; where the linker's $0200-$02FF cap turns an overflow into a LINK ERROR
; rather than a silent collision with the marker port.  Every base below is
; chosen so that base_lo + max_index <= 255, i.e. NO indexed access here
; crosses a page.
ACTB   = $0400              ; biased activation input page (the adc target)
OUTB   = $0500              ; signed matmul output, up to 128 entries
XVEC   = $0580              ; residual stream x[0..63], signed
Q4HI   = $05C0              ; (q & 15) << 4
ATTV   = $0600              ; attention output, signed
.if .not FASTATTN
ACC8   = $0640              ; 8-bit block accumulators for AV   (legacy path)
AVL    = $0680              ; 16-bit AV accumulators             (legacy path)
AVH    = $06C0
.endif
SCORL  = $0700
SCORH  = SCORL + NCTX
EXPE   = SCORH + NCTX
    .assert EXPE + NCTX <= $0800, error, "score arrays overflow page 7"
    .assert <SCORH + NCTX - 1 < 256, error, "SCORH,y would cross a page"
    .assert <EXPE + NCTX - 1 < 256, error, "EXPE,y would cross a page"

EMBED   = $A000             ; EMBBANK
.if POSINEMB
POSTAB  = $A000 + NVOCAB * NDMODEL   ; same bank as the embedding
.else
POSTAB  = $A000                      ; POSBANK - a bank of its own
.endif
HEADERS = $C000
KVBASE  = $6000

; ---- self-modified kernels and the caches they read -----------------------
; KERNBANK is PRG-RAM mapped at $8000 (the weight-stream window, which
; attention never reads) and holds the kernels whose multiply-table operands
; are patched per layer/head.  VBANK is a SECOND PRG-RAM bank holding the
; value cache in the layout the AV kernel wants:
;
;   V[t][l][d] = VBASE + t*256 + l*64 + d
;
; The 256-byte t stride is what makes the AV kernel's `ldx VBASE+t*256,y`
; free of page crossings for every one of the 192 (l,d) offsets, and it makes
; the base a BUILD-TIME CONSTANT - the kernel never patches it.
; The key cache gets the MIRROR treatment: QK sums over d for a fixed t, so
; there d is the unrolled/patched axis and t is the register axis, and the
; cache has to be TRANSPOSED for the address to be linear in t:
;
;   KT[d][l][t] = KTBASE + d*64 + (l*NCTX + t)
;
; 64-byte d stride, index l*NCTX+t <= NLAYER*NCTX-1 <= 63, so
; `ldx KTBASE+d*64,y` never crosses a page either.  K and V therefore want
; OPPOSITE layouts, which is why they live in different PRG-RAM banks.
;
; That accounts for three of the four PRG-RAM banks: 4 = keys, 5 = kernels,
; 6 = values.  Bank 7 is free.  It is also why the T = 85 context experiment
; cannot be built on top of this: at NCTX = 85 the two caches alone need
; 3*85*2*64 = 32,640 bytes, all four banks, and there is no bank left to hold
; writable code - never mind the addressing ceilings above.
.if FASTATTN
KERNBANK = 5
VBANK    = 6
KVBANK   = 4
VBASE    = $6000
KTBASE   = $6000
KTBYTES  = NDMODEL * 64     ; the key cache is a fixed 4 KB whatever NCTX is
VBYTES   = NCTX * 256
    .assert VBYTES <= $2000, error, "value cache does not fit one PRG-RAM bank"

; "ELYA" magic + count + NTOKGEN token ids, parked in the tail of the KEY
; bank (bank 4, which the 4 KB key cache leaves half empty) so an emulator
; with no scripting hook can be cross-checked through its .sav file.
SAVBASE = $8000 - (5 + NTOKGEN)
    .assert SAVBASE >= KTBASE + KTBYTES, error, "sav block overlaps the key cache"

.else
; ---- legacy long-context path --------------------------------------------
; One interleaved cache, rows of NDMODEL bytes laid out [layer][k|v][t], the
; row index resolved at run time and the PRG-RAM bank switched per row.  This
; is what makes NCTX = 85 addressable at all: 8 KB / 64 = 128 rows per bank
; exactly, so a row never straddles a bank.
KVBANK0  = 4                ; $5113 = 4,5,6,7 are the four real PRG-RAM banks
KVROWS   = NLAYER * 2 * NCTX
KVBYTES  = KVROWS * NDMODEL
KVBANKS  = (KVBYTES + 8191) / 8192
KVLAST   = (KVBYTES - 1) / 8192
    .assert KVBYTES <= 32768, error, "KV cache exceeds the 32 KB of PRG-RAM"

; the sav block goes in the tail of the LAST bank the cache occupies.  At
; NCTX = 85 that leaves it exactly 128 bytes.
SAVBASE = $8000 - (5 + NTOKGEN)
KVLASTEND = $6000 + (KVBYTES - KVLAST * 8192)
    .assert SAVBASE >= KVLASTEND, error, "sav block overlaps the KV cache"
.endif

.zeropage
wchain:  .res 2
wep:     .res 2
wfold:   .res 2
rqp:     .res 2
hptr:    .res 2
hy:      .res 1
totL:    .res 1
totH:    .res 1
blkcnt:  .res 1
blkstep: .res 1
npos:    .res 1
nneg:    .res 1
wbank:   .res 1
rowcnt:  .res 1
di:      .res 1
sptr:    .res 2
kptr:    .res 2
xsave:   .res 1
xsave2:  .res 1
cnt:     .res 1
bacc:    .res 1
scL:     .res 1
scH:     .res 1
ph:      .res 1
hbase:   .res 1
hend:    .res 1
curpos:  .res 1
curlay:  .res 1
curtok:  .res 1
tcnt:    .res 1
gi:      .res 1
kvsel:   .res 1
qkp:     .res 2
ktoff:   .res 1
blkn:    .res 1
sumL:    .res 1
sumH:    .res 1
kk:      .res 1
bestL:   .res 1
bestH:   .res 1
besti:   .res 1
t0:      .res 1
t1:      .res 1
avp:     .res 2
nbL:     .res 1              ; -(MULBIAS * live count), pre-negated so the AV
nbH:     .res 1              ; loop can seed the accumulator instead of
                             ; subtracting afterwards.  NOT sumL/sumH, which
                             ; softmax overwrites between here and the AV pass
smp:     .res 2              ; the tbl_p row for this softmax's kk
.if SM_EXACTNORM
accL:    .res 1              ; exact normaliser: the 16-bit threshold (p+1)*S
accH:    .res 1
e8L:     .res 1              ; e * SM_TARGET, 16 bit
e8H:     .res 1
pcur:    .res 1              ; the probability being assigned
ei:      .res 1              ; index into tbl_exp
.endif
.ifdef ATTNBENCH
mulp:    .res 2              ; the no-SMC multiply-row pointer, bench only, and
.endif                       ; conditional so the bench cannot move any other
                             ; zero page variable in the shipping build
avn:     .res 1              ; number of positions with p != 0
avnt:    .res 1              ; curpos + 1
ucur:    .res 1              ; chain unit cursor, filled from the top down
.if .not FASTATTN
; legacy long-context path only.  Declared LAST so that the shipping build's
; zero page is byte-for-byte the one the attention work measured.
kvt:     .res 1              ; KV cache position index
rowL:    .res 1              ; KV row index, 16 bit: row = (l*2+kv)*NCTX + t
rowH:    .res 1
bnkc:    .res 1              ; PRG-RAM bank counter, reset only
.endif

.segment "BSS"
res_tokens: .res 96
res_ntok:   .res 1
P4HI:       .res NCTX       ; quantised softmax output, one nibble per position
.if SM_EXACTNORM
; e -> p<<4, rebuilt once per softmax by the exact normaliser.  65 entries
; because the exp table's largest value is 64.  In BSS rather than page 7:
; page 7 holds SCORL/SCORH/EXPE, which are NCTX entries each and fill it at
; NCTX = 85, and the linker cap on BSS turns an overflow here into a link
; error rather than a silent collision with ACTB.
PTBL:       .res 65
    .assert (PTBL & $FF) + 64 < 256, error, "PTBL,y would cross a page"
.endif
    .assert NTOKGEN <= 96, error, "res_tokens too small for NTOKGEN"
    .assert (P4HI & $FF) + NCTX - 1 < 256, error, "P4HI,y would cross a page"

; ===========================================================================
.segment "HEADER"
    .byte "NES", $1A
    .byte (NSTREAM + 5) / 2 ; 16 KB units: NSTREAM + emb + pos + tbl + spare
                            ; + code.  The spare bank exists only so that the
                            ; bank count stays even and the image is a whole
                            ; number of 16 KB iNES units.
    .byte 1
    .byte $52               ; mapper 5, battery
    .byte $00
    .byte 4                 ; 32 KB PRG-RAM (measured: 4 real banks)
    .byte 0, 0, 0, 0, 0, 0, 0

.segment "STREAM0"
    .incbin "out/model/stream.bin", $0000, $2000
.segment "STREAM1"
    .incbin "out/model/stream.bin", $2000, $2000
.segment "STREAM2"
    .incbin "out/model/stream.bin", $4000, $2000
.segment "STREAM3"
    .incbin "out/model/stream.bin", $6000, $2000
.segment "STREAM4"
    .incbin "out/model/stream.bin", $8000, $2000
.segment "STREAM5"
    .incbin "out/model/stream.bin", $A000, $2000
.segment "STREAM6"
    .incbin "out/model/stream.bin", $C000, $2000
.if NSTREAM > 7
.segment "STREAM7"
    .incbin "out/model/stream.bin", $E000, $2000
.segment "STREAM8"
    .incbin "out/model/stream.bin", $10000, $2000
.endif

.segment "EMBED"
    .incbin "out/model/embed.bin"
.if POSINEMB
    .incbin "out/model/pos.bin"     ; POSTAB = EMBED + NVOCAB*NDMODEL
.else
.segment "POS"
    .incbin "out/model/pos.bin"     ; POSTAB = $A000 in POSBANK
.endif

.segment "HEADERS"
    .incbin "out/model/headers.bin"

.segment "TABLES"
    .align $100
tbl_mul:    .incbin "out/model/tbl_mul.bin"
    .align $100
tbl_q2:     .incbin "out/model/tbl_q2.bin"
    .align $100
tbl_q3:     .incbin "out/model/tbl_q3.bin"
    .align $100
tbl_q4:     .incbin "out/model/tbl_q4.bin"
    .align $100
tbl_clamp:  .incbin "out/model/tbl_clamp.bin"
    .align $100
tbl_entoff: .incbin "out/model/tbl_entoff.bin"
    .align $100
tbl_blkcnt: .incbin "out/model/tbl_blkcnt.bin"
    .align $100
tbl_step:   .incbin "out/model/tbl_step.bin"
tbl_exp:    .incbin "out/model/tbl_exp.bin"
.if SM_EXACTNORM
tbl_e8lo:   .incbin "out/model/tbl_e8lo.bin"   ; e * SM_TARGET, 16 bit
tbl_e8hi:   .incbin "out/model/tbl_e8hi.bin"
tbl_p4:     .incbin "out/model/tbl_p4.bin"     ; p -> p<<4
.endif

.segment "PVTABLE"
; The AV product table.  Separate from tbl_mul because AV's high nibble is an
; unsigned probability in 0..PMAX and QK's is a signed activation in -7..7; at
; SM_TARGET = 8 the two agree on every row AV reads.  Page aligned for the
; same reason tbl_mul is: av_patch writes p<<4 into the operand's LOW byte and
; the high byte is assembled, so the row address costs one store.
tbl_pv:     .incbin "out/model/tbl_pv.bin"
    .assert PBLOCK * PVMAX <= 255, error, "AV block would set carry"
    .assert PMAX <= 15, error, "a probability nibble no longer fits p<<4"
    ; Both normalisers are implemented; SM_EXACTNORM selects which, and it
    ; comes from the same generated include the packer wrote, so the kernel
    ; and the specification cannot disagree about it.

; The softmax tables live in the $A000 bank, which is mapped throughout
; softmax; the $C000 table bank has no room left.
.segment "SMTABLES"
tbl_sm:     .incbin "out/model/tbl_sm.bin"     ; page aligned by the linker
tbl_p:      .incbin "out/model/tbl_p.bin"

.if FASTATTN
; ===========================================================================
; Self-modified kernels.  Assembled for $8000 (PRG-RAM bank 5) but STORED in
; the $A000 bank; `reset` copies them across.  See RAMEXEC in FINDINGS for the
; measurement that says MMC5 PRG-RAM at $8000 is writable and executable.
; ===========================================================================
.segment "RAMKERN"
.import __RAMKERN_LOAD__, __RAMKERN_RUN__, __RAMKERN_SIZE__

; --- AV: att[d] = quant(sum_t mul[(p_t<<4)|v_t[d]] - MULBIAS*(curpos+1)) ---
;
; The 6502 cannot keep an accumulator in A while ALSO building a multiply
; index, because `ora` has no X/Y form (see DESIGN.md).  The way out is to
; stop building the index: p_t is constant across d, so the multiply table
; ROW is constant across d too, and the row's address can live in the
; INSTRUCTION rather than in a register.  Then d is the outer loop, t is the
; inner one, A holds the accumulator for the whole of it, and each element is
;
;       ldx VBASE+t*256, y      ; 4   y = l*64 + d, never crosses a page
;       adc tbl_mul+(p_t<<4), x ; 4   low byte patched by av_patch
;
; Eight cycles per multiply-add against the 25 the ora/tax/lda/adc/sta form
; costs.  The chain is laid out in DESCENDING t so that entering it at unit
; (NCTX-1-curpos) covers exactly t = curpos..0; addition is commutative so
; the order does not change the sum.
;
; Carry: every table entry is 0..25 and the folds are at most ten units
; apart, so 10*25 = 250 < 256 - a block can never set carry, exactly the
; argument that makes the ternary gather chain legal.
avchain:
.repeat NCTX, u
    .ident (.sprintf ("avu%02d", u)):
    ldx VBASE, y                ; base (+1,+2) and multiply row (+4) are all
    adc tbl_pv, x               ; patched per unit; y is just d
    .if ((u .mod PBLOCK) = PBLOCK - 1) || (u = NCTX - 1)
    adc totL                    ; fold the block into the 16-bit total
    sta totL
    bcc *+4
    inc totH
    lda #0
    clc
    .endif
.endrepeat
avu_none:                       ; entry used when every p is zero
    rts

; --- QK: score_t = sum_d mul[(q_d<<4)|k_t[d]] - MULBIAS*NDHEAD -------------
;
; The mirror image of the AV kernel.  Here it is q_d that is constant across
; the loop variable (t), so the multiply row is patched per d and the unrolled
; axis is d.  One chain per head, because a head's 32 units address a fixed
; set of KT rows; `attn_head` points qkp at the right one.
;
; Carry: a block of 8 table entries is at most 8*25 = 200, so no `clc` is
; needed between elements and the fold is every eighth unit.
.repeat NHEAD, h
    .ident (.sprintf ("qkchain%d", h)):
    lda #<(-(MULBIAS * NDHEAD))   ; seed with the bias rather than subtract it
    sta scL                       ; afterwards: 14 cycles a call instead of 30
    lda #>(-(MULBIAS * NDHEAD))
    sta scH
    lda #0
    clc
    .repeat NDHEAD, i
    ldx KTBASE + (h * NDHEAD + i) * 64, y
    .ident (.sprintf ("qkop%02d", h * NDHEAD + i)) = * + 1
    adc tbl_mul, x
    .if (i & 7) = 7
    adc scL
    sta scL
    bcc *+4
    inc scH
    lda #0
    clc
    .endif
    .endrepeat
    rts
.endrepeat

RAMKERN_END:
.endif

; ===========================================================================
; 32 gather chains, one per page of the $8000 weight window.
; ===========================================================================
.segment "CHAINS"

; The gather pointer (wchain, X) represents (stream offset - BLOCKSZ), which
; is why every bank begins with BLOCKSZ bytes of padding - it keeps that
; quantity non-negative.  Carrying the -BLOCKSZ in the POINTER rather than in
; the instruction operand keeps each chain's base page-aligned; with the
; operand form the base low byte was $F0 and essentially every indexed load
; paid the measured page-cross +1, worth a whole cycle per MAC.
;
; Entry k reads stream offset (k - BLOCKSZ), so entering the chain at entry
; (BLOCKSZ - r) covers exactly the FIRST r bytes of the list once X has been
; pre-advanced to the offset one PAST the block.  Entering a chain of constant
; offsets at the top would have covered the LAST r bytes instead - that was a
; real bug, and it produced plausible-looking output that happened to give the
; right token at positions 0 and 1.
.macro GCHAIN pg
    .repeat BLOCKSZ, k
    ldy $8000 + pg * 256 + k, x
    adc ACTB, y
    .endrepeat
    jmp (wfold)
.endmacro

; 33 chains: pages $80..$9F plus one for the position exactly one past the end
; of the bank, which a list ending on the final byte legitimately addresses.
.repeat 33, p
    .ident (.sprintf ("gchain%02d", p)):
    GCHAIN p
.endrepeat

CHAIN_SIZE = gchain01 - gchain00
    .assert CHAIN_SIZE = BLOCKSZ * 6 + 3, error, "chain size changed"
    .assert <wep <> $FF, error, "wep would hit the JMP-indirect page bug"
    .assert <wchain <> $FF, error, "wchain would hit the JMP-indirect page bug"

; ===========================================================================
.segment "CODE"

reset:
    NES_INIT
    lda #$03
    sta MMC5_PRGMODE
    lda #$02
    sta MMC5_RAMPRO1
    lda #$01
    sta MMC5_RAMPRO2
    lda #4                  ; PRG-RAM bank 4 (first of the four real banks)
    sta MMC5_RAMBANK
    lda #$80
    sta MMC5_PRG8000
    lda #EMBBANK
    sta MMC5_PRGA000
    lda #TBLBANK
    sta MMC5_PRGC000
    lda #CODEBANK
    sta MMC5_PRGE000

    lda #0
    tax
@clr:
    sta $0000,x
    sta $0200,x             ; also clears PTBL, whose e = 0 entry is 0 for
                            ; every S and is therefore never rewritten
    sta $0400,x
    sta $0500,x
    sta $0600,x
    sta $0700,x
    inx
    bne @clr
.if FASTATTN
    ; clear the transposed key cache (KTBYTES = 4 KB, RAM bank 4).  Both
    ; clears cover EVERY byte the cache can occupy.  Clearing less would leave
    ; the untouched positions reading whatever the PRG-RAM powered up with;
    ; the emulator zero-fills it, so the bug would be invisible here and only
    ; appear on hardware, which is exactly the class of thing this repo is
    ; supposed to refuse to ship.
    ldx #0
@clrkv:
    .repeat KTBYTES / 256, pg
    sta KTBASE + pg * 256, x
    .endrepeat
    inx
    bne @clrkv

    ; clear the value cache in RAM bank 6 (VBYTES = NCTX pages)
    lda #VBANK
    sta MMC5_RAMBANK
    lda #0
    ldx #0
@clrv:
    .repeat VBYTES / 256, pg
    sta VBASE + pg * 256, x
    .endrepeat
    inx
    bne @clrv
    lda #KVBANK
    sta MMC5_RAMBANK

    ; copy the self-modified kernels into PRG-RAM bank 5 at $8000
    ; the copy is a plain linear 16-bit move, so the SOURCE may sit anywhere;
    ; only the destination has to be page aligned for the loop's `ldy #0`.
    .assert <__RAMKERN_RUN__ = 0, lderror, "RAMKERN run is not page aligned"
    .assert __RAMKERN_RUN__ = $8000, lderror, "RAMKERN must run at $8000"
    lda #KERNBANK
    sta MMC5_PRG8000
    lda #<__RAMKERN_LOAD__
    sta sptr
    lda #>__RAMKERN_LOAD__
    sta sptr+1
    lda #<__RAMKERN_RUN__
    sta kptr
    lda #>__RAMKERN_RUN__
    sta kptr+1
    ldx #0
@kpg:
    cpx #>(__RAMKERN_SIZE__ + 255)      ; whole pages, rounded up
    beq @kdone
    ldy #0
@kb:
    lda (sptr),y
    sta (kptr),y
    iny
    bne @kb
    inc sptr+1
    inc kptr+1
    inx
    bne @kpg
@kdone:
    lda #$80
    sta MMC5_PRG8000
.else
    ; clear the KV cache: every byte of every bank it occupies.  At NCTX = 20
    ; that is one bank; at NCTX = 85 it is four.  Clearing only the first would
    ; leave positions 43..84 reading whatever the PRG-RAM powered up with, and
    ; the emulator zero-fills it, so the bug would be invisible here and only
    ; appear on hardware - which is precisely the kind of thing this repo is
    ; supposed to refuse to ship.
    lda #0
    sta bnkc
@clrbank:
    lda bnkc
    clc
    adc #KVBANK0
    sta MMC5_RAMBANK
    lda #<KVBASE
    sta kptr
    lda #>KVBASE
    sta kptr+1
    ldx #32                 ; 32 pages of 256 bytes = one 8 KB bank
    lda #0
@clrpage:
    ldy #0
@clrb:
    sta (kptr),y
    iny
    bne @clrb
    inc kptr+1
    dex
    bne @clrpage
    inc bnkc
    lda bnkc
    cmp #KVBANKS
    bcc @clrbank
    lda #KVBANK0
    sta MMC5_RAMBANK
.endif

    lda #SEEDTOK
    sta curtok
    lda #0
    sta curpos

    ldx #254
    stx MARKER              ; SYNC
.if FASTATTN
.ifdef ATTNBENCH
    jmp attn_bench
.endif
.ifdef RAMEXEC
    jmp ramexec_test
.endif
.endif
.ifdef BENCH
    jmp bench
.endif

@loop:
    MARKX M_BEGIN
    jsr forward
    MARKX M_END
    ldx curpos
    lda curtok
    sta res_tokens,x
    inc curpos
    lda curpos
    cmp #NTOKGEN
    bcc @loop

    lda #NTOKGEN
    sta res_ntok
    ; also park the result in battery-backed PRG-RAM, so an emulator with no
    ; scripting hook (ares) can be cross-checked through its .sav file.
    ; With the attention kernels the block lives in the key bank, which is
    ; already selected here.  On the legacy path the bank last selected is
    ; whichever KV row was touched last, so it has to be named.
.if .not FASTATTN
    lda #KVBANK0 + KVLAST   ; the sav block lives in the LAST KV bank
    sta MMC5_RAMBANK
.endif
    ldx #0
@sav:
    lda savmagic,x
    sta SAVBASE,x
    inx
    cpx #4
    bne @sav
    lda #NTOKGEN
    sta SAVBASE + 4
    ldx #0
@sav2:
    lda res_tokens,x
    sta SAVBASE + 5,x
    inx
    cpx #NTOKGEN
    bne @sav2
    ldx #M_DONE
    stx MARKER
@hang:
    jmp @hang

.ifdef DEBUG
; --- debug: snapshot XVEC for one chosen position into the unused tail of
; --- the KV PRG-RAM bank.  Slot 0 = x after embed+pos, slots 1..3 = x after
; --- layers 0..2.  Compared against host/ref.py's per-position trace.
.ifndef DBGPOS
DBGPOS = 2
.endif
; The snapshots park 32-byte slots at $7F80/$7FA0/$7FC0 for the three
; per-position arrays, so they need NCTX <= 32, and the whole block has to
; stay clear of both the key cache below it and the sav block above it.
; Refusing to assemble is the honest outcome; silently overwriting either
; would produce a ROM that disagrees with the host and blames the wrong thing.
    .assert NCTX <= 32, error, "DEBUG snapshot slots are 32 bytes per array"
    .assert $7FC0 + NCTX <= SAVBASE, error, "DEBUG snapshots collide with the sav block"
.if FASTATTN
    .assert $7E00 >= KTBASE + KTBYTES, error, "DEBUG snapshots collide with the key cache"
.else
    .assert $7E00 >= KVLASTEND, error, "DEBUG snapshots collide with the KV cache"
.endif
dbg_lo: .byte $00, $40, $80, $C0
dbg_hi: .byte $7E, $7E, $7E, $7E

; snapshot the attention internals for DBGPOS / layer 0
dbg_dump_attn:
    lda curpos
    cmp #DBGPOS
    beq :+
    rts
:   lda curlay
    beq :+
    rts
:   stx xsave
    ldy #0
@l:
    lda ATTV,y
    sta $7F00,y
    lda Q4HI,y
    sta $7F40,y
    iny
    cpy #64
    bne @l
    ldy #0
@l2:
    lda SCORL,y
    sta $7F80,y
    lda SCORH,y
    sta $7FA0,y
    lda P4HI,y
    sta $7FC0,y
    iny
    cpy #NCTX
    bne @l2
    ldx xsave
    rts

dbg_dump_x:                     ; A = slot
    ldy curpos
    cpy #DBGPOS
    beq :+
    rts
:   stx xsave
    tay
    lda dbg_lo,y
    sta sptr
    lda dbg_hi,y
    sta sptr+1
    ldy #0
@l:
    lda XVEC,y
    sta (sptr),y
    iny
    cpy #64
    bne @l
    ldx xsave
    rts
.endif

.ifdef BENCH
; --- kernel micro-benchmark ------------------------------------------------
; Runs the REAL `gather` (not a copy of it) over synthetic lists of a range of
; lengths, so the per-MAC slope and the per-list intercept can be separated
; and compared against the 8-cycle sign-separated gather primitive.
bench_lens:
    .byte 1, 8, 16, 17, 32, 64, 96, 128
BENCH_N = 8

bench:
    lda #0
    sta di
@l:
    jsr chain_reset         ; X = 0, wchain = gchain00
    lda #0
    sta totL
    sta totH
    lda #<fold_add
    sta wfold
    lda #>fold_add
    sta wfold+1
    ldy di
    lda bench_lens,y
    sta t1
    MARKX M_BEGIN
    lda t1
    jsr gather
    MARKX M_END
    inc di
    lda di
    cmp #BENCH_N
    bcc @l
    ldx #M_DONE
    stx MARKER
@hang2:
    jmp @hang2
.endif


.if FASTATTN
.ifdef RAMEXEC
; --- can MMC5 PRG-RAM be mapped at $8000 AND executed from? ----------------
; The whole self-modifying-chain plan depends on the answer, so it is a
; measurement, not an assumption.  Reports three marker bytes:
;   111 = the $8000 window read back what we wrote to it (it is RAM)
;   112 = a subroutine assembled into that RAM actually executed
;   113 = the $6000 window (RAM bank 4) still holds its own data
ramexec_stub:
    .byte $A2, 112          ; ldx #112
    .byte $8E, $00, $03     ; stx MARKER
    .byte $60               ; rts
RAMEXEC_LEN = 6

ramexec_test:
    lda #$AA
    sta $6000               ; bank 4 marker byte, must survive
    lda #5
    sta MMC5_PRG8000        ; bit 7 clear -> PRG-RAM bank 5 at $8000
    ldx #0
@cp:
    lda ramexec_stub,x
    sta $8000,x
    inx
    cpx #RAMEXEC_LEN
    bne @cp
    lda $8000
    cmp #$A2
    bne @nr
    ldx #111
    stx MARKER
@nr:
    jsr $8000               ; should emit 112
    lda $6000
    cmp #$AA
    bne @nk
    ldx #113
    stx MARKER
@nk:
    ; are RAM banks 4,5,6,7 four INDEPENDENT 8 KB banks at $6000?
    ; write the bank number into each, then read them all back.
    ldy #4
@wr:
    sty MMC5_RAMBANK
    sty $6001
    iny
    cpy #8
    bne @wr
    ldy #4
@rd:
    sty MMC5_RAMBANK
    cpy $6001
    bne @bad
    iny
    cpy #8
    bne @rd
    ldx #114                ; 114 = banks 4..7 are independent
    stx MARKER
@bad:
    lda #4
    sta MMC5_RAMBANK
    ldx #M_DONE
    stx MARKER
@h: jmp @h
.endif
.endif


.if FASTATTN
.ifdef ATTNBENCH
; --- the NO-self-modifying-code alternative, for a head-to-head number -----
; The brief's other idea: keep the multiply row in a zero-page POINTER
; instead of in the instruction, which needs no writable code at all.
;   ldy VBASE+t*256, x    ; 4   x = l*64+d, base low byte is 0, no page cross
;   adc (mulp), y         ; 5   base low byte is p<<4 <= 112, y <= 15, no cross
; Nine cycles instead of eight.  Assembled into ROM precisely because it does
; not need to be patched; benched against the real chain below.
avptr_chain:
.repeat NCTX, u
    .ident (.sprintf ("avpu%02d", u)):
    ldy VBASE + (NCTX - 1 - u) * 256, x
    adc (mulp), y
    .if ((u .mod PBLOCK) = PBLOCK - 1) || (u = NCTX - 1)
    adc totL
    sta totL
    bcc *+4
    inc totH
    lda #0
    clc
    .endif
.endrepeat
avpu_none:
    rts

avpent_lo:
    .byte <avpu_none
.repeat NCTX, n
    .byte <(.ident (.sprintf ("avpu%02d", NCTX - 1 - n)))
.endrepeat
avpent_hi:
    .byte >avpu_none
.repeat NCTX, n
    .byte >(.ident (.sprintf ("avpu%02d", NCTX - 1 - n)))
.endrepeat

avptr_call:
    jmp (avp)

; --- the two forms this port REJECTED, benched so the rejection is measured -
;
; Both are ways of giving a probability more than four bits.  Neither is
; built into the shipping kernel; they exist here only so that "we rejected
; log-domain accumulation because it costs N cycles per multiply-add" is a
; measurement rather than an opinion.  The chains below are the exact
; INSTRUCTION SEQUENCES those designs need; the tables they index are stand-in
; page-aligned tables of the right shape, because the 6502's timing here is
; data independent - every address involved is proven page-cross free, so the
; cycle count of the form does not depend on what the tables contain.  Only
; the cycle count is being measured.
BENCH_ALT_N = 10

; (1) LOG-DOMAIN: p and v are added as logs and turned back into a linear
;     value before accumulating, which is the standard way to get a wide
;     dynamic range out of narrow tables.  It evicts the accumulator from A
;     on every element, which is the thing DESIGN.md says costs everything.
avlog_chain:
.repeat BENCH_ALT_N, u
    .ident (.sprintf ("avlu%02d", u)):
    ldx VBASE + (BENCH_ALT_N - 1 - u) * 256, y   ; 4  the value nibble
    lda tbl_q2, x                                ; 4  stands in for log|v|
    adc #$00                                     ; 2  + log p, patched
    tax                                          ; 2
    lda tbl_clamp, x                             ; 4  stands in for antilog
    clc                                          ; 2
    adc totL                                     ; 3
    sta totL                                     ; 3
    bcc *+4                                      ; 2
    inc totH                                     ; 5
.endrepeat
avlu_none:
    rts

; (2) WIDE: the probability keeps a full byte, so the product no longer fits
;     one table entry and every element needs a 16-bit add.  This is the
;     honest form of "just use more bits per probability".
avwide_chain:
.repeat BENCH_ALT_N, u
    .ident (.sprintf ("avwu%02d", u)):
    ldx VBASE + (BENCH_ALT_N - 1 - u) * 256, y   ; 4
    clc                                          ; 2
    lda tbl_q2, x                                ; 4  product low byte
    adc totL                                     ; 3
    sta totL                                     ; 3
    lda tbl_clamp, x                             ; 4  product high byte
    adc totH                                     ; 3
    sta totH                                     ; 3
.endrepeat
avwu_none:
    rts

avlent_lo:
    .byte <avlu_none
.repeat BENCH_ALT_N, n
    .byte <(.ident (.sprintf ("avlu%02d", BENCH_ALT_N - 1 - n)))
.endrepeat
avlent_hi:
    .byte >avlu_none
.repeat BENCH_ALT_N, n
    .byte >(.ident (.sprintf ("avlu%02d", BENCH_ALT_N - 1 - n)))
.endrepeat
avwent_lo:
    .byte <avwu_none
.repeat BENCH_ALT_N, n
    .byte <(.ident (.sprintf ("avwu%02d", BENCH_ALT_N - 1 - n)))
.endrepeat
avwent_hi:
    .byte >avwu_none
.repeat BENCH_ALT_N, n
    .byte >(.ident (.sprintf ("avwu%02d", BENCH_ALT_N - 1 - n)))
.endrepeat

; --- isolated slope/intercept for the attention kernels --------------------
; Calls the REAL chains (not copies) over t-counts 1..NCTX so the per-MAC
; slope and the per-call intercept separate, the way BENCH does for the
; ternary gather.  Cache contents are irrelevant to the cycle count: every
; address involved is proven page-cross free, so the timing is data
; independent.  The measured window is BENCH_REP calls plus BENCH_REP copies
; of a fixed 20-cycle driver, which cancels out of the slope.
BENCH_REP = 64

attn_bench:
    lda #KERNBANK
    sta MMC5_PRG8000
    lda #VBANK
    sta MMC5_RAMBANK
    ldy #0                      ; make every position live so the sweep can
    lda #$10                    ; reach all NCTX units
@fill:
    sta P4HI,y
    iny
    cpy #NCTX
    bne @fill
    lda #NCTX
    sta avnt
    jsr av_patch
    lda #0
    sta di
@l:
    ldy di
    iny                         ; entry table is indexed by the LIVE count
    lda avent_lo,y
    sta avp
    lda avent_hi,y
    sta avp+1
    lda #BENCH_REP
    sta cnt
    MARKX M_BEGIN
@r:
    lda #0                  ; 2
    sta totL                ; 3
    sta totH                ; 3
    ldy #0                  ; 2
    clc                     ; 2
    jsr av_call             ; 6 + chain
    dec cnt                 ; 5
    bne @r                  ; 3
    MARKX M_END
    inc di
    lda di
    cmp #NCTX
    bcc @l

    ; the no-SMC pointer form, same sweep, same driver
    lda #<tbl_pv
    sta mulp
    lda #>tbl_pv
    sta mulp+1
    lda #0
    sta di
@l2:
    ldy di
    iny
    lda avpent_lo,y
    sta avp
    lda avpent_hi,y
    sta avp+1
    lda #BENCH_REP
    sta cnt
    MARKX M_BEGIN
@r2:
    lda #0                  ; 2
    sta totL                ; 3
    sta totH                ; 3
    ldx #0                  ; 2
    clc                     ; 2
    jsr avptr_call          ; 6 + chain
    dec cnt                 ; 5
    bne @r2                 ; 3
    MARKX M_END
    inc di
    lda di
    cmp #NCTX
    bcc @l2

    ; the two rejected forms, same driver, so the slopes are comparable
    lda #0
    sta di
@l3:
    ldy di
    iny
    lda avlent_lo,y
    sta avp
    lda avlent_hi,y
    sta avp+1
    lda #BENCH_REP
    sta cnt
    MARKX M_BEGIN
@r3:
    lda #0                  ; 2
    sta totL                ; 3
    sta totH                ; 3
    ldy #0                  ; 2
    clc                     ; 2
    jsr av_call             ; 6 + chain
    dec cnt                 ; 5
    bne @r3                 ; 3
    MARKX M_END
    inc di
    lda di
    cmp #BENCH_ALT_N
    bcc @l3

    lda #0
    sta di
@l4:
    ldy di
    iny
    lda avwent_lo,y
    sta avp
    lda avwent_hi,y
    sta avp+1
    lda #BENCH_REP
    sta cnt
    MARKX M_BEGIN
@r4:
    lda #0                  ; 2
    sta totL                ; 3
    sta totH                ; 3
    ldy #0                  ; 2
    clc                     ; 2
    jsr av_call             ; 6 + chain
    dec cnt                 ; 5
    bne @r4                 ; 3
    MARKX M_END
    inc di
    lda di
    cmp #BENCH_ALT_N
    bcc @l4

    ; QK does a fixed NDHEAD MACs, so there is no slope to fit - only the
    ; per-call cost.  Driver here is 10 cycles (ldy/dec/bne).
    lda #KVBANK
    sta MMC5_RAMBANK
    lda #<qkchain0
    sta qkp
    lda #>qkchain0
    sta qkp+1
    lda #BENCH_REP
    sta cnt
    MARKX M_BEGIN
@q:
    ldy #0                  ; 2
    jsr qk_call             ; 6 + chain + rts
    dec cnt                 ; 5
    bne @q                  ; 3
    MARKX M_END

    ldx #M_DONE
    stx MARKER
@bh: jmp @bh
.endif
.endif

; ===========================================================================
; forward(): curtok, curpos -> curtok
; ===========================================================================
forward:
    jsr embed_pos
.ifdef DEBUG
    lda #0
    jsr dbg_dump_x
.endif

    lda #0
    sta wbank
    lda #$80
    sta MMC5_PRG8000
    jsr chain_reset         ; also sets X = 0
    lda #<HEADERS
    sta hptr
    lda #>HEADERS
    sta hptr+1
    lda #0
    sta hy
    sta curlay
@layer:
    jsr do_layer
.ifdef DEBUG
    lda curlay
    clc
    adc #1
    jsr dbg_dump_x
.endif
    inc curlay
    lda curlay
    cmp #NLAYER
    bcc @layer

    lda #<XVEC
    sta sptr
    lda #>XVEC
    sta sptr+1
    lda #NDMODEL
    sta cnt
    jsr bias_copy
    jsr head_argmax
    rts

; --- x = clamp(emb[tok] + pos[p]) ------------------------------------------
embed_pos:
.if POSINEMB
    ; both tables are in the $A000 bank, so both rows are addressable at once
    ; and no bank write is needed anywhere in here.
    lda curtok
    jsr row64_embed         ; sptr = EMBED + tok*64
    lda sptr
    sta kptr
    lda sptr+1
    sta kptr+1
    lda curpos
    jsr row64_pos           ; sptr = POSTAB + pos*64
    ldy #0
@l:
    lda (kptr),y
    clc
    adc (sptr),y
    tax
    lda tbl_clamp,x
    sta XVEC,y
    iny
    cpy #NDMODEL
    bne @l
    rts
.else
    ; the two tables are in different banks of the same $A000 window, so the
    ; embedding row is copied into XVEC first and the positional row is added
    ; in place.  Two bank writes per token, 12 cycles, plus the copy.
    lda #EMBBANK
    sta MMC5_PRGA000
    lda curtok
    jsr row64_embed         ; sptr = EMBED + tok*64
    ldy #0
@c:
    lda (sptr),y
    sta XVEC,y
    iny
    cpy #NDMODEL
    bne @c

    lda #POSBANK
    sta MMC5_PRGA000
    lda curpos
    jsr row64_pos           ; sptr = POSTAB + pos*64
    ldy #0
@l:
    lda XVEC,y
    clc
    adc (sptr),y
    tax
    lda tbl_clamp,x
    sta XVEC,y
    iny
    cpy #NDMODEL
    bne @l
    lda #EMBBANK
    sta MMC5_PRGA000
    rts
.endif

row64_embed:
    pha
    and #$03
    tay
    lda mul64lo,y
    sta sptr
    pla
    lsr a
    lsr a
    clc
    adc #>EMBED
    sta sptr+1
    rts

row64_pos:
    pha
    and #$03
    tay
    lda mul64lo,y
    sta sptr                ; <POSTAB is 0, so no low carry is possible
    pla
    lsr a
    lsr a
    clc                     ; lsr leaves C set - it MUST be cleared here
    adc #>POSTAB
    sta sptr+1
    rts

savmagic:
    .byte $45, $4C, $59, $41    ; "ELYA"

mul64lo:
    .byte 0, 64, 128, 192

.if FASTATTN
lay20:                          ; KT[d] row offset of layer l = l * NCTX
    .byte 0, NCTX, NCTX * 2
.else
; Legacy KV layout is [layer][k|v][t]: row = (curlay*2 + kvsel) * NCTX + kvt.
; The per-(layer, k/v) blocks are then CONTIGUOUS in t, so the attention loops
; walk rows in order and cross a PRG-RAM bank boundary at most once per loop
; instead of ping-ponging.  Row 0 of each block is tabulated because
; (l*2+kv)*NCTX exceeds a byte at NCTX = 85 (max 5*85 = 425).
kvbaseL:
    .repeat NLAYER * 2, i
    .byte <(i * NCTX)
    .endrepeat
kvbaseH:
    .repeat NLAYER * 2, i
    .byte >(i * NCTX)
    .endrepeat
.endif

; ===========================================================================
; one transformer layer
; ===========================================================================
do_layer:
    lda #<XVEC
    sta sptr
    lda #>XVEC
    sta sptr+1
    lda #NDMODEL
    sta cnt
    jsr bias_copy

    jsr set_rq2
    lda #NDMODEL
    sta rowcnt
    jsr matmul              ; Wq
    jsr post_q

    jsr set_rq2
    lda #NDMODEL
    sta rowcnt
    jsr matmul              ; Wk
    lda #0
    sta kvsel
    jsr post_kv

    jsr set_rq2
    lda #NDMODEL
    sta rowcnt
    jsr matmul              ; Wv
    lda #1
    sta kvsel
    jsr post_kv

    PMARK 30
    jsr attention
    PMARK 31
.ifdef DEBUG
    jsr dbg_dump_attn
.endif

    lda #<ATTV
    sta sptr
    lda #>ATTV
    sta sptr+1
    lda #NDMODEL
    sta cnt
    jsr bias_copy
    jsr set_rq2
    lda #NDMODEL
    sta rowcnt
    jsr matmul              ; Wo
    jsr post_residual

    jsr set_rq2
    lda #NFF
    sta rowcnt
    jsr matmul              ; W1
    jsr post_relu

    jsr set_rq3
    lda #NDMODEL
    sta rowcnt
    jsr matmul              ; W2
    jsr post_residual
    rts

set_rq2:
    lda #<requant_k2
    sta rqp
    lda #>requant_k2
    sta rqp+1
    rts
set_rq3:
    lda #<requant_k3
    sta rqp
    lda #>requant_k3
    sta rqp+1
    rts

; ===========================================================================
; matmul: rowcnt rows reading ACTB, writing signed bytes to OUTB
; ===========================================================================
matmul:
    lda #0
    sta di
@row:
    jsr read_header
    PMARK 20
    jsr gather_row
    PMARK 21
    jsr do_requant
    ldy di
    sta OUTB,y
    inc di
    dec rowcnt
    bne @row
    rts

do_requant:
    jmp (rqp)

read_header:
    ldy hy
    lda (hptr),y
    cmp #$FF
    bne @ok
    inc wbank
    lda wbank
    ora #$80
    sta MMC5_PRG8000
    jsr chain_reset         ; X = 0, chain back to page $80
    jsr hdr_advance
    ldy hy
    lda (hptr),y
@ok:
    sta npos
    iny
    lda (hptr),y
    sta nneg
    iny
    lda (hptr),y
    sta totL
    iny
    lda (hptr),y
    sta totH
hdr_advance:
    lda hy
    clc
    adc #4
    sta hy
    bne @done
    inc hptr+1
@done:
    rts

chain_reset:
    lda #<gchain00
    sta wchain
    lda #>gchain00
    sta wchain+1
    ldx #0
    rts

gather_row:
    lda #<fold_add
    sta wfold
    lda #>fold_add
    sta wfold+1
    lda npos
    jsr gather
    lda #<fold_sub
    sta wfold
    lda #>fold_sub
    sta wfold+1
    lda nneg
    jsr gather
    rts

; --- gather one list of length A ------------------------------------------
gather:
    tay                     ; Y = list length, and stays valid for the tables
    beq @empty
    lda tbl_blkcnt,y
    sta blkcnt
    lda tbl_entoff,y        ; (BLOCKSZ - first block length) * 6
    clc
    adc wchain
    sta wep
    lda wchain+1
    adc #0
    sta wep+1
    txa                     ; X += first block length, chain follows on a wrap
    clc
    adc tbl_step,y
    tax
    bcc @go
    jsr chain_bump          ; bumps wchain AND wep
@go:
    lda #0
    clc                     ; the chain's first adc must not inherit a carry
    jmp (wep)
@empty:
    rts

fold_add:
    clc
    adc totL
    sta totL
    bcc @1
    inc totH
@1:
    dec blkcnt
    beq gather_done
    txa
    clc
    adc #BLOCKSZ
    tax
    bcc @2
    jsr chain_next
@2:
    lda #0
    clc
    jmp (wchain)

fold_sub:
    sta t0
    sec
    lda totL
    sbc t0
    sta totL
    lda totH
    sbc #0
    sta totH
    dec blkcnt
    beq gather_done
    txa
    clc
    adc #BLOCKSZ
    tax
    bcc @2
    jsr chain_next
@2:
    lda #0
    clc
    jmp (wchain)

; `gather` entered the chain with JMP, not JSR, so this rts unwinds straight
; to gather_row - the return address gather's own jsr pushed is still there.
gather_done:
    rts

chain_next:                 ; wchain += CHAIN_SIZE
    lda wchain
    clc
    adc #CHAIN_SIZE
    sta wchain
    bcc @1
    inc wchain+1
@1:
    rts

chain_bump:                 ; wchain += CHAIN_SIZE and wep += CHAIN_SIZE
    jsr chain_next
    lda wep
    clc
    adc #CHAIN_SIZE
    sta wep
    bcc @1
    inc wep+1
@1:
    rts

; ===========================================================================
; requantise: totL/totH -> A = signed 4-bit.  Does not touch X.
; ===========================================================================
requant_k2:
    lda totH
    beq @lo
    cmp #$FF
    beq @hi
    lda totH
    bmi @satn
@satp:
    lda #7
    rts
@satn:
    lda #<(-7)
    rts
@lo:
    ldy totL
    cpy #HI2 + 1
    bcs @satp
    lda tbl_q2,y
    rts
@hi:
    ldy totL
    cpy #LO2
    bcc @satn
    lda tbl_q2,y
    rts

requant_k3:
    lda totH
    beq @lo
    cmp #$FF
    beq @hi
    lda totH
    bmi @satn
@satp:
    lda #7
    rts
@satn:
    lda #<(-7)
    rts
@lo:
    ldy totL
    cpy #HI3 + 1
    bcs @satp
    lda tbl_q3,y
    rts
@hi:
    ldy totL
    cpy #LO3
    bcc @satn
    lda tbl_q3,y
    rts

requant_k4:
    lda totH
    beq @lo
    cmp #$FF
    beq @hi
    lda totH
    bmi @satn
@satp:
    lda #7
    rts
@satn:
    lda #<(-7)
    rts
@lo:
    ldy totL
    cpy #HI4 + 1
    bcs @satp
    lda tbl_q4,y
    rts
@hi:
    ldy totL
    cpy #LO4
    bcc @satn
    lda tbl_q4,y
    rts

; ===========================================================================
; post-matmul passes.  All of these save and restore X.
; ===========================================================================
bias_copy:
    stx xsave
    ldy #0
@l:
    lda (sptr),y
    clc
    adc #BIASV
    sta ACTB,y
    iny
    cpy cnt
    bne @l
    ldx xsave
    rts

.if FASTATTN
; post_q both records the query nibbles and patches them straight into the QK
; chains' multiply-row operands.  tbl_mul is page aligned, so the operand's
; low byte IS (q<<4).  The chains live in the $8000 window, so the weight
; stream has to step aside for the duration; wbank puts it back.
post_q:
    stx xsave
    lda #KERNBANK
    sta MMC5_PRG8000
.repeat NDMODEL, d
    lda OUTB + d
    and #$0F
    asl a
    asl a
    asl a
    asl a
    sta Q4HI + d                ; kept for the DEBUG dump only
    sta .ident (.sprintf ("qkop%02d", d))
.endrepeat
    lda wbank
    ora #$80
    sta MMC5_PRG8000
    ldx xsave
    rts

post_kv:
    stx xsave
    lda kvsel
    beq @k
    jmp @v                      ; the unrolled scatter is far out of bne range
@k:
    ; K[curlay][curpos][d] -> KT[d][curlay][curpos], a stride-64 scatter.
    ; Unrolled because the destination page is then a build-time constant and
    ; `sta abs,x` costs 5 whatever X is: 11 cycles an element against the 19
    ; the pointer form cost.
    ldx curlay
    lda lay20,x
    clc
    adc curpos
    tax
.repeat NDMODEL, d
    lda OUTB + d
    and #$0F
    sta KTBASE + d * 64, x
.endrepeat
    ldx xsave
    rts

@v:                             ; V[curpos][curlay][d], RAM bank 6
    lda #VBANK
    sta MMC5_RAMBANK
    ldy curlay
    lda mul64lo,y
    sta kptr                    ; curlay * 64
    lda curpos
    clc
    adc #>VBASE
    sta kptr+1                  ; VBASE + curpos * 256
@go:
    ldy #0
@l:
    lda OUTB,y
    and #$0F
    sta (kptr),y
    iny
    cpy #NDMODEL
    bne @l
    lda #KVBANK
    sta MMC5_RAMBANK
    ldx xsave
    rts

.else
; ---- legacy long-context path --------------------------------------------
post_q:
    stx xsave
    ldy #0
@l:
    lda OUTB,y
    and #$0F
    asl a
    asl a
    asl a
    asl a
    sta Q4HI,y
    iny
    cpy #NDMODEL
    bne @l
    ldx xsave
    rts

post_kv:
    stx xsave
    lda curpos
    sta kvt
    jsr kv_ptr
    ldy #0
@l:
    lda OUTB,y
    and #$0F
    sta (kptr),y
    iny
    cpy #NDMODEL
    bne @l
    ldx xsave
    rts

; row  = (curlay*2 + kvsel) * NCTX + kvt          (0 .. NLAYER*2*NCTX - 1)
; bank = row >> 7,   kptr = KVBASE + (row & 127) * 64        (clobbers A/X/Y)
;
; 8 KB / 64 = 128 rows per bank exactly, so a row NEVER straddles a bank and,
; being 64-byte aligned, (kptr),y with y < 64 never crosses a page either -
; the same alignment argument the one-bank version relied on, unchanged.
; X is free here: every caller (post_kv, attn_head) has already parked the
; weight-stream offset in xsave.
kv_ptr:
    lda curlay
    asl a
    clc
    adc kvsel
    tay
    lda kvbaseL,y
    clc
    adc kvt
    sta rowL
    lda kvbaseH,y
    adc #0
    sta rowH
    lda rowL
    asl a                   ; C = bit 7 of the row index
    lda rowH
    rol a                   ; A = row >> 7
    clc
    adc #KVBANK0
    sta MMC5_RAMBANK
    lda rowL
    and #$03
    tay
    lda mul64lo,y
    sta kptr
    lda rowL
    and #$7F
    lsr a
    lsr a
    clc
    adc #>KVBASE
    sta kptr+1
    rts
.endif

post_residual:
    stx xsave
    ldy #0
@l:
    lda XVEC,y
    clc
    adc OUTB,y
    tax
    lda tbl_clamp,x
    sta XVEC,y
    clc
    adc #BIASV
    sta ACTB,y
    iny
    cpy #NDMODEL
    bne @l
    ldx xsave
    rts

post_relu:
    stx xsave
    ldy #0
@l:
    lda OUTB,y
    bpl @pos
    lda #0
@pos:
    clc
    adc #BIASV
    sta ACTB,y
    iny
    cpy #NFF
    bne @l
    ldx xsave
    rts

; ===========================================================================
; attention
; ===========================================================================
.if FASTATTN
attention:
    stx xsave
    lda curpos
    clc
    adc #1
    sta avnt                    ; number of cached positions, curpos + 1
    ; map the self-modified kernels over the weight-stream window
    lda #KERNBANK
    sta MMC5_PRG8000
    lda #0
    sta hbase
@head:
    lda hbase
    clc
    adc #NDHEAD
    sta hend
    jsr attn_head
    lda hend
    sta hbase
    cmp #NDMODEL
    bcc @head
    lda wbank                   ; put the weight stream back at $8000
    ora #$80
    sta MMC5_PRG8000
    ldx xsave
    rts

attn_head:
    ; ---- scores for t = 0..curpos ------------------------------------
    AMARK 34
    lda hbase                   ; hbase / NDHEAD -> this head's QK chain
    lsr a
    lsr a
    lsr a
    lsr a
    lsr a
    tay
    lda qkent_lo,y
    sta qkp
    lda qkent_hi,y
    sta qkp+1
    ldy curlay
    lda lay20,y
    sta ktoff                   ; y index into KT[d] = curlay*20 + t
    lda #0
    sta tcnt
@score:
    ldy ktoff
    jsr qk_call
    AMARK 33
    ldy tcnt
    lda scL
    sta SCORL,y
    lda scH
    sta SCORH,y
    inc ktoff
    inc tcnt
    lda tcnt
    cmp curpos
    beq @score
    bcc @score
    AMARK 35

    AMARK 36
    jsr softmax
    AMARK 37

    ; ---- AV ----------------------------------------------------------
    ; d is the OUTER loop and t the inner one, which is what lets the
    ; accumulator stay in A; see the avchain comment in the RAMKERN segment.
    AMARK 40
    jsr av_patch                ; pack the p != 0 positions into the chain
    lda #VBANK
    sta MMC5_RAMBANK
    ldy avn
    lda avent_lo,y              ; enter so the chain runs exactly avn units
    sta avp
    lda avent_hi,y
    sta avp+1
    lda #0                      ; nb = -(PVBIAS * avn); seeding the
    sta nbL                     ; accumulator with it is 14 cycles a dimension
    sta nbH                     ; cheaper than subtracting it afterwards
    beq @bdone
@bias:
    sec
    lda nbL
    sbc #PVBIAS
    sta nbL
    lda nbH
    sbc #0
    sta nbH
@bdone:
    dey
    bpl @bias
    ldy hbase
@d:
    lda nbL
    sta totL
    lda nbH
    sta totH
    sty t1                      ; requant clobbers Y
    lda #0
    clc                         ; A = 0, C = 0: the chain's contract
    jsr av_call
    AMARK 39
    ; --- requant_k4, inlined: the jsr/rts was 12 cycles a dimension --------
    lda totH
    beq @lo
    cmp #$FF
    beq @hi
    lda totH
    bmi @satn
@satp:
    lda #7
    bne @put                    ; 7 is never zero, so this is unconditional
@satn:
    lda #<(-7)
    bne @put
@lo:
    ldy totL
    cpy #HI4 + 1
    bcs @satp
    lda tbl_q4,y
    jmp @put
@hi:
    ldy totL
    cpy #LO4
    bcc @satn
    lda tbl_q4,y
@put:
    ldy t1
    sta ATTV,y
    iny
    cpy hend
    bne @d
    lda #KVBANK
    sta MMC5_RAMBANK
    AMARK 41
    rts

av_call:
    AMARK 38
    jmp (avp)

; --- build the AV chain for this head --------------------------------------
; mul[(0<<4)|v] is 13 for EVERY v, so a position whose softmax nibble is zero
; contributes nothing but MULBIAS.  Those positions are dropped from the chain
; entirely and paid for by shrinking the bias to MULBIAS * (live count) - the
; arithmetic is unchanged, the work is not.  Measured on the trained model,
; 86.9% of AV positions are pure bias like this.
;
; Live positions are packed into the TOP of the chain, so entering at unit
; (NCTX - avn) runs exactly them.  Each unit needs two bytes: the V page
; ($60+t, because the t stride is a whole page) and the multiply row (P4HI[t],
; because tbl_mul is page aligned).
av_patch:
    lda #0
    sta avn
    lda #NCTX - 1
    sta ucur
    ldy #0
@l:
    lda P4HI,y
    beq @skip
    sty t0
    ldx ucur
    ldy avuoff,x                ; Y = this unit's byte offset (ldx abs,x does
    sta avchain + 4, y          ; not exist, so the offset goes in Y and the
    ldx curlay                  ; stores are abs,y)
    lda mul64lo,x
    sta avchain + 1, y          ; base low  = curlay * 64
    lda t0
    clc
    adc #>VBASE
    sta avchain + 2, y          ; base high = VBASE_hi + t
    inc avn
    dec ucur
    ldy t0
@skip:
    iny
    cpy avnt
    bne @l
    rts

avuoff:
.repeat NCTX, u
    .byte <(.ident (.sprintf ("avu%02d", u)) - avchain)
.endrepeat

; entry point for a chain carrying n live units, n = 0..NCTX
avent_lo:
    .byte <avu_none
.repeat NCTX, n
    .byte <(.ident (.sprintf ("avu%02d", NCTX - 1 - n)))
.endrepeat
avent_hi:
    .byte >avu_none
.repeat NCTX, n
    .byte >(.ident (.sprintf ("avu%02d", NCTX - 1 - n)))
.endrepeat
    .assert <avp <> $FF, error, "avp would hit the JMP-indirect page bug"

qk_call:
    AMARK 32
    jmp (qkp)

qkent_lo:
.repeat NHEAD, h
    .byte <(.ident (.sprintf ("qkchain%d", h)))
.endrepeat
qkent_hi:
.repeat NHEAD, h
    .byte >(.ident (.sprintf ("qkchain%d", h)))
.endrepeat
    .assert <qkp <> $FF, error, "qkp would hit the JMP-indirect page bug"

.else
; ---- legacy long-context path --------------------------------------------
; The pointer-walking kernels the self-modified chains replaced.  Kept because
; they are the only form that works when the cache is too large for the chains'
; assembled absolute addresses; three to four times slower per multiply-add.
attention:
    stx xsave
    lda #0
    sta hbase
@head:
    lda hbase
    clc
    adc #NDHEAD
    sta hend
    jsr attn_head
    lda hend
    sta hbase
    cmp #NDMODEL
    bcc @head
    ldx xsave
    rts

attn_head:
    ; ---- scores for t = 0..curpos ------------------------------------
    lda #0
    sta tcnt
@score:
    lda #0
    sta kvsel
    lda tcnt
    sta kvt
    jsr kv_ptr
    jsr dot_qk
    ldy tcnt
    lda scL
    sta SCORL,y
    lda scH
    sta SCORH,y
    inc tcnt
    lda tcnt
    cmp curpos
    beq @score
    bcc @score

    jsr softmax

    ; ---- AV ----------------------------------------------------------
    ldy hbase
    lda #0
@zero:
    sta AVL,y
    sta AVH,y
    iny
    cpy hend
    bne @zero

    lda #0
    sta tcnt
@group:
    ldy hbase
    lda #0
@z2:
    sta ACC8,y
    iny
    cpy hend
    bne @z2
    lda #0
    sta gi
@avt:
    lda #1
    sta kvsel
    lda tcnt
    sta kvt
    jsr kv_ptr
    ldy tcnt
    lda P4HI,y
    sta ph
    jsr acc_av
    inc tcnt
    inc gi
    lda tcnt
    cmp curpos
    beq @cont
    bcs @flush
@cont:
    lda gi
    cmp #8
    bcc @avt
@flush:
    jsr av_fold
    lda tcnt
    cmp curpos
    beq @group
    bcc @group
    jsr av_quant
    rts

; ---- score = sum_j mul[(q<<4)|k] - MULBIAS*NDHEAD, blocked by 8 -----------
dot_qk:
    lda #0
    sta scL
    sta scH
    ldy hbase
    lda #NDHEAD / 8
    sta blkn
@blk:
    lda #0
    sta bacc
    clc
    .repeat 8
    lda (kptr),y
    ora Q4HI,y
    tax
    lda bacc
    adc tbl_mul,x
    sta bacc
    iny
    .endrepeat
    lda bacc
    clc
    adc scL
    sta scL
    bcc @1
    inc scH
@1:
    dec blkn
    beq @out
    jmp @blk                ; the unrolled block is >127 bytes, so bne cannot reach
@out:
    sec
    lda scL
    sbc #<(MULBIAS * NDHEAD)
    sta scL
    lda scH
    sbc #>(MULBIAS * NDHEAD)
    sta scH
    rts

; ---- add one position's products into the 8-bit block accumulators -------
acc_av:
    ldy hbase
    lda #NDHEAD / 8
    sta blkn
    clc
@blk:
    .repeat 8
    lda (kptr),y
    ora ph
    tax
    lda ACC8,y
    adc tbl_pv,x
    sta ACC8,y
    iny
    .endrepeat
    dec blkn
    beq @out
    jmp @blk                ; the unrolled block is >127 bytes
@out:
    rts

av_fold:
    ; uses X, not Y: the 6502 has no `inc abs,y`.  X is free here because
    ; `attention` already saved the weight-stream offset in xsave.
    ldx hbase
@l:
    lda ACC8,x
    clc
    adc AVL,x
    sta AVL,x
    bcc @1
    inc AVH,x
@1:
    inx
    cpx hend
    bne @l
    rts

; ---- att[j] = quant(AV[j] - MULBIAS*(curpos+1), 4) ------------------------
av_quant:
    lda curpos
    clc
    adc #1
    sta t0
    lda #0
    sta sumL
    sta sumH
    ldy t0
@mul:
    lda sumL
    clc
    adc #PVBIAS
    sta sumL
    lda sumH
    adc #0
    sta sumH
    dey
    bne @mul

    ldy hbase
@l:
    sty t1
    sec
    lda AVL,y
    sbc sumL
    sta totL
    lda AVH,y
    sbc sumH
    sta totH
    jsr requant_k4
    ldy t1
    sta ATTV,y
    iny
    cpy hend
    bne @l
    rts
.endif

; ===========================================================================
; softmax over SCORL/SCORH[0..curpos] -> P4HI
; ===========================================================================
softmax:
    lda SCORL
    sta bestL
    lda SCORH
    sta bestH
    lda #1
    sta tcnt
@mx:
    lda tcnt
    cmp curpos
    beq @mxgo
    bcs @mxdone
@mxgo:
    ; signed 16-bit: is SCOR[tcnt] > best ?
    ldy tcnt
    lda SCORH,y
    eor #$80
    sta t0
    lda bestH
    eor #$80
    cmp t0
    bcc @mxset
    bne @mxnext
    lda bestL
    cmp SCORL,y
    bcs @mxnext
@mxset:
    ldy tcnt
    lda SCORL,y
    sta bestL
    lda SCORH,y
    sta bestH
@mxnext:
    inc tcnt
    jmp @mx
@mxdone:

    lda #0
    sta tcnt
    sta sumL
    sta sumH
@e:
    ; diff = SCOR[t] - best, always <= 0.  The high byte decides which of
    ; three cases applies and the low byte indexes the table, so the whole
    ; shift-and-clamp is one 4-cycle load.
    ldy tcnt
    sec
    lda SCORL,y
    sbc bestL
    tax
    lda SCORH,y
    sbc bestH
    beq @zero               ; diff == 0
    cmp #$FF
    bne @far                ; diff <= -257: past the bottom of the table
    lda tbl_sm,x
    bcs @have               ; sbc left C set: diff > -32768, always true here
@zero:
    lda #EXP_TOP
    bne @have               ; EXP_TOP is 64, never zero
@far:
    lda #EXP_BOT
@have:
    ldy tcnt
    sta EXPE,y
    clc
    adc sumL
    sta sumL
    bcc @1
    inc sumH
@1:
    inc tcnt
    lda tcnt
    cmp curpos
    beq @e
    bcc @e

.if SM_EXACTNORM
    ; ---- exact normalisation ---------------------------------------------
    ; p_t = min(e_t * SM_TARGET / S, PMAX), with S = sum(e) intact.
    ;
    ; The shipped normaliser is a SHARED EXPONENT: kk is the smallest shift
    ; with S>>kk <= SM_TARGET, so the realised sum lands anywhere in
    ; (SM_TARGET/2, SM_TARGET] and a quarter of the time the softmax runs on
    ; half its budget.  Measured, removing that waste is worth 0.0111
    ; nats/char - more than widening the nibble bought, which was nothing.
    ;
    ; e takes only the EXP_N values in tbl_exp and they ASCEND, so p ascends
    ; with them.  Walk the table once, advancing a 16-bit threshold
    ; accumulator (p+1)*S, and write p<<4 into a 65-entry RAM map indexed by
    ; e.  EXP_N outer steps and at most PMAX inner ones, ONCE per softmax;
    ; the per-position loop then costs one absolute,y load, which is a cycle
    ; CHEAPER than the (smp),y it replaces.
    lda sumL
    sta accL
    lda sumH
    sta accH
    lda #0
    sta pcur
    ; entries below EXP_FIRST are all zero and all map to p = 0, so the walk
    ; starts past them; PTBL[0] is set once at reset and never changes.
    ldx #EXP_FIRST
@ex:
    lda tbl_e8lo,x              ; e8 = e * SM_TARGET, precomputed
    sta e8L
    lda tbl_e8hi,x
    sta e8H
@exw:
    lda pcur
    cmp #PMAX
    bcs @exs                    ; already at the clamp
    lda e8L                     ; 16-bit unsigned: is e8 >= (p+1)*S ?
    cmp accL
    lda e8H
    sbc accH
    bcc @exs
    inc pcur
    clc
    lda accL
    adc sumL
    sta accL
    lda accH
    adc sumH
    sta accH
    jmp @exw
@exs:
    ldy pcur
    lda tbl_p4,y                ; the AV chain wants the row's low byte,
    ldy tbl_exp,x               ; which IS p<<4
    sta PTBL,y
    inx
    cpx #EXP_N
    bcc @ex

    lda #0
    sta tcnt
@p:
    ldy tcnt
    lda EXPE,y
    tay
    lda PTBL,y
    ldy tcnt
    sta P4HI,y
    inc tcnt
    lda tcnt
    cmp curpos
    beq @p
    bcc @p
    rts
.else
    ; kk = smallest k with (S >> k) <= SM_TARGET
    lda #0
    sta kk
@kkl:
    lda sumH
    bne @shift
    lda sumL
    cmp #SM_TARGET + 1
    bcc @kkdone
@shift:
    lsr sumH
    ror sumL
    inc kk
    jmp @kkl
@kkdone:
    lda kk                  ; rows from kk = 7 up are all zero, so clamping
    cmp #SM_KROWS           ; the row index is exact, not an approximation
    bcc @kok
    lda #SM_KROWS - 1
@kok:
    tax
    lda smrowlo,x
    sta smp
    lda smrowhi,x
    sta smp+1

    lda #0
    sta tcnt
@p:
    ldy tcnt
    lda EXPE,y
    tay
    lda (smp),y             ; = min(e >> kk, 7) << 4
    ldy tcnt
    sta P4HI,y
    inc tcnt
    lda tcnt
    cmp curpos
    beq @p
    bcc @p
    rts

smrowlo:
.repeat SM_KROWS, k
    .byte <(tbl_p + k * SM_PROW)
.endrepeat
smrowhi:
.repeat SM_KROWS, k
    .byte >(tbl_p + k * SM_PROW)
.endrepeat
    .assert <smp <> $FF, error, "smp must not straddle a zero page wrap"
.endif

; ===========================================================================
; output head: NVOCAB raw rows, argmax with ties going to the lowest index
; ===========================================================================
head_argmax:
    lda #$00
    sta bestL
    lda #$80                ; -32768
    sta bestH
    lda #0
    sta besti
    sta di
    lda #NVOCAB
    sta rowcnt
@row:
    jsr read_header
    PMARK 20
    jsr gather_row
    PMARK 21
    ; is tot > best ?  (signed 16-bit)
    lda totH
    eor #$80
    sta t0
    lda bestH
    eor #$80
    cmp t0
    bcc @better
    bne @next
    lda bestL
    cmp totL
    bcs @next
@better:
    lda totL
    sta bestL
    lda totH
    sta bestH
    lda di
    sta besti
@next:
    inc di
    dec rowcnt
    bne @row
    lda besti
    sta curtok
    rts

; ===========================================================================
.segment "VECTORS"
    .word reset
    .word reset
    .word reset
