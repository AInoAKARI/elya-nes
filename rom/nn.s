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
; SAME value through NES_T.  The bound is the KV cache: L * NCTX * 2 rows of
; NDMODEL bytes, one byte per 4-bit activation, in the 32 KB of PRG-RAM the
; MMC5 windows at $6000 in four 8 KB banks.
.ifndef NCTX
NCTX     = 20
.endif
NTOKGEN  = NCTX - 1         ; >= 16, the verification bar

KVROWS   = NLAYER * 2 * NCTX        ; 64-byte rows: [layer][k|v][t]
KVBYTES  = KVROWS * NDMODEL
KVBANKS  = (KVBYTES + 8191) / 8192
KVBANK0  = 4                ; $5113 = 4,5,6,7 are the four real PRG-RAM banks
KVLAST   = (KVBYTES - 1) / 8192
    .assert KVBYTES <= 32768, error, "KV cache exceeds the 32 KB of PRG-RAM"
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
; NCTX*NDMODEL, which at NCTX = 85 is 5,440 - together 9,536 bytes, more than
; the 8 KB $A000 window holds.  So they get a bank each and `embed_pos`
; switches between them once per token (6 cycles, MMC5).  The spare bank keeps
; the PRG image a whole number of 16 KB iNES units.
EMBBANK  = $80 + NSTREAM          ; $A000 window: embedding
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

; ---- fixed pages in system RAM (constants, NOT bss - see nn.cfg) ----------
; The four per-position arrays (SCORL, SCORH, EXPE, P4HI) are NCTX entries
; each, so at NCTX = 85 they need 340 bytes where the T = 20 map spent 128.
; Three of them fit page 7 alongside nothing else; P4HI goes in BSS, where the
; linker's $0200-$02FF cap turns an overflow into a LINK ERROR rather than a
; silent collision with the marker port.  Every base below is chosen so that
; base_lo + max_index <= 255, i.e. NO indexed access here crosses a page.
ACTB   = $0400              ; biased activation input page (the adc target)
OUTB   = $0500              ; signed matmul output, up to 128 entries
XVEC   = $0580              ; residual stream x[0..63], signed
Q4HI   = $05C0              ; (q & 15) << 4
ATTV   = $0600              ; attention output, signed
ACC8   = $0640              ; 8-bit block accumulators for AV
AVL    = $0680
AVH    = $06C0
SCORL  = $0700
SCORH  = SCORL + NCTX
EXPE   = SCORH + NCTX
    .assert EXPE + NCTX <= $0800, error, "score arrays overflow page 7"
    .assert <SCORH + NCTX - 1 < 256, error, "SCORH,y would cross a page"
    .assert <EXPE + NCTX - 1 < 256, error, "EXPE,y would cross a page"

EMBED   = $A000             ; EMBBANK
POSTAB  = $A000             ; POSBANK - its own bank, see above
HEADERS = $C000
KVBASE  = $6000
; "ELYA" magic + count + NTOKGEN token ids, parked in the tail of the last KV
; bank so an emulator with no scripting hook can be cross-checked through its
; .sav file.  At NCTX = 85 the cache leaves only 128 bytes there.
SAVBASE = $8000 - (5 + NTOKGEN)
KVLASTEND = $6000 + (KVBYTES - KVLAST * 8192)
    .assert SAVBASE >= KVLASTEND, error, "sav block overlaps the KV cache"

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
kvt:     .res 1
blkn:    .res 1
sumL:    .res 1
sumH:    .res 1
kk:      .res 1
bestL:   .res 1
bestH:   .res 1
besti:   .res 1
t0:      .res 1
t1:      .res 1
rowL:    .res 1             ; KV row index, 16 bit: row = (l*2+kv)*NCTX + t
rowH:    .res 1
bnkc:    .res 1             ; PRG-RAM bank counter, reset only

.segment "BSS"
res_tokens: .res 96
res_ntok:   .res 1
P4HI:       .res NCTX       ; quantised softmax output, one nibble per position
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

.segment "POS"
    .incbin "out/model/pos.bin"

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
    lda #KVBANK0            ; PRG-RAM bank 4 (first of the four real banks)
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
    sta $0200,x
    sta $0400,x
    sta $0500,x
    sta $0600,x
    sta $0700,x
    inx
    bne @clr
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

    lda #SEEDTOK
    sta curtok
    lda #0
    sta curpos

    ldx #254
    stx MARKER              ; SYNC
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
    ; scripting hook (ares) can be cross-checked through its .sav file
    lda #KVBANK0 + KVLAST   ; the sav block lives in the LAST KV bank
    sta MMC5_RAMBANK
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
; The debug snapshots park 188 bytes at $7E00-$7FCF, which is free only while
; the KV cache leaves that much of the last bank unused.  At NCTX = 85 it
; leaves 128 bytes.  Refusing to assemble is the honest outcome; silently
; overwriting the KV cache would produce a ROM that disagrees with the host
; and blames the wrong thing.
.if KVBYTES > $1E00
.error "DEBUG snapshots collide with the KV cache at this NCTX"
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
; The embedding and the positional table are in different banks of the same
; $A000 window (together they are 9,536 bytes at NCTX = 85), so this reads the
; embedding row into XVEC first, then switches the window and adds the
; positional row in place.  Two bank writes per token, 12 cycles, against
; ~1.3 million.
embed_pos:
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
    rts

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

; KV layout is [layer][k|v][t]: row = (curlay*2 + kvsel) * NCTX + kvt.  The
; per-(layer, k/v) blocks are then CONTIGUOUS in t, so the attention loops walk
; rows in order and cross a PRG-RAM bank boundary at most once per loop instead
; of ping-ponging.  Row 0 of each block is tabulated because (l*2+kv)*NCTX
; exceeds a byte at NCTX = 85 (max 5*85 = 425).
kvbaseL:
    .repeat NLAYER * 2, i
    .byte <(i * NCTX)
    .endrepeat
kvbaseH:
    .repeat NLAYER * 2, i
    .byte >(i * NCTX)
    .endrepeat

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
    adc tbl_mul,x
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
    adc #MULBIAS
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
    ldy tcnt
    sec
    lda SCORL,y
    sbc bestL
    sta totL
    lda SCORH,y
    sbc bestH
    sta totH
    ldx #3
@sh:
    lda totH
    cmp #$80                ; C = sign bit, so ror is an arithmetic shift
    ror totH
    ror totL
    dex
    bne @sh
    lda totH
    beq @use
    cmp #$FF
    bne @clampd
    lda totL
    cmp #<(-14)
    bcs @use
@clampd:
    lda #<(-14)
    sta totL
@use:
    lda totL
    clc
    adc #14
    tax
    lda tbl_exp,x
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

    ; kk = smallest k with (S >> k) <= 8
    lda #0
    sta kk
@kkl:
    lda sumH
    bne @shift
    lda sumL
    cmp #9
    bcc @kkdone
@shift:
    lsr sumH
    ror sumL
    inc kk
    jmp @kkl
@kkdone:

    lda #0
    sta tcnt
@p:
    ldy tcnt
    lda EXPE,y
    ldx kk
    beq @noshift
@sh2:
    lsr a
    dex
    bne @sh2
@noshift:
    cmp #8
    bcc @ok
    lda #7
@ok:
    asl a
    asl a
    asl a
    asl a
    ldy tcnt
    sta P4HI,y
    inc tcnt
    lda tcnt
    cmp curpos
    beq @p
    bcc @p
    rts

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
