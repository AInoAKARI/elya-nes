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
NCTX     = 20
NTOKGEN  = 19               ; >= 16, the verification bar

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
EMBBANK  = $80 + NSTREAM          ; $A000 window: embedding + positions
TBLBANK  = $80 + NSTREAM + 1      ; $C000 window: row headers + tables
CODEBANK = $80 + NSTREAM + 2      ; $E000 window: code, chains, vectors

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
ACTB   = $0400              ; biased activation input page (the adc target)
OUTB   = $0500              ; signed matmul output, up to 128 entries
XVEC   = $0600              ; residual stream x[0..63], signed
Q4HI   = $0640              ; (q & 15) << 4
ATTV   = $0680              ; attention output, signed
ACC8   = $06C0              ; 8-bit block accumulators for AV
SCORL  = $0700
SCORH  = $0720
EXPE   = $0740
P4HI   = $0760
AVL    = $0780
AVH    = $07C0

EMBED   = $A000
POSTAB  = $A000 + NVOCAB * NDMODEL
HEADERS = $C000
KVBASE  = $6000

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

.segment "BSS"
res_tokens: .res 32
res_ntok:   .res 1

; ===========================================================================
.segment "HEADER"
    .byte "NES", $1A
    .byte (NSTREAM + 3 + 1) / 2   ; 16 KB units: NSTREAM + emb + tbl + code
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
    lda #$04                ; PRG-RAM bank 4 (first of the four real banks)
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
    ; clear the KV cache
    ldx #0
@clrkv:
    sta $6000,x
    sta $6100,x
    sta $6200,x
    sta $6300,x
    sta $6400,x
    sta $6500,x
    sta $6600,x
    sta $6700,x
    sta $6800,x
    sta $6900,x
    sta $6A00,x
    sta $6B00,x
    sta $6C00,x
    sta $6D00,x
    sta $6E00,x
    sta $6F00,x
    sta $7000,x
    sta $7100,x
    sta $7200,x
    sta $7300,x
    inx
    bne @clrkv

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
    ldx #0
@sav:
    lda savmagic,x
    sta $7FE0,x
    inx
    cpx #4
    bne @sav
    lda #NTOKGEN
    sta $7FE4
    ldx #0
@sav2:
    lda res_tokens,x
    sta $7FE5,x
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
    cpy #20
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
embed_pos:
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

lay40:
    .byte 0, NCTX * 2, NCTX * 4

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

; kptr = KVBASE + (lay40[curlay] + kvt*2 + kvsel) * 64      (clobbers A/Y)
kv_ptr:
    ldy curlay
    lda lay40,y
    clc
    adc kvt
    adc kvt
    adc kvsel
    pha
    and #$03
    tay
    lda mul64lo,y
    sta kptr
    pla
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
    AMARK 34
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
    AMARK 35

    AMARK 36
    jsr softmax
    AMARK 37

    ; ---- AV ----------------------------------------------------------
    AMARK 40
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
    AMARK 41
    rts

; ---- score = sum_j mul[(q<<4)|k] - MULBIAS*NDHEAD, blocked by 8 -----------
dot_qk:
    AMARK 32
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
    AMARK 33
    rts

; ---- add one position's products into the 8-bit block accumulators -------
acc_av:
    AMARK 38
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
    AMARK 39
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
