; ---------------------------------------------------------------------------
; prim.s - MMC5 primitive measurements
;
; Everything the port's cost model rests on, measured rather than assumed:
; bank switching, the $6000 PRG-RAM window, page-aligned table reads, the four
; accumulator widths, and the two candidate ternary inner loops.
; ---------------------------------------------------------------------------

.include "common.inc"

.export reset
.export res_rambank, res_sig, res_nbank

; --- MMC5 registers --------------------------------------------------------
MMC5_PRGMODE = $5100
MMC5_RAMPRO1 = $5102
MMC5_RAMPRO2 = $5103
MMC5_RAMBANK = $5113        ; PRG-RAM bank at $6000
MMC5_PRG8000 = $5114
MMC5_PRGA000 = $5115
MMC5_PRGC000 = $5116
MMC5_PRGE000 = $5117

TRIT_ZERO = 0
TRIT_POS  = 1
TRIT_NEG  = 2

; --- fixed data pages (constants, NOT bss - see the RAM cap in mmc5.cfg) ----
ACT   = $0400               ; page-aligned activation table (RAM)
SCR   = $0500               ; page-aligned scratch page (RAM)

.zeropage
acc8:   .res 1
acc16:  .res 2
acc32:  .res 4
zt:     .res 1

.segment "BSS"
res_rambank: .res 8         ; PRG-RAM bank aliasing probe
res_sig:     .res 4         ; ROM bank signatures seen at $8000
res_nbank:   .res 1

; ===========================================================================
.segment "HEADER"
    .byte "NES", $1A
    .byte 4                 ; 4 x 16 KB = 64 KB PRG  (eight 8 KB banks)
    .byte 1                 ; 1 x  8 KB CHR
    .byte $52               ; flags6: mapper 5 low nibble, battery
    .byte $00               ; flags7: mapper 5 high nibble = 0
    .byte 8                 ; PRG-RAM: DECLARE 8 x 8 KB = 64 KB and see what
                            ; we actually get back (prior run: only 4 banks)
    .byte 0, 0, 0, 0, 0, 0, 0

; --- bank signature bytes, one per switchable 8 KB bank --------------------
.segment "BANK0"
    .byte $B0
.segment "BANK1"
    .byte $B1
.segment "BANK2"
    .byte $B2
.segment "BANK3"
    .byte $B3
.segment "BANK4"
    .byte $B4
.segment "BANK5"
    .byte $B5
.segment "BANK6"
    .byte $B6

; ===========================================================================
.segment "CODE"

reset:
    NES_INIT

    ; ---- MMC5 into 8 KB PRG banking, PRG-RAM writable --------------------
    lda #$03
    sta MMC5_PRGMODE        ; mode 3: four 8 KB PRG windows
    lda #$02
    sta MMC5_RAMPRO1
    lda #$01
    sta MMC5_RAMPRO2        ; $5102=2,$5103=1 unlocks PRG-RAM writes
    lda #$00
    sta MMC5_RAMBANK        ; PRG-RAM bank 0 at $6000
    lda #$80
    sta MMC5_PRG8000        ; bit7 = ROM; bank 0 at $8000
    lda #$81
    sta MMC5_PRGA000
    lda #$82
    sta MMC5_PRGC000
    lda #$87
    sta MMC5_PRGE000        ; bank 7 fixed at $E000 (this code)

    ; ---- clear RAM -------------------------------------------------------
    lda #0
    tax
@clr:
    sta $0200,x
    sta $0400,x
    sta $0500,x
    sta $0600,x
    sta $0700,x
    sta $0000,x
    inx
    bne @clr

    ; ---- fill the activation page with a known ramp ----------------------
    ldx #0
@fill:
    txa
    sta ACT,x
    inx
    bne @fill

    ; ---- PRG-RAM bank aliasing probe: write b to bank b, read all back ----
    ldx #0
@wr:
    stx MMC5_RAMBANK
    txa
    ora #$A0
    sta $6000
    inx
    cpx #8
    bne @wr
    ldx #0
@rd:
    stx MMC5_RAMBANK
    lda $6000
    sta res_rambank,x
    inx
    cpx #8
    bne @rd
    lda #0
    sta MMC5_RAMBANK

    ; ---- bank signature before / after a switch --------------------------
    lda #$80
    sta MMC5_PRG8000
    lda $8000
    sta res_sig+0           ; expect $B0

    ldx #254
    stx MARKER              ; SYNC

    ; ---------------- the measurement list -------------------------------
    jsr p_mmc5_switch6      ;  1  lda #imm + sta $5114
    jsr p_mmc5_switch4      ;  2  sta $5114 with the value already in A
    jsr p_xram_lda          ;  3  lda $6000
    jsr p_xram_sta          ;  4  sta $6000
    jsr p_xram_ldy_ok       ;  5  lda $6000,y   page aligned
    jsr p_xram_ldy_cross    ;  6  lda $60FA,y   page cross
    jsr p_rom_ldy_ok        ;  7  lda romtbl,y  page aligned
    jsr p_rom_ldy_cross     ;  8  lda romtbl+250,y
    jsr p_acc8_A            ;  9  one accumulate, acc resident in A
    jsr p_acc8_A16          ; 10  sixteen of them (linearity check)
    jsr p_acc8_zp           ; 11  accumulator spilled to zero page
    jsr p_acc16             ; 12
    jsr p_acc32             ; 13
    jsr p_tern_gather       ; 14  sign-separated gather, one element
    jsr p_tern_gather16     ; 15  sixteen of them
    jsr p_tern_skip         ; 16  branchy: zero trit
    jsr p_tern_add          ; 17  branchy: +1 trit
    jsr p_tern_sub          ; 18  branchy: -1 trit
    jsr p_zp_lda            ; 19  lda zp (3) for the $6000 comparison

    ; final signature: the switch tests left bank 2 mapped
    lda $8000
    sta res_sig+1

    ldx #M_DONE
    stx MARKER
@hang:
    jmp @hang

; ---------------------------------------------------------------------------
p_mmc5_switch6:             ; expect 6
    MARKX M_BEGIN
    lda #$81
    sta MMC5_PRG8000
    MARKX M_END
    lda $8000
    sta res_sig+2           ; expect $B1 - proves the switch really happened
    rts

p_mmc5_switch4:             ; expect 4 - value already in A
    lda #$82
    MARKX M_BEGIN
    sta MMC5_PRG8000
    MARKX M_END
    lda $8000
    sta res_sig+3           ; expect $B2
    rts

p_xram_lda:                 ; expect 4 - no cartridge penalty
    MARKX M_BEGIN
    lda $6000
    MARKX M_END
    rts

p_xram_sta:                 ; expect 4
    MARKX M_BEGIN
    sta $6000
    MARKX M_END
    rts

p_xram_ldy_ok:              ; expect 4
    ldy #0
    MARKX M_BEGIN
    lda $6000,y
    MARKX M_END
    rts

p_xram_ldy_cross:           ; expect 5 - $6000 has the same page-cross hazard
    ldy #10
    MARKX M_BEGIN
    lda $60FA,y
    MARKX M_END
    rts

p_rom_ldy_ok:               ; expect 4
    ldy #0
    MARKX M_BEGIN
    lda romtbl,y
    MARKX M_END
    rts

p_rom_ldy_cross:            ; expect 5
    ldy #10
    MARKX M_BEGIN
    lda romtbl+250,y
    MARKX M_END
    rts

p_zp_lda:                   ; expect 3
    MARKX M_BEGIN
    lda zt
    MARKX M_END
    rts

; --- accumulator widths ----------------------------------------------------
p_acc8_A:                   ; expect 4 per element
    lda #0
    clc
    MARKX M_BEGIN
    adc ACT+0
    MARKX M_END
    rts

p_acc8_A16:                 ; expect 64 = 16 x 4
    lda #0
    clc
    MARKX M_BEGIN
    .repeat 16, i
    adc ACT+i
    .endrepeat
    MARKX M_END
    rts

p_acc8_zp:                  ; expect 12
    MARKX M_BEGIN
    lda acc8
    clc
    adc ACT+0
    sta acc8
    MARKX M_END
    rts

p_acc16:                    ; expect 20
    MARKX M_BEGIN
    lda acc16+0
    clc
    adc ACT+0
    sta acc16+0
    lda acc16+1
    adc #0
    sta acc16+1
    MARKX M_END
    rts

p_acc32:                    ; expect 36
    MARKX M_BEGIN
    lda acc32+0
    clc
    adc ACT+0
    sta acc32+0
    lda acc32+1
    adc #0
    sta acc32+1
    lda acc32+2
    adc #0
    sta acc32+2
    lda acc32+3
    adc #0
    sta acc32+3
    MARKX M_END
    rts

; --- ternary inner ops -----------------------------------------------------
; sign-separated: the trit sign is baked into which list the index came from,
; so there is no test and no branch, and the accumulator never leaves A.
p_tern_gather:              ; expect 8
    ldx #0
    lda #0
    clc
    MARKY M_BEGIN
    ldy idxtbl,x
    adc ACT,y
    MARKY M_END
    rts

p_tern_gather16:            ; expect 128 = 16 x 8
    ldx #0
    lda #0
    clc
    MARKY M_BEGIN
    .repeat 16, i
    ldy idxtbl+i,x
    adc ACT,y
    .endrepeat
    MARKY M_END
    rts

; ===========================================================================
; Branchy ternary, for comparison.  The trit is tested at run time, which
; forces the accumulator out of A and into memory.
; Placed in an aligned segment so the taken branches provably do not cross a
; page (a cross would silently add 1 and make the comparison flattering).
; ===========================================================================
.segment "ALIGN256"
    .align $100

p_tern_skip:                ; expect 7 = lda abs,x (4) + beq taken (3)
    ldx #TRIT_ZERO
    MARKY M_BEGIN
    lda trittbl,x
    beq @skip
@a1:
    nop
    nop
@skip:
    MARKY M_END
    rts
    .assert >(@a1) = >(@skip), error, "tern_skip beq must not cross a page"

p_tern_add:                 ; expect 20
    ldx #TRIT_POS
    MARKY M_BEGIN
    lda trittbl,x           ; 4
    beq @skip               ; 2 not taken
    bmi @neg                ; 2 not taken
    lda acc8                ; 3
    clc                     ; 2
    adc ACT,x               ; 4
    sta acc8                ; 3
    MARKY M_END
    rts
@neg:
@skip:
    MARKY M_END
    rts

p_tern_sub:                 ; expect 21
    ldx #TRIT_NEG
    MARKY M_BEGIN
    lda trittbl,x           ; 4
    beq @skip               ; 2 not taken
    bmi @neg                ; 3 TAKEN, same page
@a2:
    jmp @skip
@neg:
    lda acc8                ; 3
    sec                     ; 2
    sbc ACT,x               ; 4
    sta acc8                ; 3
    MARKY M_END
    rts
@skip:
    MARKY M_END
    rts
    .assert >(@a2) = >(@neg), error, "tern_sub bmi must not cross a page"

; ===========================================================================
.segment "RODATA"
    .align $100
romtbl:
    .repeat 256, i
    .byte i
    .endrepeat
    .repeat 256, i
    .byte 255 - i
    .endrepeat

    .align $100
idxtbl:                     ; sign-separated index list (page aligned)
    .repeat 256, i
    .byte (i * 7) & $3F
    .endrepeat

    .align $100
trittbl:                    ; 0 / +1 / -1 trits for the branchy variant
    .byte $00, $01, $FF
    .res 253, $00

; ===========================================================================
.segment "VECTORS"
    .word reset
    .word reset
    .word reset
