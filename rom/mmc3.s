; ---------------------------------------------------------------------------
; mmc3.s - MMC3 (mapper 4) PRG bank switch cost, measured on a real MMC3 cart.
;
; MMC3 is a two-register port: select the bank register at $8000, then write
; the bank at $8001.  A full switch is 12 cycles; if the register is already
; selected ("hot"), a switch is 6.
; ---------------------------------------------------------------------------
.include "common.inc"
.export reset, res_sig

.segment "BSS"
res_sig: .res 4

.segment "HEADER"
    .byte "NES", $1A
    .byte 4                 ; 4 x 16 KB = 64 KB PRG (eight 8 KB banks)
    .byte 1
    .byte $40               ; flags6: mapper 4
    .byte $00
    .byte 0, 0, 0, 0, 0, 0, 0, 0

.segment "BANK0"
    .byte $D0
.segment "BANK1"
    .byte $D1
.segment "BANK2"
    .byte $D2
.segment "BANK3"
    .byte $D3
.segment "BANK4"
    .byte $D4
.segment "BANK5"
    .byte $D5

.segment "CODE"
reset:
    NES_INIT

    lda #$06                ; R6 = the $8000 window
    sta $8000
    lda #$00
    sta $8001               ; bank 0 at $8000
    lda $8000
    sta res_sig+0           ; expect $D0

    ldx #254
    stx MARKER

    jsr p_mmc3_full         ; 1: select register + write bank
    jsr p_mmc3_hot          ; 2: register already selected

    ldx #M_DONE
    stx MARKER
@hang:
    jmp @hang

; --- 12 cycles: lda#(2) + sta abs(4) + lda#(2) + sta abs(4) ----------------
p_mmc3_full:
    MARKX M_BEGIN
    lda #$06
    sta $8000
    lda #$01
    sta $8001
    MARKX M_END
    lda $8000
    sta res_sig+1           ; expect $D1
    rts

; --- 6 cycles: R6 is still selected ----------------------------------------
p_mmc3_hot:
    MARKX M_BEGIN
    lda #$03
    sta $8001
    MARKX M_END
    lda $8000
    sta res_sig+2           ; expect $D3
    rts

.segment "VECTORS"
    .word reset
    .word reset
    .word reset
