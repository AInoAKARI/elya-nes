; ---------------------------------------------------------------------------
; mmc1.s - MMC1 (mapper 1) PRG bank switch cost, measured on a real MMC1 cart.
;
; MMC1 has no parallel port: the 5-bit bank number is clocked in one bit per
; write to $E000, so a switch is lda + five stores + four shifts.  That is the
; whole reason the port uses MMC5 instead.
; ---------------------------------------------------------------------------
.include "common.inc"
.export reset, res_sig

.segment "BSS"
res_sig: .res 4

.segment "HEADER"
    .byte "NES", $1A
    .byte 4                 ; 4 x 16 KB PRG
    .byte 1                 ; 1 x  8 KB CHR
    .byte $10               ; flags6: mapper 1
    .byte $00
    .byte 0, 0, 0, 0, 0, 0, 0, 0

.segment "BANK0"
    .byte $C0
.segment "BANK1"
    .byte $C1
.segment "BANK2"
    .byte $C2

.segment "CODE"
reset:
    NES_INIT

    ; reset the MMC1 shift register, then set control = $0C
    ; (PRG mode 3: $8000 switchable, $C000 fixed to the last bank)
    lda #$80
    sta $8000               ; bit7 set = reset shift register
    lda #$0C
    jsr mmc1_write_8000

    lda #0
    jsr mmc1_write_prg      ; bank 0 at $8000
    lda $8000
    sta res_sig+0           ; expect $C0

    ldx #254
    stx MARKER

    jsr p_mmc1_switch       ; the measurement

    ldx #M_DONE
    stx MARKER
@hang:
    jmp @hang

; --- 30 cycles: lda #imm (2) + 5 x sta abs (20) + 4 x lsr a (8) ------------
p_mmc1_switch:
    MARKX M_BEGIN
    lda #2
    sta $E000
    lsr a
    sta $E000
    lsr a
    sta $E000
    lsr a
    sta $E000
    lsr a
    sta $E000
    MARKX M_END
    lda $8000
    sta res_sig+1           ; expect $C2 - proves the switch happened
    rts

; --- helpers (outside any measurement window) ------------------------------
mmc1_write_prg:
    sta $E000
    lsr a
    sta $E000
    lsr a
    sta $E000
    lsr a
    sta $E000
    lsr a
    sta $E000
    rts

mmc1_write_8000:
    sta $8000
    lsr a
    sta $8000
    lsr a
    sta $8000
    lsr a
    sta $8000
    lsr a
    sta $8000
    rts

.segment "VECTORS"
    .word reset
    .word reset
    .word reset
