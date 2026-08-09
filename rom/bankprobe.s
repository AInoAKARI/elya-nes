; ---------------------------------------------------------------------------
; bankprobe.s - how many 8 KB PRG banks can this cartridge actually reach?
;
; The MoE plan buys parameters with cartridge ROM, so the budget is the number
; of banks the whole chain will address: the MMC5's 7-bit bank number, the
; iNES header's PRG size field, and the emulator's mapper implementation.  The
; MMC5 documentation says 128 banks = 1 MB.  That is a claim, not a
; measurement, and the shipping cartridge only ever uses twelve.
;
; Every bank is stamped by tools/gen_bankstamp.py with
;     $8000 = bank ^ $5A      $9FFF = bank ^ $A5
; and this ROM maps each bank in turn and checks BOTH stamps.  Checking both
; ends distinguishes "bank not mapped" from "bank mapped but truncated", and
; the XOR keeps a stamp from ever equalling the bank number or the $00/$FF an
; unmapped window reads back.
;
; Results land in RAM at $0200, one byte per bank:
;     2 = both stamps correct   1 = low stamp only   0 = neither
; and marker 200 is emitted once every bank in range has passed.
; ---------------------------------------------------------------------------

.include "common.inc"

.export reset

MMC5_PRGMODE = $5100
MMC5_RAMPRO1 = $5102
MMC5_RAMPRO2 = $5103
MMC5_PRG8000 = $5114
MMC5_PRGA000 = $5115
MMC5_PRGC000 = $5116
MMC5_PRGE000 = $5117

.ifndef NBANK
NBANK = 127                 ; banks stamped with data; the last one is code
.endif

.zeropage
bnk:    .res 1
ok:     .res 1

.segment "BSS"
res:      .res 128
res_pass: .res 1

; ===========================================================================
.segment "HEADER"
    .byte "NES", $1A
    .byte (NBANK + 1) / 2   ; 16 KB units.  NBANK stamped banks + the code bank
    .byte 1                 ; 1 x 8 KB CHR
    .byte $52               ; mapper 5, battery
    .byte $00
    .byte 4
    .byte 0, 0, 0, 0, 0, 0, 0

.segment "STAMPS"
    .incbin "out/model/bankstamp.bin"

; ===========================================================================
.segment "CODE"
reset:
    NES_INIT
    lda #$03
    sta MMC5_PRGMODE        ; PRG mode 3: four switchable 8 KB windows
    lda #$02
    sta MMC5_RAMPRO1
    lda #$01
    sta MMC5_RAMPRO2

    ldx #254
    stx MARKER              ; SYNC

    lda #0
    sta bnk
    sta res_pass
@bank:
    lda bnk
    ora #$80                ; bit 7 = ROM
    sta MMC5_PRG8000
    lda #0
    sta ok
    lda $8000
    eor #$5A
    cmp bnk
    bne @lo
    inc ok
@lo:
    lda $9FFF
    eor #$A5
    cmp bnk
    bne @hi
    inc ok
@hi:
    ldx bnk
    lda ok
    sta res,x
    cmp #2
    bne @next
    inc res_pass
@next:
    inc bnk
    lda bnk
    cmp #NBANK
    bne @bank

    lda res_pass
    cmp #NBANK
    bne @done
    ldx #200                ; 200 = every stamped bank read back both stamps
    stx MARKER
@done:
    ldx #M_DONE
    stx MARKER
@hang:
    jmp @hang

.segment "VECTORS"
    .word reset, reset, reset
