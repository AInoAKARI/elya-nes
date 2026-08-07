; ---------------------------------------------------------------------------
; calib.s - 6502 / RP2A03 datasheet calibration ROM (NROM-256, mapper 0)
;
; 27 payloads whose cycle counts are fixed by the datasheet.  The set is
; chosen to probe exactly the places a plausible-but-wrong emulator diverges:
;
;   * indexed LOADS take +1 on a page cross, indexed STORES never do
;     (they are always 5 / 6 because the dummy read is unconditional)
;   * read-modify-write absolute is 6, because the dummy write is modelled
;   * a branch has three distinct costs: 2 not taken, 3 taken, 4 taken across
;     a page - and the page cross is decided by the address of the byte after
;     the branch, which is why the branch tests are placed by hand in an
;     aligned segment and re-verified from the raw ROM bytes, not from labels
;   * the empty payload must measure 0
; ---------------------------------------------------------------------------

.include "common.inc"

.export t_br_nottaken, t_br_taken, t_br_cross, reset, rts_stub

; --- allocated slots (allocated, never assumed; a collision here previously
; --- made banks read back ROM signatures and looked like an emulator fault) --
SCR      = $0310            ; absolute scratch byte
RTBL     = $0400            ; 512-byte page-aligned RAM table ($0400-$05FF)

.zeropage
zscr:    .res 1             ; zero-page scratch
ptr0:    .res 2             ; -> RTBL+0     (page aligned)
ptr250:  .res 2             ; -> RTBL+250   (page cross at +6)
ptrx:    .res 2             ; -> RTBL+0, indexed via (zp,x)

; ===========================================================================
.segment "HEADER"
    .byte "NES", $1A
    .byte 2                 ; 2 x 16 KB PRG
    .byte 1                 ; 1 x  8 KB CHR
    .byte $00               ; flags6: mapper 0 lo, horizontal mirroring
    .byte $00               ; flags7: mapper 0 hi
    .byte 0, 0, 0, 0, 0, 0, 0, 0

; ===========================================================================
.segment "CODE"

reset:
    NES_INIT

    ; clear RAM $0000-$07FF so nothing is left over from a previous reset
    lda #0
    tax
@clr:
    sta $0000,x
    sta $0100,x
    sta $0200,x
    sta $0400,x
    sta $0500,x
    sta $0600,x
    sta $0700,x
    inx
    bne @clr

    ; pointers for the (zp),y and (zp,x) tests
    lda #<RTBL
    sta ptr0
    sta ptrx
    lda #>RTBL
    sta ptr0+1
    sta ptrx+1
    lda #<(RTBL+250)
    sta ptr250
    lda #>(RTBL+250)
    sta ptr250+1

    ; SYNC: tells the host to discard any events from an earlier reset
    ldx #254
    stx MARKER

    ; ---- the test list, in the order the host expects -------------------
    jsr t_empty             ;  1
    jsr t_nop               ;  2
    jsr t_lda_imm           ;  3
    jsr t_lda_zp            ;  4
    jsr t_lda_abs           ;  5
    jsr t_sta_abs           ;  6
    jsr t_lda_absx_ok       ;  7
    jsr t_lda_absx_cross    ;  8
    jsr t_lda_absy_ok       ;  9
    jsr t_lda_absy_cross    ; 10
    jsr t_sta_absx_ok       ; 11
    jsr t_sta_absx_cross    ; 12
    jsr t_sta_absy_cross    ; 13
    jsr t_lda_indy_ok       ; 14
    jsr t_lda_indy_cross    ; 15
    jsr t_sta_indy_cross    ; 16
    jsr t_lda_indx          ; 17
    jsr t_inc_abs           ; 18
    jsr t_inc_zp            ; 19
    jsr t_asl_a             ; 20
    jsr t_jsr_rts           ; 21
    jsr t_pha_pla           ; 22
    jsr t_lda_zpx           ; 23
    jsr t_inc_absx          ; 24
    jsr t_br_nottaken       ; 25
    jsr t_br_taken          ; 26
    jsr t_br_cross          ; 27
    ; the 6-cycle reference used to derive the CPU clock: lda #imm + sta abs
    jsr t_clock_ref         ; 28

    ldx #M_DONE
    stx MARKER
@hang:
    jmp @hang

; ---------------------------------------------------------------------------
; Each test: setup OUTSIDE the window, MARK, payload, MARK, rts.
; MARKX clobbers X, MARKY clobbers Y, neither touches A or the carry flag
; (ldx/ldy #imm sets N and Z only).
; ---------------------------------------------------------------------------

t_empty:                    ; expect 0
    MARKX M_BEGIN
    MARKX M_END
    rts

t_nop:                      ; expect 2
    MARKX M_BEGIN
    nop
    MARKX M_END
    rts

t_lda_imm:                  ; expect 2
    MARKX M_BEGIN
    lda #$00
    MARKX M_END
    rts

t_lda_zp:                   ; expect 3
    MARKX M_BEGIN
    lda zscr
    MARKX M_END
    rts

t_lda_abs:                  ; expect 4
    MARKX M_BEGIN
    lda SCR
    MARKX M_END
    rts

t_sta_abs:                  ; expect 4
    MARKX M_BEGIN
    sta SCR
    MARKX M_END
    rts

t_lda_absx_ok:              ; expect 4  (X live -> MARKY)
    ldx #0
    MARKY M_BEGIN
    lda RTBL,x
    MARKY M_END
    rts

t_lda_absx_cross:           ; expect 5
    ldx #10
    MARKY M_BEGIN
    lda RTBL+250,x
    MARKY M_END
    rts

t_lda_absy_ok:              ; expect 4  (Y live -> MARKX)
    ldy #0
    MARKX M_BEGIN
    lda RTBL,y
    MARKX M_END
    rts

t_lda_absy_cross:           ; expect 5
    ldy #10
    MARKX M_BEGIN
    lda RTBL+250,y
    MARKX M_END
    rts

t_sta_absx_ok:              ; expect 5 (indexed store: always 5)
    ldx #0
    MARKY M_BEGIN
    sta RTBL,x
    MARKY M_END
    rts

t_sta_absx_cross:           ; expect 5 - NOT 6.  The trap.
    ldx #10
    MARKY M_BEGIN
    sta RTBL+250,x
    MARKY M_END
    rts

t_sta_absy_cross:           ; expect 5 - NOT 6.
    ldy #10
    MARKX M_BEGIN
    sta RTBL+250,y
    MARKX M_END
    rts

t_lda_indy_ok:              ; expect 5
    ldy #0
    MARKX M_BEGIN
    lda (ptr0),y
    MARKX M_END
    rts

t_lda_indy_cross:           ; expect 6
    ldy #10
    MARKX M_BEGIN
    lda (ptr250),y
    MARKX M_END
    rts

t_sta_indy_cross:           ; expect 6 - indirect indexed store is always 6
    ldy #10
    MARKX M_BEGIN
    sta (ptr250),y
    MARKX M_END
    rts

t_lda_indx:                 ; expect 6
    ldx #0
    MARKY M_BEGIN
    lda (ptrx,x)
    MARKY M_END
    rts

t_inc_abs:                  ; expect 6 - models the RMW dummy write
    MARKX M_BEGIN
    inc SCR
    MARKX M_END
    rts

t_inc_zp:                   ; expect 5
    MARKX M_BEGIN
    inc zscr
    MARKX M_END
    rts

t_asl_a:                    ; expect 2
    MARKX M_BEGIN
    asl a
    MARKX M_END
    rts

t_jsr_rts:                  ; expect 12
    MARKX M_BEGIN
    jsr rts_stub
    MARKX M_END
    rts
rts_stub:
    rts

t_pha_pla:                  ; expect 7
    MARKX M_BEGIN
    pha
    pla
    MARKX M_END
    rts

t_lda_zpx:                  ; expect 4 - zero page,X never crosses a page
    ldx #0
    MARKY M_BEGIN
    lda zscr,x
    MARKY M_END
    rts

t_inc_absx:                 ; expect 7 - absolute,X RMW is always 7
    ldx #0
    MARKY M_BEGIN
    inc RTBL,x
    MARKY M_END
    rts

; --- the 6-cycle clock reference: lda #imm (2) + sta abs (4) ---------------
t_clock_ref:                ; expect 6
    MARKX M_BEGIN
    lda #$00
    sta SCR
    MARKX M_END
    rts

; ===========================================================================
; Branch tests.  Placement is by hand and verified from raw ROM bytes.
; The page cross is decided by comparing the high byte of (branch operand
; address + 1) with the high byte of the target.
; ===========================================================================
.segment "ALIGN256"

; --- not taken: carry is clear, bcs falls through -> 2 ---------------------
; A not-taken branch costs 2 whatever the target's page, so no placement
; constraint applies here.
t_br_nottaken:
    clc                     ; outside the window; ldy #imm does not touch C
    ldy #M_BEGIN
    sty MARKER
    bcs @tgt                ; not taken -> 2
@tgt:
    ldy #M_END
    sty MARKER
    rts

; --- taken, same page -> 3 -------------------------------------------------
; Placed at offset 0 of a page:
;   +0 clc  +1 ldy#  +3 sty abs  +6 bcc (operand +7, PC after = +8)
;   +8..+10 filler, target at +11 -> same page.
    .align $100
t_br_taken:
    clc
    ldy #M_BEGIN
    sty MARKER
    bcc @tgt                ; taken, no page cross -> 3
@after:
    .res 3, $EA
@tgt:
    ldy #M_END
    sty MARKER
    rts
    .assert >(@after) = >(@tgt), error, "br_taken must NOT cross a page"

; --- taken across a page boundary -> 4 -------------------------------------
; Placed at offset 245 of a page:
;   +245 clc  +246 ldy#  +248 sta abs  +251 bcc (operand +252, PC after +253)
;   +253..+255 filler, target at +256 -> next page.
    .align $100
    .res 245, $EA
t_br_cross:
    clc
    ldy #M_BEGIN
    sty MARKER
    bcc @tgt                ; taken across a page -> 4
@after:
    .res 3, $EA
@tgt:
    ldy #M_END
    sty MARKER
    rts
    .assert >(@after) <> >(@tgt), error, "br_cross MUST cross a page"

; ===========================================================================
.segment "VECTORS"
    .word reset             ; NMI  - never fires (PPUCTRL=0)
    .word reset             ; RESET
    .word reset             ; IRQ  - never fires (SEI + IRQ inhibit)
