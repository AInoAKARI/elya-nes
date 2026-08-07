; ---------------------------------------------------------------------------
; common.inc - marker protocol + deterministic machine init
;
; The measurement window is delimited by two writes to MARKER ($0300).  A
; MAME Lua write tap timestamps each write with manager.machine.time, so the
; raw delta between the two writes is
;
;       raw = payload_cycles + MARK_COST
;
; where MARK_COST is the cost of the second marker macro itself: the
; immediate load (2) plus the absolute store (4) = 6.  The host runner
; subtracts MARK_COST, so an EMPTY payload must report exactly 0.  That is
; the check that no constant is being smuggled into the numbers.
;
; TRAP (cost a day previously): the marker macro must not clobber the
; register the payload is carrying live state in.  MARKA is therefore NOT
; provided - there is no "store A immediate" - and instead two variants
; exist, MARKX (clobbers X) and MARKY (clobbers Y), both exactly 6 cycles.
; A is never touched by either.
; ---------------------------------------------------------------------------

MARKER      = $0300         ; write tap port
MARK_COST   = 6             ; ldx/ldy #imm (2) + stx/sty abs (4)

M_BEGIN     = 1
M_END       = 2
M_DONE      = 255

; --- 6 cycles, clobbers X, preserves A and Y -------------------------------
.macro MARKX v
    ldx #v                  ; 2
    stx MARKER              ; 4  <- tap fires here
.endmacro

; --- 6 cycles, clobbers Y, preserves A and X -------------------------------
.macro MARKY v
    ldy #v                  ; 2
    sty MARKER              ; 4  <- tap fires here
.endmacro

; ---------------------------------------------------------------------------
; Deterministic init.  Rendering off, no NMI, no APU frame IRQ, no DMC.
; Nothing steals a cycle from the 2A03 after this runs.
; ---------------------------------------------------------------------------
.macro NES_INIT
    sei
    cld
    ldx #$FF
    txs
    inx                     ; X = 0
    stx $2000               ; PPUCTRL  = 0 : NMI off
    stx $2001               ; PPUMASK  = 0 : rendering off
    stx $4010               ; DMC freq = 0 : no DMC IRQ
    stx $4015               ; APU status = 0 : all channels off
    lda #$40
    sta $4017               ; frame counter: IRQ inhibit
.endmacro
