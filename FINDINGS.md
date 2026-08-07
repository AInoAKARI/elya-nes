# NES/Famicom cycle-measurement instrument + transformer port — FINDINGS

Journal. Appended after every discrete result. Committed alongside.
Rebuild after a scratchpad wipe; measured values from the prior run are
carried in as *expectations to reproduce*, never as results.

---

## 1. Instrument rebuilt and calibrated (2026-08-07)

**Method**: MAME 0.277 `nes` driver, headless (`-video none -sound none
-nothrottle -seconds_to_run N`), Lua `install_write_tap` on `$0300`,
timestamping `manager.machine.time` in attoseconds. No polling anywhere; the
tap handle is held in a global `KEEP` table. `nes` needs no BIOS ROM set.

**Marker protocol**: `MARKX v` = `ldx #v` (2) + `stx $0300` (4) = 6 cycles;
`MARKY v` = `ldy`/`sty`, also 6. Neither touches A, and `ld[xy] #imm` sets
only N/Z so the carry flag survives — that is what makes the branch tests
possible. A window is BEGIN..END, so `raw = payload + 6`; the host subtracts
6, and the empty payload therefore reads **0**.

**Clock, DERIVED not assumed.** Reference payload `lda #imm` (2) + `sta abs`
(4); window = 12 cycles including the closing marker.

| candidate | Hz | measured cycles | error |
|---|---|---|---|
| **1789772 (MAME truncation)** | 1789772.0000 | **11.999999999983** | -1.7e-11 |
| 21477272/12 (true NTSC) | 1789772.6667 | 12.000004469826 | +4.5e-06 |
| 1789773 (rounded) | 1789773.0000 | 12.000006704747 | +6.7e-06 |
| 1789800 (folklore) | 1789800.0000 | 12.000187733392 | +1.9e-04 |

Confirms the prior run: MAME instantiates the NTSC 2A03 at an integer
1789772 Hz, truncating 1789772.667. `cycles = round(delta_as*1789772/1e18)`.

**Resolution**: 1 CPU cycle = 558 730 385 770 as = 558.73 ns. Individual
cycles are resolved exactly — a single `nop` reads 2.

**Reproducibility**: 3 runs, marker streams compared **including absolute
timestamps** — BIT-IDENTICAL. MAME's NES boot is deterministic from reset.

**Datasheet calibration: 28/28, 0 mismatches.** Worst deviation of any window
from an integer cycle count: 2.6e-11. The set deliberately probes where a
plausible-but-wrong emulator diverges, and every trap came out right:

- indexed LOAD `lda abs,x|y` 4 -> **5** on a page cross;
- indexed STORE `sta abs,x|y` **5 whether or not the page is crossed** (the
  dummy read is unconditional) — this is the one a naive emulator gets wrong;
- `sta (zp),y` always 6; `lda (zp),y` 5 -> 6 on cross;
- RMW `inc abs` = **6** (dummy write modelled), `inc abs,x` = 7, `inc zp` = 5;
- branch = **three distinct values**: 2 not taken / 3 taken / 4 taken across a
  page;
- empty payload = **0**.

Branch placement is verified from the **raw ROM bytes**, not labels
(`tools/check_branches.py` decodes the opcode and signed relative operand out
of the .nes image and recomputes `page(PC_after) != page(target)`):
`t_br_taken` bcc @ $8406, PC_after $8408 -> $840B, no cross;
`t_br_cross` bcc @ $85FB, PC_after $85FD -> $8600, cross. Link-time
`.assert`s in the ROM enforce the same thing.

Files: `rom/common.inc`, `rom/calib.s`, `rom/nrom.cfg`,
`tools/nes_bench.py`, `tools/run_calib.py`, `tools/check_branches.py`,
report in `out/calib_report.txt`.
