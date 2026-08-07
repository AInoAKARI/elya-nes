#!/usr/bin/env python3
"""Verify branch placement from the RAW ROM BYTES, not from labels.

The three branch payloads only measure three distinct values (2 / 3 / 4) if
the branch really is not-taken / taken-same-page / taken-across-a-page.  A
label can say whatever the assembler wants; the ROM image is the truth, so
this decodes the actual opcode and relative operand out of the .nes file and
recomputes the page cross the way the CPU does:

    PC_after = address_of_branch + 2
    target   = PC_after + signed(operand)
    crossed  = (PC_after >> 8) != (target >> 8)

Usage: check_branches.py <rom.nes> <labels.lbl>
"""
import sys

BRANCH = {0x10: "bpl", 0x30: "bmi", 0x50: "bvc", 0x70: "bvs",
          0x90: "bcc", 0xB0: "bcs", 0xD0: "bne", 0xF0: "beq"}

# name -> (expected mnemonic, expected taken?, expected crossed?, expected cycles)
EXPECT = {
    "t_br_nottaken": ("bcs", False, None, 2),
    "t_br_taken":    ("bcc", True,  False, 3),
    "t_br_cross":    ("bcc", True,  True,  4),
}


def load_labels(path):
    out = {}
    for line in open(path):
        p = line.split()
        if len(p) == 3 and p[0] == "al":
            out[p[2].lstrip(".")] = int(p[1], 16)
    return out


def main():
    rom = open(sys.argv[1], "rb").read()
    lbl = load_labels(sys.argv[2])
    prg_base = 0x8000
    hdr = 16

    def byte(addr):
        return rom[hdr + (addr - prg_base)]

    ok = True
    for name, (mn, taken, crossed, cyc) in EXPECT.items():
        if name not in lbl:
            print("MISSING LABEL %s" % name)
            ok = False
            continue
        a = lbl[name]
        # walk the block looking for the first branch opcode, decoding
        # instruction lengths so we do not mistake an operand for an opcode
        LEN = {0x18: 1, 0xA0: 2, 0x8C: 3, 0xA2: 2, 0x8E: 3}
        p = a
        for _ in range(8):
            op = byte(p)
            if op in BRANCH:
                break
            if op not in LEN:
                print("%-16s UNKNOWN OPCODE $%02X at $%04X" % (name, op, p))
                ok = False
                break
            p += LEN[op]
        else:
            print("%-16s no branch found" % name)
            ok = False
            continue
        if op not in BRANCH:
            continue
        rel = byte(p + 1)
        if rel >= 0x80:
            rel -= 0x100
        pc_after = p + 2
        target = pc_after + rel
        cr = (pc_after >> 8) != (target >> 8)
        good = (BRANCH[op] == mn) and (crossed is None or cr == crossed)
        print("%-16s $%04X %s rel=%+d  PC_after=$%04X target=$%04X "
              "crossed=%s  expect %s taken=%s crossed=%s -> %d cyc  %s"
              % (name, p, BRANCH[op], rel, pc_after, target, cr,
                 mn, taken, crossed, cyc, "OK" if good else "MISMATCH"))
        ok = ok and good
    print("BRANCH PLACEMENT:", "OK" if ok else "BAD")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
