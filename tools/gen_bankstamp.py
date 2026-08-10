#!/usr/bin/env python3
"""Stamp every 8 KB PRG bank with an identity pattern for the bank probe.

The MoE plan spends cartridge ROM it does not currently have, so "how many
banks can this cartridge actually address" has to be a measurement of the
whole chain - MMC5 register, iNES header, and the emulator - not a reading of
the MMC5 documentation.  Each bank gets two stamps, one at each end, so a bank
that is mapped but truncated is distinguishable from one that is not mapped.

    bank b:  byte at $8000 = b ^ 0x5A
             byte at $9FFF = b ^ 0xA5

The XORs keep the stamps from being equal to the bank number, to each other,
or to the $00/$FF a missing bank reads back as.
"""
import os
import sys

BANK = 8192


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 127
    out = sys.argv[2] if len(sys.argv) > 2 else "out/model/bankstamp.bin"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    img = bytearray(b"\x00" * (n * BANK))
    for b in range(n):
        img[b * BANK] = b ^ 0x5A
        img[b * BANK + BANK - 1] = b ^ 0xA5
    with open(out, "wb") as f:
        f.write(bytes(img))
    print("wrote %s: %d banks, %d bytes" % (out, n, len(img)))


if __name__ == "__main__":
    main()
