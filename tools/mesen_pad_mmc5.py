#!/usr/bin/env python3
"""Pad a non-power-of-two iNES MMC5 PRG for Mesen's reset mapping.

The benchmark's 96 KiB PRG is twelve 8 KiB MMC5 banks.  MAME maps bank 11
at reset, while Mesen 2.1.1 leaves the reset vector unmapped because MMC5
bank masks assume a power-of-two PRG.  Expanding to 128 KiB and duplicating
the fixed bank at bank 15 preserves every runtime bank number and gives Mesen
the same fixed code at reset.  The measured BEGIN/END region is unchanged.
"""

import argparse
from pathlib import Path


HEADER_SIZE = 16
INES_MAGIC = b"NES\x1a"
MMC5_MAPPER = 5
PRG_UNIT = 16 * 1024
MMC5_BANK = 8 * 1024


def next_power_of_two(value):
    return 1 << (value - 1).bit_length()


def pad_mmc5_image(image):
    if len(image) < HEADER_SIZE or image[:4] != INES_MAGIC:
        raise ValueError("not an iNES image")
    if image[6] & 0x04:
        raise ValueError("trainer-bearing images are not supported")
    mapper = (image[6] >> 4) | (image[7] & 0xF0)
    if mapper != MMC5_MAPPER:
        raise ValueError("expected MMC5 mapper 5, got %d" % mapper)

    prg_size = image[4] * PRG_UNIT
    chr_size = image[5] * 8 * 1024
    expected_size = HEADER_SIZE + prg_size + chr_size
    if len(image) != expected_size:
        raise ValueError(
            "header declares %d bytes, file has %d" % (expected_size, len(image))
        )

    bank_count = prg_size // MMC5_BANK
    padded_banks = next_power_of_two(bank_count)
    if padded_banks == bank_count:
        return image

    prg = image[HEADER_SIZE:HEADER_SIZE + prg_size]
    chr_data = image[HEADER_SIZE + prg_size:]
    fixed_bank = prg[-MMC5_BANK:]
    padding = b"\xff" * ((padded_banks - bank_count - 1) * MMC5_BANK)

    header = bytearray(image[:HEADER_SIZE])
    header[4] = padded_banks // 2
    return bytes(header) + prg + padding + fixed_bank + chr_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    padded = pad_mmc5_image(source.read_bytes())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(padded)
    print("%s: %d -> %d bytes" % (output, source.stat().st_size, len(padded)))


if __name__ == "__main__":
    main()
