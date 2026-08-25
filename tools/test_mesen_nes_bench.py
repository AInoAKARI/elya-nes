import importlib.util
from pathlib import Path
import unittest
import tempfile


MODULE_PATH = Path(__file__).with_name("mesen_nes_bench.py")
SPEC = importlib.util.spec_from_file_location("mesen_nes_bench", MODULE_PATH)
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)

PAD_PATH = Path(__file__).with_name("mesen_pad_mmc5.py")
PAD_SPEC = importlib.util.spec_from_file_location("mesen_pad_mmc5", PAD_PATH)
PAD = importlib.util.module_from_spec(PAD_SPEC)
PAD_SPEC.loader.exec_module(PAD)


class MesenBenchTests(unittest.TestCase):
    def test_pair_windows_preserves_both_clocks(self):
        events = [
            {"marker": 1, "cpu_cycle": 10, "master_clock": 10,
             "ppu_master_clock": 121},
            {"marker": 2, "cpu_cycle": 25, "master_clock": 25,
             "ppu_master_clock": 301},
        ]
        self.assertEqual(BENCH.pair_windows(events), [{
            "position": 0,
            "cpu_cycles": 15,
            "master_clocks": 180,
            "master_clocks_div_12": 15.0,
            "mesen_master_clock_delta": 15,
            "begin": events[0],
            "end": events[1],
        }])

    def test_sync_marker_is_not_a_measurement_window(self):
        events = [
            {"marker": 254, "cpu_cycle": 1, "master_clock": 1,
             "ppu_master_clock": 12},
            {"marker": 1, "cpu_cycle": 2, "master_clock": 2,
             "ppu_master_clock": 24},
            {"marker": 2, "cpu_cycle": 5, "master_clock": 5,
             "ppu_master_clock": 60},
            {"marker": 255, "cpu_cycle": 6, "master_clock": 6,
             "ppu_master_clock": 72},
        ]
        self.assertEqual(len(BENCH.pair_windows(events)), 1)

    def test_lua_string_normalizes_windows_path(self):
        self.assertEqual(BENCH.lua_string(r"C:\temp\a.txt"), '"C:/temp/a.txt"')

    def test_portable_settings_never_overwrites_existing_config(self):
        with tempfile.TemporaryDirectory() as temp:
            executable = Path(temp) / "Mesen.exe"
            executable.touch()
            self.assertTrue(BENCH.ensure_portable_settings(executable))
            settings = Path(temp) / "settings.json"
            self.assertEqual(settings.read_text(encoding="utf-8"), "{}\n")
            settings.write_text('{"keep":true}\n', encoding="utf-8")
            self.assertFalse(BENCH.ensure_portable_settings(executable))
            self.assertEqual(settings.read_text(encoding="utf-8"), '{"keep":true}\n')

    def test_mmc5_padding_preserves_banks_and_duplicates_fixed_bank(self):
        header = bytearray(b"NES\x1a" + bytes(12))
        header[4], header[5], header[6] = 6, 1, 0x50
        banks = [bytes([index]) * PAD.MMC5_BANK for index in range(12)]
        image = bytes(header) + b"".join(banks) + b"C" * PAD.MMC5_BANK
        padded = PAD.pad_mmc5_image(image)
        self.assertEqual(padded[4], 8)
        new_prg = padded[PAD.HEADER_SIZE:PAD.HEADER_SIZE + 16 * PAD.MMC5_BANK]
        self.assertEqual(new_prg[:12 * PAD.MMC5_BANK], b"".join(banks))
        self.assertEqual(new_prg[-PAD.MMC5_BANK:], banks[-1])
        self.assertEqual(padded[-PAD.MMC5_BANK:], b"C" * PAD.MMC5_BANK)

    def test_mmc5_padding_is_idempotent_for_power_of_two_prg(self):
        header = bytearray(b"NES\x1a" + bytes(12))
        header[4], header[5], header[6] = 8, 0, 0x50
        image = bytes(header) + b"P" * (8 * PAD.PRG_UNIT)
        self.assertIs(PAD.pad_mmc5_image(image), image)


if __name__ == "__main__":
    unittest.main()
