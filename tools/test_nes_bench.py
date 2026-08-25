#!/usr/bin/env python3
"""Tests for portable MAME executable discovery."""
import os
import unittest
from unittest import mock

import nes_bench


class FindMameTests(unittest.TestCase):
    def test_prefers_explicit_mame_bin(self):
        with mock.patch.dict(os.environ, {"MAME_BIN": r"C:\mame\mame.exe"}):
            self.assertEqual(nes_bench.find_mame(), r"C:\mame\mame.exe")

    def test_uses_path_before_linux_fallback(self):
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(nes_bench.shutil, "which", side_effect=[None, r"C:\mame.exe"]),
        ):
            self.assertEqual(nes_bench.find_mame(), r"C:\mame.exe")


if __name__ == "__main__":
    unittest.main()
