#!/usr/bin/env python3
"""Run an Elya NES ROM under Mesen 2 and capture exact marker clocks."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


MESEN_RELEASE = "2.1.1"
MESEN_WINDOWS_SHA256 = (
    "23ccc2bc060b663c68dad3a8c5d6da7d23a50f872d04f135bafa2b04ff7d5cbe"
)
SOURCE_COMMIT = "2de04401cd7de09d9c44712d3eb4b2dd7190518c"
MAME_PUBLISHED_MEAN = 1_117_248
CPU_MASTER_DIVISOR = 12

M_BEGIN, M_END, M_SYNC, M_DONE = 1, 2, 254, 255


def find_mesen():
    """Return the configured Mesen executable, with portable defaults."""
    configured = os.environ.get("MESEN_BIN")
    if configured:
        return configured
    for candidate in ("Mesen.exe", "Mesen"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise FileNotFoundError("set MESEN_BIN to the Mesen 2 executable")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def lua_string(value):
    """Quote a string for Lua without platform-specific path escaping."""
    return "\"" + str(value).replace("\\", "/").replace("\"", "\\\"") + "\""


def ensure_portable_settings(mesen):
    """Bypass Mesen's first-run GUI without overwriting an existing config."""
    settings = Path(mesen).resolve().with_name("settings.json")
    if not settings.exists():
        settings.write_text("{}\n", encoding="utf-8")
        return True
    return False


def parse_result(path):
    result = {"events": []}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        key, value = line.split("=", 1)
        if key == "event":
            marker, cpu_cycle, master_clock, ppu_master_clock = value.split(",")
            result["events"].append({
                "marker": int(marker),
                "cpu_cycle": int(cpu_cycle),
                "master_clock": int(master_clock),
                "ppu_master_clock": int(ppu_master_clock),
            })
        else:
            result[key] = value
    return result


def pair_windows(events):
    """Return BEGIN/END deltas in both CPU and NES master clocks."""
    windows = []
    start = None
    for event in events:
        if event["marker"] == M_BEGIN:
            start = event
        elif event["marker"] == M_END and start is not None:
            master_delta = (
                event["ppu_master_clock"] - start["ppu_master_clock"]
            )
            windows.append({
                "position": len(windows),
                "cpu_cycles": event["cpu_cycle"] - start["cpu_cycle"],
                "master_clocks": master_delta,
                "master_clocks_div_12": master_delta / CPU_MASTER_DIVISOR,
                "mesen_master_clock_delta": (
                    event["master_clock"] - start["master_clock"]
                ),
                "begin": start,
                "end": event,
            })
            start = None
    return windows


def run_mesen(mesen, rom, lua_core, timeout):
    with tempfile.TemporaryDirectory(prefix="elya-mesen-") as temp_dir:
        result_path = Path(temp_dir) / "result.txt"
        script_path = Path(temp_dir) / "runner.lua"
        config = (
            "CFG = {out=%s, marker_address=0x0300, token_address=0x0200, "
            "token_count=96, sync_marker=%d, done_marker=%d}\n"
            % (lua_string(result_path), M_SYNC, M_DONE)
        )
        script_path.write_text(
            config + Path(lua_core).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        command = [
            str(mesen), "--testRunner", "--enableStdout", "--timeout=%d" % timeout,
            "--debug.scriptWindow.allowIoOsAccess=true",
            str(script_path), str(rom),
        ]
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=timeout + 30
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                "Mesen timed out\nstdout:\n%s\nstderr:\n%s"
                % ((error.stdout or b"")[-3000:], (error.stderr or b"")[-3000:])
            ) from error
        if completed.returncode != 0 or not result_path.exists():
            raise RuntimeError(
                "Mesen failed (%d)\nstdout:\n%s\nstderr:\n%s"
                % (completed.returncode, completed.stdout[-3000:],
                   completed.stderr[-3000:])
            )
        return parse_result(result_path), command


def build_report(raw, rom, expected, mesen, source_rom=None):
    expected_doc = json.loads(Path(expected).read_text(encoding="utf-8"))
    wanted = expected_doc["tokens"][1:]
    observed = list(bytes.fromhex(raw["tokens"]))[:len(wanted)]
    windows = pair_windows(raw["events"])
    cpu_counts = [window["cpu_cycles"] for window in windows]
    master_counts = [window["master_clocks"] for window in windows]
    mean_cycles = sum(cpu_counts) // len(cpu_counts) if cpu_counts else None
    return {
        "provenance": {
            "source_commit": SOURCE_COMMIT,
            "emulator": "Mesen",
            "emulator_release": MESEN_RELEASE,
            "mesen_executable": str(Path(mesen).resolve()),
            "mesen_executable_sha256": sha256(mesen),
            "official_windows_zip_sha256": MESEN_WINDOWS_SHA256,
            "rom": str(Path(rom)),
            "rom_sha256": sha256(rom),
            "source_rom": (
                str(Path(source_rom)) if source_rom else None
            ),
            "source_rom_sha256": sha256(source_rom) if source_rom else None,
            "rom_transform": (
                "96 KiB MMC5 PRG padded to 128 KiB; fixed bank 11 mirrored "
                "at bank 15 for Mesen reset mapping" if source_rom else None
            ),
            "expected": str(Path(expected)),
            "checkpoint": expected_doc["info"].get("weights_npz"),
        },
        "validation": {
            "status": raw["status"],
            "expected_tokens": wanted,
            "observed_tokens": observed,
            "matching_tokens": sum(a == b for a, b in zip(observed, wanted)),
            "token_count": len(wanted),
            "byte_identical": observed == wanted,
        },
        "measurement": {
            "marker_address": "0x0300",
            "cpu_cycles": cpu_counts,
            "master_clocks": master_counts,
            "total_cpu_cycles": sum(cpu_counts),
            "total_master_clocks": sum(master_counts),
            "mean_cpu_cycles": mean_cycles,
            "master_clock_ratio_is_12": all(
                master == cpu * CPU_MASTER_DIVISOR
                for cpu, master in zip(cpu_counts, master_counts)
            ),
            "mame_published_mean_cpu_cycles": MAME_PUBLISHED_MEAN,
            "delta_vs_mame_cycles": (
                mean_cycles - MAME_PUBLISHED_MEAN if mean_cycles is not None else None
            ),
            "delta_vs_mame_percent": (
                (mean_cycles / MAME_PUBLISHED_MEAN - 1) * 100
                if mean_cycles is not None else None
            ),
            "windows": windows,
            "raw_events": raw["events"],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rom")
    parser.add_argument("expected")
    parser.add_argument("--mesen", default=None)
    parser.add_argument("--source-rom")
    parser.add_argument("--lua", default=str(Path(__file__).with_suffix(".lua")))
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--output")
    args = parser.parse_args()

    mesen = args.mesen or find_mesen()
    ensure_portable_settings(mesen)
    raw, _ = run_mesen(mesen, args.rom, args.lua, args.timeout)
    report = build_report(
        raw, args.rom, args.expected, mesen, source_rom=args.source_rom
    )
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["validation"]["byte_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
