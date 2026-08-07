#!/usr/bin/env python3
"""NES/Famicom cycle-measurement instrument.

Boots a .nes image in MAME 0.277's `nes` driver headless and timestamps every
write the ROM makes to the marker port $0300 using a Lua memory write tap on
manager.machine.time (attoseconds).  Cycle counts come out of the attosecond
deltas:

    cycles = round(delta_as * CPU_HZ / 1e18)

CPU_HZ is DERIVED, not assumed - see derive_clock().  MAME instantiates the
NTSC 2A03 with an integer Hz clock, truncating the true 21477272/12 =
1789772.667, so the emulator's own timebase is 1789772 Hz exactly.

Two traps this runner is built around, both of which cost a day previously on
the Genesis version of this work:

  * MAME's Lua bindings garbage-collect a tap subscription if you do not hold
    a reference to the handle.  Dropping it silently stops every callback,
    which looks EXACTLY like the ROM hanging.  Hence the global KEEP table.
  * Never poll with emu.register_periodic.  It drops MAME to ~0.07x realtime.
    Everything here is driven from the write tap.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

MAME = "/usr/games/mame"

# Candidate CPU clocks tested by derive_clock().
CLOCK_CANDIDATES = {
    "1789772 (MAME truncated)": 1789772.0,
    "21477272/12 (true NTSC)": 21477272.0 / 12.0,
    "1789773 (rounded)": 1789773.0,
    "1789800 (folklore)": 1789800.0,
}
CPU_HZ = 1789772.0

M_BEGIN, M_END, M_SYNC, M_DONE = 1, 2, 254, 255

LUA = r'''
local mac = manager.machine
local cpu = mac.devices[":maincpu"]
local sp  = cpu.spaces["program"]

-- MAME's Lua bindings garbage-collect tap subscriptions if nothing holds a
-- reference.  Losing the handle silently stops all callbacks and looks
-- exactly like the ROM hanging.  KEEP is global on purpose.
KEEP = {}

local function t2as(t) return t.seconds * 1e18 + t.attoseconds end

local ev = {}
local done = false

local function finish(status)
  if done then return end
  done = true
  local f = io.open(CFG.out, "w")
  f:write("status=" .. status .. "\n")
  f:write("events=" .. table.concat(ev, " ") .. "\n")
  -- optional memory dumps: CFG.dump = { {name, addr, len}, ... }
  for _, d in ipairs(CFG.dump or {}) do
    local b = {}
    for i = 0, d[3] - 1 do b[#b+1] = string.format("%02X", sp:read_u8(d[2] + i)) end
    f:write("dump." .. d[1] .. "=" .. table.concat(b, "") .. "\n")
  end
  f:close()
  mac:exit()
end

KEEP[#KEEP+1] = sp:install_write_tap(CFG.marker, CFG.marker, "elya_nes_marker",
  function(offset, data, mask)
    local v = data & 0xFF
    if v == 254 then
      ev = {}                       -- MAME's boot resets the console once;
                                    -- a fresh SYNC discards the earlier run
    elseif v == 255 then
      ev[#ev+1] = string.format("255:%.0f", t2as(mac.time))
      finish("OK")
    else
      ev[#ev+1] = string.format("%d:%.0f", v, t2as(mac.time))
    end
    return data
  end)

KEEP[#KEEP+1] = emu.add_machine_stop_notifier(function()
  if not done then
    local f = io.open(CFG.out, "w")
    f:write("status=TIMEOUT\nevents=" .. table.concat(ev, " ") .. "\n")
    f:close()
  end
end)
'''


def run_rom(rom, seconds=60, marker=0x0300, dump=None, timeout=1200):
    """Boot `rom` and return {'status':..., 'events':[(val, as), ...], dumps}."""
    outfd, outpath = tempfile.mkstemp(suffix=".txt")
    os.close(outfd)
    luafd, luapath = tempfile.mkstemp(suffix=".lua")
    os.close(luafd)
    dumps = dump or []
    dl = ", ".join("{%r, 0x%04X, %d}" % (n, a, l) for n, a, l in dumps)
    with open(luapath, "w") as f:
        f.write("CFG = { marker=0x%04X, out=%r, dump={%s} }\n"
                % (marker, outpath, dl))
        f.write(LUA)
    cmd = [MAME, "nes", "-cart", rom, "-autoboot_script", luapath,
           "-sound", "none", "-video", "none", "-nothrottle",
           "-window", "-skip_gameinfo", "-seconds_to_run", str(seconds)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    res = {}
    with open(outpath) as f:
        for line in f:
            if "=" in line:
                k, v = line.rstrip("\n").split("=", 1)
                res[k] = v
    os.unlink(outpath)
    os.unlink(luapath)
    if not res:
        raise SystemExit("no result from MAME.\nstdout:\n%s\nstderr:\n%s"
                         % (p.stdout[-3000:], p.stderr[-3000:]))
    ev = []
    for tok in res.get("events", "").split():
        v, t = tok.split(":")
        ev.append((int(v), float(t)))
    res["events"] = ev
    return res


def pair_windows(events):
    """Fold a flat marker stream into [(delta_as)] for each BEGIN/END pair."""
    out, start = [], None
    for v, t in events:
        if v == M_BEGIN:
            start = t
        elif v == M_END and start is not None:
            out.append(t - start)
            start = None
    return out


def derive_clock(delta_as, known_cycles):
    """Report every candidate clock against a payload of known length."""
    rows = []
    for name, hz in CLOCK_CANDIDATES.items():
        rows.append((name, hz, delta_as * hz / 1e18))
    rows.sort(key=lambda r: abs(r[2] - known_cycles))
    return rows


def cycles(delta_as, hz=CPU_HZ):
    return int(round(delta_as * hz / 1e18))
