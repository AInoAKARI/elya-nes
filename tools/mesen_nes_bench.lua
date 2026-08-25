-- Mesen 2 headless marker capture for the Elya NES benchmark.
-- CFG is prepended by tools/mesen_nes_bench.py.

local events = {}
local done = false
local frames = 0

local function append_line(line)
  local f = assert(io.open(CFG.out, "a"))
  f:write(line .. "\n")
  f:close()
end

append_line("status=RUNNING")
local rom_info = emu.getRomInfo()
append_line(string.format("rom=%s,%s", rom_info.name, rom_info.fileSha1Hash))
append_line(string.format("reset_vector=%02X%02X",
  emu.read(0xFFFD, emu.memType.nesDebug),
  emu.read(0xFFFC, emu.memType.nesDebug)))

local function write_result(status)
  if done then return end
  done = true

  local f = assert(io.open(CFG.out, "w"))
  f:write("status=" .. status .. "\n")
  for _, event in ipairs(events) do
    f:write(string.format("event=%d,%.0f,%.0f,%.0f\n",
      event.value, event.cpu_cycle, event.master_clock,
      event.ppu_master_clock))
  end

  local bytes = {}
  for address = CFG.token_address, CFG.token_address + CFG.token_count - 1 do
    bytes[#bytes + 1] = string.format("%02X",
      emu.read(address, emu.memType.nesDebug))
  end
  f:write("tokens=" .. table.concat(bytes, "") .. "\n")
  f:close()
end

local function marker_write(_, value)
  local state = emu.getState()
  local event = {
    value = value,
    cpu_cycle = state["cpu.cycleCount"],
    master_clock = state["masterClock"],
    ppu_master_clock = state["ppu.masterClock"]
  }

  if value == CFG.sync_marker then
    -- Mesen can reset the machine once during initial load.  The last SYNC is
    -- the authoritative start of the benchmark stream.
    events = {}
  elseif value == CFG.done_marker then
    events[#events + 1] = event
    write_result("OK")
    emu.stop(0)
  else
    events[#events + 1] = event
    append_line(string.format("progress=%d,%.0f,%.0f,%.0f",
      event.value, event.cpu_cycle, event.master_clock,
      event.ppu_master_clock))
  end
end

emu.addMemoryCallback(marker_write, emu.callbackType.write,
  CFG.marker_address, CFG.marker_address)

local function heartbeat()
  frames = frames + 1
  if frames % 60 == 0 then
    local state = emu.getState()
    append_line(string.format("heartbeat=%d,%.0f,%.0f,%04X",
      frames, state["cpu.cycleCount"], state["masterClock"], state["cpu.pc"]))
  end
end

emu.addEventCallback(heartbeat, emu.eventType.endFrame)
