"""Find out how this Mac shows that the USB-C plug is connected.

  python -m plugsense.probe_link diff     # snapshot, you plug in by hand, shows what changed
  python -m plugsense.probe_link watch    # live: prints a line each time the power state changes

Run `diff` first with the cable in the Mac and the other end loose. Plug the loose
end into the power bank when asked. The keys that change are what the logger will
poll as the electrical "connected" label.
"""
import re
import subprocess
import sys
import time

NOISY = re.compile(r"Time|Counter|Timestamp|Uptime|Tick|Sequence|Statistics|Idle|kCG|HID", re.I)


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, errors="replace").stdout


def snapshot():
    lines = {}
    for line in run(["ioreg", "-l", "-w0"]).splitlines():
        line = line.strip(" |+-o")
        if "=" in line and not NOISY.search(line):
            key, _, val = line.partition("=")
            lines.setdefault(key.strip(), set()).add(val.strip()[:160])
    return lines


def power_state():
    batt = run(["ioreg", "-rn", "AppleSmartBattery"])
    out = {}
    for key in ("ExternalConnected", "IsCharging", "Amperage"):
        m = re.search(rf'"{key}" = (\S+)', batt)
        out[key] = m.group(1) if m else "?"
    out["usb_devices"] = str(run(["ioreg", "-p", "IOUSB"]).count("+-o") - 1)
    return out


if len(sys.argv) < 2 or sys.argv[1] not in ("diff", "watch"):
    sys.exit(__doc__)

if sys.argv[1] == "diff":
    input("Cable in the Mac, other end NOT plugged in. Press Enter...")
    a = snapshot()
    input("Now plug the other end into the power bank, wait 3 s, press Enter...")
    b = snapshot()
    changed = [(k, a.get(k), b.get(k)) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
    print(f"\n{len(changed)} keys changed:")
    for k, before, after in changed[:80]:
        print(f"  {k}\n      before: {sorted(before or [])[:3]}\n      after:  {sorted(after or [])[:3]}")
    if not changed:
        print("  nothing: this Mac does not see the power bank over this cable")
else:
    last, t0 = None, time.monotonic()
    print("watching (Ctrl+C to stop). Plug and unplug a few times.")
    while True:
        s = power_state()
        if s != last:
            print(f"{time.monotonic() - t0:7.2f}s  {s}")
            last = s
        time.sleep(0.05)
