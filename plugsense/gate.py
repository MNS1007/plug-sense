"""Gate tests G4 and G5: can this arm reseat the plug on its own, and from how far off?

  python -m plugsense.gate --mic-device 1                 # G4: 20 replays from the taught pose
  python -m plugsense.gate --mic-device 1 --probe         # G4, then G5: pushes from known offsets

You teach two poses by teleop: hover (plug about 10 mm straight above the port) and
seat (plug fully in). The arm then works alone. KEEP ONE HAND NEAR THE POWER SWITCH.
At the end it waits for you to hold the arm before it lets go.

Every push is saved under data/pushes/ with servo positions, loads and contact audio.
The G5 pushes carry the offset they started from, which is the training label for
the network that tells which way a jam was off.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
import wave
from pathlib import Path

import logging

import numpy as np

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

from plugsense.arm import ARM, ELBOW, PAN, Arm
from plugsense.attempt_log import FOLLOWER_PORT, LEADER_PORT, RATE, Mic

logging.getLogger("lerobot.robots.utils").setLevel(logging.ERROR)  # hide the per-step clamp warnings during teleop

STEPS = [1, 2, 4, 6]


def teleop_until_enter(robot, leader, prompt: str) -> None:
    print(prompt)
    done = threading.Event()
    threading.Thread(target=lambda: (sys.stdin.readline(), done.set()), daemon=True).start()
    while not done.is_set():
        t0 = time.monotonic()
        robot.send_action(leader.get_action())
        time.sleep(max(0.0, 1 / 30 - (time.monotonic() - t0)))


def pull_out(arm, hover, seat, lead: float = 0.3) -> None:
    """Pull the plug out; if it stays stuck, wait for a hand and carry on from above the port."""
    if arm.retract(hover, seat, lead_max=max(0.4, 1.3 * lead)):
        return
    input("  The arm could not pull the plug out. Free it by hand, then press Enter.")
    arm.sync_goal()
    arm.goto(hover - 0.6 * (seat - hover), speed=200.0)


def save_push(folder: Path, result: dict, meta: dict, mic) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    rows = result.pop("rows")
    if rows:
        with open(folder / "signals.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    meta = {**meta, **result}
    if mic:
        audio, audio_t0 = mic.cut(result["t_start"] - 0.2, time.monotonic())
        with wave.open(str(folder / "audio.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(audio.tobytes())
        meta.update(audio_t0=audio_t0, audio_rate=RATE, audio_peak=int(np.abs(audio).max()) if len(audio) else 0)
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mic-device", default=None)
    ap.add_argument("--replays", type=int, default=20)
    ap.add_argument("--probe", action="store_true", help="also run G5: pushes from known offsets")
    ap.add_argument("--wiggle", type=float, default=3.0, help="wiggle size in encoder counts for the wiggle runs")
    ap.add_argument("--lead", type=float, default=1.0, help="push strength: how far the command may run ahead of the arm, as a fraction of the hover-to-seat distance")
    ap.add_argument("--spot", default="", help="grid spot number, saved with every push")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "data" / "pushes"))
    ap.add_argument("--rock", type=float, default=40.0, help="wrist rocking size for pull-outs, in encoder counts")
    a = ap.parse_args()

    robot = SO101Follower(SO101FollowerConfig(port=FOLLOWER_PORT, id="follower", max_relative_target=10.0))
    leader = SO101Leader(SO101LeaderConfig(port=LEADER_PORT, id="leader"))
    robot.connect()
    leader.connect()
    arm = Arm.from_robot(robot)
    arm.rock = a.rock
    mic = Mic(int(a.mic_device) if a.mic_device and a.mic_device.isdigit() else a.mic_device) if a.mic_device else None
    session = Path(a.out) / time.strftime("%Y%m%d-%H%M%S")

    try:
        teleop_until_enter(robot, leader, "\n1/2  Hold the plug about 10 mm straight ABOVE the port. Press Enter.")
        hover = arm.pos()
        teleop_until_enter(robot, leader, "2/2  Now SEAT the plug fully. Press Enter while it is seated.")
        seat = arm.pos()
        line = seat - hover
        print("hover -> seat, counts per joint:", dict(zip(ARM, line.astype(int))))
        if abs(line[4]) > 30:
            print(f"NOTE: the wrist twisted {abs(line[4]) * 0.088:.0f} degrees between hover and seat. "
                  "Teach hover with the same twist as the seated pose if you can.")
        if np.abs(line).max() > 90:
            print("NOTE: hover looks far from the seat. About 10 mm above the port is enough.")
        if np.abs(line).max() < 8:
            print("These two poses are almost the same. Start again with a clearer hover.")
            return
        session.mkdir(parents=True, exist_ok=True)
        (session / "teach.json").write_text(json.dumps(
            {"motors": ARM, "hover": hover.tolist(), "seat": seat.tolist(), "spot": a.spot}, indent=2))

        input("\nLet go of the leader arm. The follower now moves by itself. Press Enter to start.")
        pull_out(arm, hover, seat, a.lead)

        def push(name: str, offset=None, wiggle=0.0) -> str:
            arm.approach(hover + (0 if offset is None else offset), seat + (0 if offset is None else offset))
            r = arm.descend(hover, seat, offset=offset, wiggle=wiggle, lead_max=a.lead)
            meta = {"spot": a.spot, "offset": None if offset is None else offset.tolist(), "wiggle": wiggle,
                    "hover": hover.tolist(), "seat": seat.tolist(), "lead": a.lead}
            outcome = r["outcome"]
            print(f"  {name:<22} {outcome:<7} depth {r['frac']:.2f}  {r['seconds']:.1f} s")
            save_push(session / name, r, meta, mic)
            pull_out(arm, hover, seat, a.lead)
            return outcome

        print(f"\nG4: {a.replays} replays from the taught pose")
        g4 = [push(f"g4-{i:02d}") for i in range(a.replays)]
        seated = g4.count("seated")
        print(f"G4 result: {seated} of {a.replays} seated  ->  {'PASS' if seated * 2 >= a.replays else 'FAIL'} (pass is half or more)")

        if a.probe:
            print("\nG5: pushes from known offsets (pan = sideways, elbow = toward/away)")
            tally: dict[str, list[str]] = {}
            for joint, jname in ((PAN, "pan"), (ELBOW, "elbow")):
                for step in STEPS:
                    for sign in (+1, -1):
                        off = np.zeros(len(ARM))
                        off[joint] = sign * step
                        for wig in (0.0, a.wiggle):
                            key = f"{step} counts, {'wiggle' if wig else 'straight'}"
                            name = f"g5-{jname}{sign * step:+d}-{'w' if wig else 's'}"
                            tally.setdefault(key, []).append(push(name, off, wig))
            print("\nG5 result (seated out of 4 pushes at each offset size):")
            for key, outs in tally.items():
                print(f"  {key:<22} {outs.count('seated')} of {len(outs)}")
        print(f"\nsaved under {session}")
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        try:
            input("\nHOLD THE ARM with your hand, then press Enter. The motors will let go.")
        except (EOFError, KeyboardInterrupt):
            pass
        if mic:
            mic.close()
        leader.disconnect()
        robot.disconnect()


if __name__ == "__main__":
    main()
