"""Gate tests G4 and G5: can this arm repeat your plug-in on its own, and from how far off?

  python -m plugsense.gate --mic-device 1 --replays 3      # short first try
  python -m plugsense.gate --mic-device 1 --probe          # 20 replays, then pushes from known offsets

The follower goes to the start pose. You bring the leader to match, then plug in (Enter)
and pull out (Enter) once. After that the arm works alone: for every replay it returns to
the start pose, repeats the last stretch of your push, checks whether it reached the pose
where you had the plug seated, and repeats your pull-out.

KEEP A HAND NEAR THE POWER SWITCH. At the end it waits for you to hold the arm.
Every push is saved under data/pushes/ with commands, servo positions, loads and audio.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time
import wave
from pathlib import Path

import numpy as np

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

from plugsense import demo
from plugsense.arm import ARM, ELBOW, PAN, Arm
from plugsense.attempt_log import FOLLOWER_PORT, LEADER_PORT, RATE, Mic

logging.getLogger("lerobot.robots.utils").setLevel(logging.ERROR)  # hide the per-step clamp warnings during teleop

STEPS = [1, 2, 4, 6]


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


def pull_out(arm, skill, offset=None) -> None:
    """Repeat your pull-out; if the arm still thinks the plug is in, hand over to you until it is out."""
    out = demo.run_pull(arm, skill, offset)
    while not out:
        arm.sync_goal()  # stop pulling while you work on it
        said = input("  The arm thinks the plug is still in. If it is already out type o, otherwise pull it out by hand. Then press Enter. ")
        out = demo.is_free(arm, skill, offset, told_out=True, say=True) if said.strip().lower() == "o" else demo.run_pull(arm, skill, offset)


def cycle(arm, skill, home, folder: Path, name: str, mic, offset=None, extra: dict | None = None, ask: bool = False,
          after_push=None) -> str:
    """One full attempt: start pose, your push, save, your pull-out."""
    arm.goto(home, speed=300.0)
    r = demo.run_push(arm, skill, offset)
    outcome = r["outcome"]
    if after_push:
        after_push()  # e.g. photograph the display while the plug is still in
    seen = input(f"  {name:<16} arm depth {r['frac']:.2f}. Is the display lit? [y/n] ").strip().lower() if ask else ""
    if seen in ("y", "n"):
        outcome = "seated" if seen == "y" else "jammed"
        r["outcome"], r["display"] = outcome, seen == "y"
    else:
        print(f"  {name:<16} {outcome:<7} depth {r['frac']:.2f}  {r['seconds']:.1f} s")
    save_push(folder / name, r, {"offset": None if offset is None else np.asarray(offset).tolist(), **(extra or {})}, mic)
    pull_out(arm, skill, offset)
    return outcome


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mic-device", default=None)
    ap.add_argument("--replays", type=int, default=20)
    ap.add_argument("--probe", action="store_true", help="also run G5: pushes from known offsets")
    ap.add_argument("--near", type=float, default=45.0, help="the push starts where the busiest joint (twist excluded) was this many counts from the seat, roughly 8 mm")
    ap.add_argument("--lead-in", type=float, default=60.0, help="seconds of your approach to repeat before the last stretch; the default repeats all of it")
    ap.add_argument("--ask", action="store_true", help="after each push, ask you whether the display is lit and use that as the result")
    ap.add_argument("--spot", default="", help="saved with every push")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "data" / "pushes"))
    a = ap.parse_args()

    robot = SO101Follower(SO101FollowerConfig(port=FOLLOWER_PORT, id="follower", max_relative_target=10.0))
    leader = SO101Leader(SO101LeaderConfig(port=LEADER_PORT, id="leader"))
    robot.connect()
    leader.connect()
    arm = Arm.from_robot(robot)
    mic = Mic(int(a.mic_device) if a.mic_device and a.mic_device.isdigit() else a.mic_device) if a.mic_device else None
    session = Path(a.out) / time.strftime("%Y%m%d-%H%M%S")
    home = skill = None
    try:
        home = demo.get_home(robot, leader, arm)
        skill = demo.teach(robot, leader, arm, home, a.near, a.lead_in)
        if skill is None:
            return
        session.mkdir(parents=True, exist_ok=True)
        skill.save(session / "skill.json")
        demo.save_taught(skill, session / "taught.json")
        input("\nLower the leader arm onto the desk. The follower now works alone. Press Enter to start.")
        arm.sync_goal()

        print(f"\nG4: {a.replays} replays of your push")
        g4 = [cycle(arm, skill, home, session, f"g4-{i:02d}", mic, extra={"spot": a.spot}, ask=a.ask) for i in range(a.replays)]
        seated = g4.count("seated")
        print(f"G4 result: {seated} of {a.replays} seated  ->  {'PASS' if seated * 2 >= a.replays else 'FAIL'} (pass is half or more)")

        if a.probe:
            print("\nG5: your push started from known offsets (pan = sideways, elbow = toward/away)")
            tally: dict[int, list[str]] = {}
            for joint, jname in ((PAN, "pan"), (ELBOW, "elbow")):
                for step in STEPS:
                    for sign in (+1, -1):
                        off = np.zeros(len(ARM))
                        off[joint] = sign * step
                        tally.setdefault(step, []).append(
                            cycle(arm, skill, home, session, f"g5-{jname}{sign * step:+d}", mic, off, {"spot": a.spot}, a.ask))
            print("\nG5 result (seated out of 4 pushes at each offset size):")
            for step, outs in tally.items():
                print(f"  {step} counts off     {outs.count('seated')} of {len(outs)}")
        print(f"\nsaved under {session}")
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        try:
            arm.sync_goal()  # stop pushing or pulling
            if home is not None and input("\nReturn to the start pose first? [Y/n] ").strip().lower() != "n":
                if skill is not None and not demo.is_free(arm, skill, say=True):
                    input("The arm is still at the port. Make sure the plug is out (pull it by hand if needed), then press Enter.")
                    arm.sync_goal()
                arm.goto(home, speed=200.0)
            input("HOLD THE FOLLOWER ARM with your hand, then press Enter. The motors will let go.")
        except (EOFError, KeyboardInterrupt):
            pass
        if mic:
            mic.close()
        leader.disconnect()
        robot.disconnect()


if __name__ == "__main__":
    main()
