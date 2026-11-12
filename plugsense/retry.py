"""Push, look at the display, and try again if it stayed dark.

  python -m plugsense.retry --mic-device 1 --trials 10
  python -m plugsense.retry --mic-device 1 --trials 10 --reuse     # skip teaching, use the last taught push

You plug in and pull out once by teleop. Then for every trial the arm starts from the start
pose and repeats your push. The front camera reads the power bank display: lit means the plug
latched. If it stayed dark the arm pulls out and pushes again, with a small sideways nudge on
later tries. A trial ends at the first lit display or after --tries pushes.

Every push is saved under data/retry/<time>/ with servo positions, loads, sound, a photo and
the display reading, so the same runs give labelled data for the sound and load check.
KEEP A HAND NEAR THE POWER SWITCH.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from plugsense import demo, display
from plugsense.arm import ARM, ELBOW, PAN

logging.getLogger("lerobot.robots.utils").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parent.parent / "data"
LAST_SKILL = ROOT / "skill_last.json"
JOINT = {"pan": PAN, "elbow": ELBOW}


def parse_nudges(text: str) -> list[np.ndarray]:
    """'0,0,pan+3,pan-3' -> one joint offset (counts) per try."""
    out = []
    for part in text.split(","):
        off = np.zeros(len(ARM))
        part = part.strip()
        if part not in ("", "0"):
            name = "pan" if part.startswith("pan") else "elbow"
            off[JOINT[name]] = float(part[len(name):])
        out.append(off)
    return out


def run_trial(arm, skill, home, read_share, pull, nudges, tries: int, base=None, on_push=None, lit_wait: float = 4.0) -> dict:
    """One trial: up to `tries` pushes from the start pose. Returns which try lit the display (0 = none).

    read_share() -> share of bright pixels in the display box now.  pull(offset) gets the plug out.
    base: an offset added to every push on purpose, to test recovery from a misplaced start.
    """
    base = np.zeros(len(ARM)) if base is None else np.asarray(base, float)
    t_start, lit_on, pushes = time.monotonic(), 0, []
    for k in range(tries):
        offset = base + nudges[min(k, len(nudges) - 1)]
        arm.goto(home, speed=300.0)
        dark = read_share()
        r = demo.run_push(arm, skill, offset)
        t_push = time.monotonic()
        after = read_share()
        while after - dark < display.LIT_SHARE and time.monotonic() - t_push < lit_wait:
            time.sleep(0.2)  # the display can take a few seconds to turn on after the plug goes in
            after = read_share()
        r.update(attempt=k + 1, offset=offset.tolist(), share_before=round(dark, 4), share_after=round(after, 4),
                 lit=bool(after - dark >= display.LIT_SHARE), t_seen=time.monotonic() - t_start)
        if on_push:
            on_push(r)
        pushes.append({key: r[key] for key in ("attempt", "offset", "lit", "frac", "share_before", "share_after")})
        if r["lit"]:
            lit_on = k + 1
            break
        pull(offset)
    seconds = time.monotonic() - t_start
    if lit_on:
        pull(offset)
    return {"lit_on": lit_on, "seconds": round(seconds, 2), "pushes": pushes}


def summary(trials: list[dict], tries: int) -> str:
    n = len(trials)
    if not n:
        return "no trials"
    first = sum(t["lit_on"] == 1 for t in trials)
    done = [t for t in trials if t["lit_on"]]
    lines = [f"first try: {first} of {n}", f"within {tries} tries: {len(done)} of {n}"]
    if done:
        lines.append(f"time to a lit display, successful trials: mean {np.mean([t['seconds'] for t in done]):.1f} s")
    return "\n".join(lines)


def main():
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

    from plugsense.arm import Arm
    from plugsense.attempt_log import FOLLOWER_PORT, LEADER_PORT, Mic
    from plugsense.cameras import Camera
    from plugsense.gate import pull_out, save_push

    ap = argparse.ArgumentParser()
    ap.add_argument("--mic-device", default=None)
    ap.add_argument("--front", type=int, default=0)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--tries", type=int, default=3, help="pushes allowed per trial")
    ap.add_argument("--nudges", default="0,0,pan+3,pan-3", help="offset for each try, in encoder counts")
    ap.add_argument("--lit-wait", type=float, default=4.0, help="seconds to keep watching the display after a push before calling it dark")
    ap.add_argument("--base", default="0", help="offset added to every push on purpose, e.g. pan+6")
    ap.add_argument("--reuse", action="store_true", help="use the last taught push instead of teaching again")
    ap.add_argument("--near", type=float, default=45.0)
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    robot = SO101Follower(SO101FollowerConfig(port=FOLLOWER_PORT, id="follower", max_relative_target=10.0))
    leader = SO101Leader(SO101LeaderConfig(port=LEADER_PORT, id="leader"))
    robot.connect()
    leader.connect()
    arm = Arm.from_robot(robot)
    cam = Camera(a.front, 1280, 720, name="front")
    mic = Mic(int(a.mic_device) if a.mic_device and a.mic_device.isdigit() else a.mic_device) if a.mic_device else None
    session = ROOT / "retry" / time.strftime("%Y%m%d-%H%M%S")
    home = skill = None
    trials = []
    try:
        home = demo.get_home(robot, leader, arm)
        if a.reuse and LAST_SKILL.exists():
            skill = demo.Skill.load(LAST_SKILL)
            arm.sync_goal()
        else:
            skill = demo.teach(robot, leader, arm, home, a.near)
            if skill is None:
                return
            skill.save(LAST_SKILL)
            input("\nLower the leader arm onto the desk. The follower now works alone. Press Enter to start.")
            arm.sync_goal()
        session.mkdir(parents=True, exist_ok=True)
        skill.save(session / "skill.json")
        demo.save_taught(skill, session / "taught.json")
        nudges, base = parse_nudges(a.nudges), parse_nudges(a.base)[0]

        def read_share():
            return display.share(cam.fresh(timeout=2.0))

        def wait_dark(limit=15.0):  # the display stays lit for up to 10 s after unplugging
            t0 = time.monotonic()
            while read_share() >= display.LIT_SHARE / 2 and time.monotonic() - t0 < limit:
                time.sleep(0.5)

        for i in range(a.trials):
            arm.goto(home, speed=300.0)
            wait_dark()

            def on_push(r, i=i):
                name = f"t{i:02d}-try{r['attempt']}"
                cv2.imwrite(str(session / f"{name}.jpg"), cam.fresh(timeout=2.0))
                print(f"  {name}: display {'LIT ' if r['lit'] else 'dark'} (bright share {r['share_before']:.3f} -> {r['share_after']:.3f}), depth {r['frac']:.2f}")
                save_push(session / name, dict(r), {"trial": i, "note": a.note}, mic)

            t = run_trial(arm, skill, home, read_share, lambda off: pull_out(arm, skill, off), nudges, a.tries, base, on_push, a.lit_wait)
            trials.append(t)
            print(f"trial {i + 1}/{a.trials}: " + (f"lit on try {t['lit_on']} after {t['seconds']:.1f} s" if t["lit_on"] else f"not lit in {a.tries} tries"))
            (session / "trials.json").write_text(json.dumps({"tries": a.tries, "nudges": a.nudges, "base": a.base, "note": a.note, "trials": trials}, indent=2))
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        print("\n" + summary(trials, a.tries) + f"\nsaved under {session}")
        try:
            arm.sync_goal()
            if home is not None and input("\nReturn to the start pose first? [Y/n] ").strip().lower() != "n":
                if skill is not None and not demo.is_free(arm, skill, say=True):
                    input("The arm is still at the port. Make sure the plug is out (pull it by hand if needed), then press Enter.")
                    arm.sync_goal()
                arm.goto(home, speed=200.0)
            input("HOLD THE FOLLOWER ARM with your hand, then press Enter. The motors will let go.")
        except (EOFError, KeyboardInterrupt):
            pass
        cam.close()
        if mic:
            mic.close()
        leader.disconnect()
        robot.disconnect()


if __name__ == "__main__":
    main()
