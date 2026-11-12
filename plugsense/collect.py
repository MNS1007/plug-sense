"""Self-collection at a taught location: the arm labels its own images and pushes.

  python -m plugsense.collect --mic-device 1 --session 1

You plug in and pull out once by teleop (as in plugsense.gate). The arm then works alone:
  - photographs many poses around the start of your push; each is labelled with the joint
    offset back to that start pose, which needs no human and no calibration
  - repeats your push from a few known offsets, saving sound and servo load (labels for
    the network that tells which way a jam was off)
  - repeats your push once with no offset, to check the power bank has not moved
  - returns to the start pose and photographs the scene
Sessions 4, 8, 15, 17, 23 and 26 are evaluation only: fewer samples, never trained on.
Re-grip the plug or re-teach between sessions so the data covers small changes.

Data goes to data/collect/<time>-spot<NN>/.  KEEP A HAND NEAR THE POWER SWITCH.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path

import numpy as np

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

from plugsense import demo
from plugsense.arm import ARM, ELBOW, LIFT, PAN, Arm
from plugsense.attempt_log import FOLLOWER_PORT, LEADER_PORT, Mic
from plugsense.cameras import Camera
from plugsense.gate import cycle

logging.getLogger("lerobot.robots.utils").setLevel(logging.ERROR)

HELD_OUT = {4, 8, 15, 17, 23, 26}
ROOT = Path(__file__).resolve().parent.parent / "data"


def sample_offsets(n: int, lateral: float, rng) -> list[tuple[np.ndarray, float]]:
    """Random poses around the push start: sideways (pan), toward/away (elbow, lift) and height along the push."""
    out = []
    for _ in range(n):
        off = np.zeros(len(ARM))
        off[PAN] = rng.uniform(-lateral, lateral)
        off[ELBOW] = rng.uniform(-lateral, lateral)
        off[LIFT] = rng.uniform(-lateral / 3, lateral / 3)
        out.append((np.round(off), float(rng.uniform(-0.6, 0.0))))  # only at or above the push start, so an offset pose cannot touch the power bank
    return out


def push_offsets(n: int, rng) -> list[np.ndarray]:
    choices = [(j, s * k) for j in (PAN, ELBOW) for k in (2, 4, 6, 9) for s in (1, -1)]
    out = []
    for i in rng.choice(len(choices), size=min(n, len(choices)), replace=False):
        off = np.zeros(len(ARM))
        off[choices[i][0]] = choices[i][1]
        out.append(off)
    return out


def collect_session(arm, cams, mic, session: int, skill, home, a, rng) -> dict:
    held = session in HELD_OUT
    n = a.held_samples if held else a.samples
    folder = ROOT / "collect" / f"{time.strftime('%Y%m%d-%H%M%S')}-spot{session:02d}"
    (folder / "img").mkdir(parents=True, exist_ok=True)
    skill.save(folder / "skill.json")
    line = np.append(skill.line, 0.0)  # no twist along the way
    arm.goto(home, speed=a.speed)

    rows = []
    for i, (off, t) in enumerate(sample_offsets(n, a.lateral, rng)):
        target = skill.hover + off + t * line
        arm.approach(target, target + line, speed=a.speed)
        measured = arm.pos()
        row = {"i": i, "t_line": round(t, 3)}
        row.update({f"cmd_{m}": v for m, v in zip(ARM, arm.goal)})
        row.update({f"meas_{m}": v for m, v in zip(ARM, measured)})
        for name, cam in cams.items():
            file = f"img/{i:03d}_{name}.jpg"
            row[name] = file if cam.save(folder / file) else ""
        rows.append(row)
        print(f"\r  sample {i + 1}/{n}", end="", flush=True)
    print()
    with open(folder / "samples.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def push(name, offset=None):
        def snap():
            time.sleep(0.8)  # the display takes a moment to turn red
            cams["front"].save(folder / f"{name}_front_after.jpg")

        outcome = cycle(arm, skill, home, folder / "pushes", name, mic, offset, {"session": session}, ask=a.ask, after_push=snap)
        time.sleep(a.dark_wait if outcome == "seated" else 0.2)
        return outcome

    pushes = {}
    if not held:
        for k, off in enumerate(push_offsets(a.pushes, rng)):
            j = "pan" if off[PAN] else "elbow"
            name = f"{k:02d}-{j}{int(off[PAN] or off[ELBOW]):+d}"
            pushes[name] = push(name, off)
    check = push("check")
    if check != "seated":
        print("  check push did not seat (blind pushes miss about half the time; the photos are still used)")

    arm.goto(home, speed=a.speed)
    time.sleep(0.3)
    cams["front"].save(folder / "scene_front.jpg")
    meta = {"spot": session, "held_out": held, "motors": ARM, "hover": skill.hover.tolist(), "seat": skill.seat.tolist(),
            "home": home.tolist(), "samples": len(rows), "pushes": pushes, "check": check}
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mic-device", default=None)
    ap.add_argument("--front", type=int, default=0, help="front camera index")
    ap.add_argument("--wrist", type=int, default=1, help="wrist camera index")
    ap.add_argument("--samples", type=int, default=60)
    ap.add_argument("--held-samples", type=int, default=20)
    ap.add_argument("--pushes", type=int, default=8)
    ap.add_argument("--lateral", type=float, default=24.0, help="largest sideways offset, in encoder counts (about 8 mm)")
    ap.add_argument("--speed", type=float, default=400.0)
    ap.add_argument("--near", type=float, default=45.0)
    ap.add_argument("--dark-wait", type=float, default=11.0, help="seconds to wait after a seated push so the display goes dark")
    ap.add_argument("--no-ask", dest="ask", action="store_false", help="do not ask whether the display is lit; use the arm's depth instead (unreliable)")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    cams = {"front": Camera(a.front, name="front"), "wrist": Camera(a.wrist, mjpg=True, name="wrist")}
    robot = SO101Follower(SO101FollowerConfig(port=FOLLOWER_PORT, id="follower", max_relative_target=10.0))
    leader = SO101Leader(SO101LeaderConfig(port=LEADER_PORT, id="leader"))
    robot.connect()
    leader.connect()
    arm = Arm.from_robot(robot)
    mic = Mic(int(a.mic_device) if a.mic_device and a.mic_device.isdigit() else a.mic_device) if a.mic_device else None
    done, home = [], None
    try:
        home = demo.get_home(robot, leader, arm)
        while True:
            raw = input(f"\ndone: {done}\nSession number? (1-30, q = quit) ").strip().lower()
            if raw == "q":
                break
            if not raw.isdigit() or not 1 <= int(raw) <= 30:
                continue
            skill = demo.teach(robot, leader, arm, home, a.near)
            if skill is None:
                continue
            input("Lower the leader arm onto the desk. Press Enter and the follower works alone.")
            arm.sync_goal()
            meta = collect_session(arm, cams, mic, int(raw), skill, home, a, rng)
            done.append(int(raw))
            print(f"session {raw} saved ({'evaluation' if meta['held_out'] else 'training'}). Arm is at the start pose.")
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        try:
            input("\nHOLD THE FOLLOWER ARM with your hand, then press Enter. The motors will let go.")
        except (EOFError, KeyboardInterrupt):
            pass
        if mic:
            mic.close()
        for cam in cams.values():
            cam.close()
        leader.disconnect()
        robot.disconnect()


if __name__ == "__main__":
    main()
