"""Self-collection: you seat the plug once per spot, the arm labels its own images.

  python -m plugsense.collect --mic-device 1

Once per session you teach a HOME pose (arm raised, clear of the power bank in the
front camera). Then, for every grid spot:
  1. type the spot number
  2. seat the plug fully by teleop, coming down from above, press Enter
  3. (the hover pose is taken from your approach, about 1 cm before the seat)
  4. let go of the leader; the arm pulls out and works alone:
       - photographs many poses around hover; each is labelled with the joint offset
         back to hover, which needs no human and no calibration
       - pushes from a few known offsets, saving sound and servo load (labels for the
         network that tells which way a jam was off)
       - pushes once from the taught pose to check the power bank has not moved
       - returns home and photographs the scene for the coarse network
Held-out spots get fewer samples and are never trained on.

Data goes to data/collect/<time>-spot<NN>/.  KEEP A HAND NEAR THE POWER SWITCH.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

from plugsense.arm import ARM, ELBOW, LIFT, PAN, Arm
from plugsense.attempt_log import FOLLOWER_PORT, LEADER_PORT, Mic
from plugsense.cameras import Camera
from plugsense.gate import pull_out, save_push, teach, teleop_until_enter

HELD_OUT = {4, 8, 15, 17, 23, 26}
ROOT = Path(__file__).resolve().parent.parent / "data"


def sample_offsets(n: int, lateral: float, rng) -> list[tuple[np.ndarray, float]]:
    """Random poses around hover: sideways (pan), toward/away (elbow, lift) and height along the insertion line."""
    out = []
    for _ in range(n):
        off = np.zeros(len(ARM))
        off[PAN] = rng.uniform(-lateral, lateral)
        off[ELBOW] = rng.uniform(-lateral, lateral)
        off[LIFT] = rng.uniform(-lateral / 3, lateral / 3)
        out.append((np.round(off), float(rng.uniform(-0.5, 0.4))))  # -0.5 = higher than hover, 0.4 = lower
    return out


def push_offsets(n: int, rng) -> list[np.ndarray]:
    choices = [(j, s * k) for j in (PAN, ELBOW) for k in (2, 4, 6) for s in (1, -1)]
    picks = rng.choice(len(choices), size=min(n, len(choices)), replace=False)
    out = []
    for i in picks:
        off = np.zeros(len(ARM))
        off[choices[i][0]] = choices[i][1]
        out.append(off)
    return out


def collect_spot(arm, cams, mic, spot: int, hover, seat, home, a, rng) -> dict:
    held = spot in HELD_OUT
    n = a.held_samples if held else a.samples
    folder = ROOT / "collect" / f"{time.strftime('%Y%m%d-%H%M%S')}-spot{spot:02d}"
    (folder / "img").mkdir(parents=True, exist_ok=True)
    line = seat - hover
    pull_out(arm, hover, seat, a.lead)

    rows = []
    for i, (off, t) in enumerate(sample_offsets(n, a.lateral, rng)):
        target = hover + off + t * line
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
        base_h = hover if offset is None else hover + offset
        arm.approach(base_h, base_h + line, speed=a.speed)
        r = arm.descend(hover, seat, offset=offset, lead_max=a.lead)
        outcome = r["outcome"]
        time.sleep(0.6)  # give the power bank's display time to react
        cams["front"].save(folder / f"{name}_front_after.jpg")
        save_push(folder / "pushes" / name, r, {"spot": spot, "offset": None if offset is None else offset.tolist(),
                                              "hover": hover.tolist(), "seat": seat.tolist(), "lead": a.lead}, mic)
        pull_out(arm, hover, seat, a.lead)
        time.sleep(a.dark_wait if outcome == "seated" else 0.2)
        return outcome

    pushes = {}
    if not held:
        for k, off in enumerate(push_offsets(a.pushes, rng)):
            j = "pan" if off[PAN] else "elbow"
            name = f"{k:02d}-{j}{int(off[PAN] or off[ELBOW]):+d}"
            pushes[name] = push(name, off)
            print(f"  push {name:<12} {pushes[name]}")
    check = push("check")
    print(f"  check push from the taught pose: {check}" + ("" if check == "seated" else "   <-- power bank may have moved; redo this spot"))

    arm.goto(home, speed=a.speed)
    time.sleep(0.3)
    cams["front"].save(folder / "scene_front.jpg")
    meta = {"spot": spot, "held_out": held, "motors": ARM, "hover": hover.tolist(), "seat": seat.tolist(),
            "home": home.tolist(), "samples": len(rows), "pushes": pushes, "check": check, "grasp": a.grasp}
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mic-device", default=None)
    ap.add_argument("--front", type=int, default=0, help="front camera index")
    ap.add_argument("--wrist", type=int, default=1, help="wrist camera index")
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--held-samples", type=int, default=10)
    ap.add_argument("--pushes", type=int, default=6)
    ap.add_argument("--lateral", type=float, default=24.0, help="largest sideways offset, in encoder counts (about 8 mm)")
    ap.add_argument("--speed", type=float, default=400.0)
    ap.add_argument("--push", type=float, default=12.0, help="push strength in degrees, as in plugsense.gate")
    ap.add_argument("--near", type=float, default=45.0)
    ap.add_argument("--dark-wait", type=float, default=11.0, help="seconds to wait after a seated push so the display goes dark")
    ap.add_argument("--grasp", type=int, default=1, help="raise by one every time you re-grip the plug")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--rock", type=float, default=40.0, help="wrist rocking size for pull-outs, in encoder counts")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    cams = {"front": Camera(a.front, name="front"), "wrist": Camera(a.wrist, mjpg=True, name="wrist")}
    robot = SO101Follower(SO101FollowerConfig(port=FOLLOWER_PORT, id="follower", max_relative_target=10.0))
    leader = SO101Leader(SO101LeaderConfig(port=LEADER_PORT, id="leader"))
    robot.connect()
    leader.connect()
    arm = Arm.from_robot(robot)
    arm.rock = a.rock
    mic = Mic(int(a.mic_device) if a.mic_device and a.mic_device.isdigit() else a.mic_device) if a.mic_device else None
    done = []
    try:
        home_file = ROOT / "home.json"
        if home_file.exists() and input("Reuse the saved HOME pose? [Y/n] ").strip().lower() != "n":
            home = np.array(json.loads(home_file.read_text())["home"], float)
        else:
            teleop_until_enter(robot, leader, "Teach HOME: arm raised, clear of the power bank in the front camera. Press Enter.")
            home = arm.pos()
            ROOT.mkdir(parents=True, exist_ok=True)
            home_file.write_text(json.dumps({"motors": ARM, "home": home.tolist()}))
        while True:
            raw = input(f"\ndone: {done}\nPlate on which spot? (number, g = I re-gripped the plug, q = quit) ").strip().lower()
            if raw == "q":
                break
            if raw == "g":
                a.grasp += 1
                print(f"grasp number is now {a.grasp}")
                continue
            if not raw.isdigit() or not 1 <= int(raw) <= 30:
                continue
            spot = int(raw)
            print("Hold the leader arm (the follower will move to match it).")
            hover, seat, a.lead = teach(robot, leader, arm, a.near, a.push)
            if hover is None:
                continue
            input("Let go of the leader arm. Press Enter and the follower works alone.")
            meta = collect_spot(arm, cams, mic, spot, hover, seat, home, a, rng)
            done.append(spot)
            print(f"spot {spot} saved ({'held-out' if meta['held_out'] else 'training'}). Arm is at home. Move the plate.")
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        try:
            input("\nHOLD THE ARM with your hand, then press Enter. The motors will let go.")
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
