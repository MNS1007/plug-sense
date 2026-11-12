"""ACT with a check on the result and a taught fallback.

  python -m plugsense.harness --policy outputs/act_stiff/pretrained_model --dataset plug_act_stiff --trials 10

Every trial:
  1. ACT attempts the plug-in (lerobot-rollout, from the average start pose of its demonstrations).
  2. The front camera reads the power bank display: lit means the plug latched.
  3. If it stayed dark, the arm backs out and repeats the plug-in you taught once
     (the last demonstration given to plugsense.retry), then checks the display again.
The arm pulls the plug out after every trial and returns to the start pose.

Prints two numbers at the end: ACT alone, and ACT with the check and fallback.
Results are saved under data/harness/<time>/trials.json.  KEEP A HAND NEAR THE POWER SWITCH.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import subprocess
import time
from pathlib import Path

import numpy as np

from plugsense import demo, display
from plugsense.retry import LAST_TAUGHT, parse_nudges, run_trial

logging.getLogger("lerobot.robots.utils").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parent.parent / "data"
LEROBOT_DATA = Path.home() / ".cache/huggingface/lerobot/local"
CAMERAS = "{front: {type: opencv, index_or_path: 0, width: 1280, height: 720, fps: 30}, wrist: {type: opencv, index_or_path: 1, width: 1280, height: 720, fps: 60}}"


def demo_start_pose(dataset: str) -> dict[str, float]:
    """Average first pose of the demonstrations ACT was trained on, as the robot's own action keys (degrees)."""
    import pandas as pd

    root = LEROBOT_DATA / dataset
    names = json.loads((root / "meta" / "info.json").read_text())["features"]["observation.state"]["names"]
    data = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(root / "data" / "*" / "*.parquet")))])
    first = np.stack(data[data.frame_index == 0]["observation.state"].to_numpy())
    return {name: float(v) for name, v in zip(names, np.median(first, axis=0))}


def summary(trials: list[dict]) -> str:
    n = len(trials)
    if not n:
        return "no trials"
    act = sum(t["act_lit"] for t in trials)
    both = sum(t["act_lit"] or t["fallback_lit"] for t in trials)
    return f"ACT alone:                    {act} of {n}\nACT + check + taught fallback: {both} of {n}"


def main():
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    from plugsense.arm import Arm
    from plugsense.attempt_log import FOLLOWER_PORT
    from plugsense.cameras import Camera

    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="folder of the trained ACT model (pretrained_model)")
    ap.add_argument("--dataset", required=True, help="the dataset ACT was trained on; its start poses are used")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--act-seconds", type=float, default=25.0, help="time ACT gets per attempt")
    ap.add_argument("--act-steps", type=int, default=100, help="moves ACT makes between looks at the cameras")
    ap.add_argument("--fallback-tries", type=int, default=2)
    ap.add_argument("--no-fallback", action="store_true", help="ACT alone: only check the display and pull out")
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    raw = json.loads(LAST_TAUGHT.read_text())
    unpack = lambda rows: [(t, np.array(c, float), np.array(p, float)) for t, c, p in rows]
    skill = demo.build_skill(unpack(raw["down"]), unpack(raw["up"]))
    if isinstance(skill, str) or raw.get("grip") is None:
        raise SystemExit("No usable taught plug-in. Run plugsense.retry once (without --reuse) first.")
    skill.grip = raw["grip"]
    home = np.array(json.loads(demo.HOME_FILE.read_text())["home"], float)
    start = demo_start_pose(a.dataset)
    session = ROOT / "harness" / time.strftime("%Y%m%d-%H%M%S")
    session.mkdir(parents=True, exist_ok=True)
    rollout = ["lerobot-rollout", "--strategy.type=base", f"--policy.path={a.policy}", "--policy.device=mps", f"--policy.n_action_steps={a.act_steps}",
               "--robot.type=so101_follower", f"--robot.port={FOLLOWER_PORT}", "--robot.id=follower", f"--robot.cameras={CAMERAS}",
               "--robot.disable_torque_on_disconnect=false", "--return_to_initial_position=false", "--play_sounds=false",
               "--task=Insert the USB-C plug into the port", f"--duration={a.act_seconds}"]

    def connect():
        robot = SO101Follower(SO101FollowerConfig(port=FOLLOWER_PORT, id="follower", max_relative_target=10.0, disable_torque_on_disconnect=False))
        robot.connect()
        arm = Arm.from_robot(robot)
        arm.sync_goal()
        return robot, arm

    def to_start(robot, seconds=2.5):
        """Smoothly to the pose ACT's demonstrations started from (all six joints, in degrees)."""
        now = robot.get_observation()
        for k in range(1, int(seconds * 30) + 1):
            f = k / (seconds * 30)
            robot.send_action({key: now[key] + f * (start[key] - now[key]) for key in start})
            time.sleep(1 / 30)
        time.sleep(0.3)

    trials = []
    robot, arm = connect()
    try:
        input("Plug out, power bank on its marks. The arm will move to ACT's start pose. Press Enter.")
        arm.goto(home, speed=200.0)
        to_start(robot)
        for i in range(a.trials):
            robot.disconnect()  # the motors keep holding; ACT takes over the arm and the cameras
            t0 = time.monotonic()
            log = open(session / f"t{i:02d}-act.log", "w")
            code = subprocess.run(rollout, stdout=log, stderr=subprocess.STDOUT).returncode
            act_seconds = time.monotonic() - t0
            log.close()
            if code:
                print(f"ACT did not run (exit code {code}). Last lines of {log.name}:")
                print("".join(open(log.name).readlines()[-6:]))
                robot, arm = connect()
                break
            cam = Camera(0, 1280, 720, name="front")
            read_share = lambda: display.share(cam.fresh(timeout=2.0))
            t_look = time.monotonic()
            share = read_share()
            while share < display.LIT_SHARE and time.monotonic() - t_look < 3.0:
                time.sleep(0.2)
                share = read_share()
            act_lit = bool(share >= display.LIT_SHARE)
            import cv2
            cv2.imwrite(str(session / f"t{i:02d}-after-act.jpg"), cam.fresh(timeout=2.0))
            print(f"trial {i + 1}/{a.trials}: ACT {'LIT ' if act_lit else 'dark'} (bright share {share:.3f}, {act_seconds:.0f} s" + (f", rollout exit code {code}" if code else "") + ")")

            def wait_dark(limit=15.0) -> bool:
                t = time.monotonic()
                while read_share() >= display.LIT_SHARE / 2:
                    if time.monotonic() - t > limit:
                        return False
                    time.sleep(0.5)
                return True

            def pull(offset, was_lit):
                arm.replay(skill.pull, offset)
                if was_lit and not wait_dark():
                    arm.relax()
                    input("  The display is still lit, so the plug seems to be in. Pull it out by hand, then press Enter. ")
                    arm.sync_goal()
                    wait_dark()

            robot, arm = connect()
            at_port = float(np.abs(arm.pos() - skill.seat)[:4].max()) < 2 * skill.near
            if act_lit or at_port:
                pull(None, act_lit)  # back out along your taught pull-out
            t = {"act_lit": act_lit, "act_seconds": round(act_seconds, 1), "fallback_lit": False, "fallback": None}
            if not act_lit and not a.no_fallback:
                fb = run_trial(arm, skill, home, read_share, pull, parse_nudges("0,0"), a.fallback_tries)
                t.update(fallback_lit=bool(fb["lit_on"]), fallback=fb)
                print(f"           fallback {'LIT on try ' + str(fb['lit_on']) if fb['lit_on'] else 'dark'} ({fb['seconds']:.0f} s)")
            trials.append(t)
            (session / "trials.json").write_text(json.dumps({"policy": a.policy, "act_steps": a.act_steps, "act_seconds": a.act_seconds,
                                                             "note": a.note, "trials": trials}, indent=2))
            cam.close()
            arm.goto(home, speed=200.0)
            to_start(robot)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        print("\n" + summary(trials) + f"\nsaved under {session}")
        try:
            if not robot.is_connected:
                robot, arm = connect()
            input("HOLD THE FOLLOWER ARM with your hand, then press Enter. The motors will let go.")
            robot.bus.disable_torque()
            robot.disconnect()
        except (EOFError, KeyboardInterrupt, Exception) as e:  # noqa: BLE001
            print(f"could not release the motors cleanly ({e}); switch the arm off by hand")


if __name__ == "__main__":
    main()
