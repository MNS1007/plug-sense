"""Teach by doing: you plug in and pull out once, the arm repeats your commands.

The push and the pull-out are not designed motions. They are your own teleop commands,
recorded at 30 Hz and sent again with the same timing, so the motors push, twist, shake
and pull with the force you used. Everything is in raw encoder counts.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

from plugsense.arm import ARM, Arm

ROOT = Path(__file__).resolve().parent.parent / "data"
HOME_FILE = ROOT / "home.json"
TELEOP_HZ = 30.0


def enter_pressed() -> threading.Event:
    done = threading.Event()
    threading.Thread(target=lambda: (sys.stdin.readline(), done.set()), daemon=True).start()
    return done


def teleop(robot, leader, arm: Arm, prompt: str) -> list:
    """Follow the leader until Enter. Returns (seconds, commanded goal, measured pose) per step."""
    print(prompt)
    done, steps, t_start = enter_pressed(), [], time.monotonic()
    while not done.is_set():
        t0 = time.monotonic()
        robot.send_action(leader.get_action())
        steps.append((t0 - t_start, arm.commanded(), arm.pos()))
        time.sleep(max(0.0, 1 / TELEOP_HZ - (time.monotonic() - t0)))
    return steps


def wait_for_match(robot, leader, tolerance: float = 12.0, timeout: float = 120.0) -> bool:
    """Wait until the leader is held close to the follower's pose, so the follower does not jump when it starts following."""
    print("Bring the leader arm to the same pose as the follower...", end="", flush=True)
    t_start, shown = time.monotonic(), 0.0
    while time.monotonic() - t_start < timeout:
        want = leader.get_action()
        have = robot.bus.sync_read("Present_Position")
        gaps = {m: abs(want[f"{m}.pos"] - have[m]) for m in ARM if f"{m}.pos" in want}
        worst = max(gaps, key=gaps.get)
        if gaps[worst] <= tolerance:
            print(" matched.")
            return True
        if time.monotonic() - shown > 1.0:
            print(f"\n  {worst} is {gaps[worst]:.0f} degrees off", end="", flush=True)
            shown = time.monotonic()
        time.sleep(0.05)
    print(" timed out.")
    return False


def get_home(robot, leader, arm: Arm, ask: bool = True) -> np.ndarray:
    if HOME_FILE.exists() and (not ask or input("Reuse the saved start pose? [Y/n] ").strip().lower() != "n"):
        return np.array(json.loads(HOME_FILE.read_text())["home"], float)
    teleop(robot, leader, arm, "Teach the START pose: arm raised, plug well above and clear of the power bank. Press Enter.")
    home = arm.pos()
    ROOT.mkdir(parents=True, exist_ok=True)
    HOME_FILE.write_text(json.dumps({"motors": ARM, "home": home.tolist()}))
    return home


class Skill:
    """Your seat pose, the last stretch of your push, and your pull-out."""

    def __init__(self, seat, hover, push, pull, near: float = 45.0):
        self.near = near
        self.seat, self.hover = np.asarray(seat, float), np.asarray(hover, float)
        self.push, self.pull = push, pull            # lists of (seconds, commanded goal)
        self.line = (self.seat - self.hover)[:4]     # wrist twist left out: it is not travel

    def depth(self, pose) -> float:
        """0 at the start of the push, 1 at the pose where you had the plug seated."""
        return float((np.asarray(pose, float) - self.hover)[:4] @ self.line / (self.line @ self.line))

    def away(self, pose) -> bool:
        """True when the arm is clearly away from the seated pose, in any direction (wrist twist left out)."""
        return bool(np.abs(np.asarray(pose, float) - self.seat)[:4].max() >= self.near)

    def save(self, path: Path) -> None:
        pack = lambda seq: [[t, q.tolist()] for t, q in seq]
        path.write_text(json.dumps({"motors": ARM, "seat": self.seat.tolist(), "hover": self.hover.tolist(),
                                    "push": pack(self.push), "pull": pack(self.pull), "near": self.near}))

    @classmethod
    def load(cls, path: Path) -> "Skill":
        d = json.loads(path.read_text())
        unpack = lambda seq: [(t, np.array(q, float)) for t, q in seq]
        return cls(d["seat"], d["hover"], unpack(d["push"]), unpack(d["pull"]), d.get("near", 45.0))


def build_skill(down: list, up: list, near: float = 45.0, lead_in: float = 60.0) -> Skill | str:
    """Cut the two recordings into a push and a pull-out. Returns a message if they cannot be used.

    down: teleop steps ending with the plug seated.  up: teleop steps of the pull-out.
    The push starts where the arm was last `near` counts (busiest joint, twist excluded) from the
    seated pose (that pose is the reference for depth). The replayed motion begins lead_in seconds
    before that; by default that is your whole approach from the start pose, so the arm arrives
    with the same history of moves, which matters on servos with play.
    """
    seat = down[-1][2]
    far = [i for i, s in enumerate(down) if np.abs(s[2] - seat)[:4].max() >= near]
    if not far:
        return "The arm was never far enough from the seated pose. Start from higher up."
    start = far[-1]
    hover = down[start][2]
    first = max(0, start - int(lead_in * TELEOP_HZ))
    moving = next((i for i, s in enumerate(down) if np.abs(s[1] - down[0][1]).max() > 3), 0)
    first = max(first, moving - 3, 0)  # skip the time you spent holding still before starting
    push = [(t - down[first][0], q) for t, q, _ in down[first:]]
    skill = Skill(seat, hover, push, [], near)
    away = [skill.away(s[2]) for s in up]
    if not away[-1]:
        return "At the end of the pull-out the arm was still at the seated pose. Pull the plug out and move it away before pressing Enter."
    last_close = max((i for i, a in enumerate(away) if not a), default=0)  # after this step the arm stayed away for good
    moving = next((i for i, s in enumerate(up) if np.abs(s[1] - up[0][1]).max() > 3), 0)
    cut = up[max(0, moving - 3):last_close + 15]
    skill.pull = [(t - cut[0][0], q) for t, q, _ in cut]
    return skill


def teach(robot, leader, arm: Arm, home: np.ndarray, near: float = 45.0, lead_in: float = 60.0) -> Skill | None:
    arm.sync_goal()
    arm.goto(home, speed=300.0)
    if not wait_for_match(robot, leader):
        return None
    down = teleop(robot, leader, arm, "\nYou have control. Come down and SEAT the plug. Press Enter while it is seated.")
    up = teleop(robot, leader, arm, "Now PULL THE PLUG OUT the way that works. Press Enter once it is out and above the port.")
    skill = build_skill(down, up, near, lead_in)
    if isinstance(skill, str):
        print(skill)
        return None
    print(f"plug-in: {skill.push[-1][0]:.1f} s of your motion will be repeated.  pull-out: {skill.pull[-1][0]:.1f} s.")
    print("push travel, counts per joint:", {m: int(v) for m, v in zip(ARM, skill.seat - skill.hover)})
    return skill


def run_push(arm: Arm, skill: Skill, shift=None, hold: float = 0.4, seated_at: float = 0.85) -> dict:
    """Repeat your push (optionally shifted by a known offset) and report where the arm ended up."""
    t_start = time.monotonic()
    rows = arm.replay(skill.push, shift, log=True)
    time.sleep(hold)
    off = np.zeros(len(ARM)) if shift is None else np.asarray(shift, float)
    depth = skill.depth(arm.pos() - off)
    outcome = "jammed" if depth < seated_at else ("seated" if depth <= 1.3 else "overrun")
    return {"outcome": outcome, "frac": round(depth, 3), "seconds": round(time.monotonic() - t_start, 2),
            "t_start": t_start, "rows": rows}


def run_pull(arm: Arm, skill: Skill, shift=None, tries: int = 2) -> bool:
    """Repeat your pull-out until the plug is clear. Does nothing if it is not in."""
    off = np.zeros(len(ARM)) if shift is None else np.asarray(shift, float)
    for _ in range(tries + 1):
        if skill.away(arm.pos() - off):
            return True
        arm.replay(skill.pull, shift)
        time.sleep(0.2)
    return skill.away(arm.pos() - off)
