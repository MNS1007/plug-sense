"""Motion primitives for the SO-101 follower, in raw encoder counts.

Everything here works in the servos' own units (4096 counts per turn, 0.088 deg per
count) so there is no calibration or kinematics between a label and a command.
The gripper is never commanded: it keeps holding with the goal teleop left it.
"""
from __future__ import annotations

import math
import time

import numpy as np

ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
PAN, LIFT, ELBOW, WRIST = 0, 1, 2, 3
HZ = 50.0


def min_jerk(n: int) -> np.ndarray:
    """Fractions 0..1 for a smooth move in n steps (zero speed at both ends)."""
    t = np.linspace(0.0, 1.0, max(n, 2))[1:]
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def decode_load(v: float) -> float:
    """Present_Load is sign-magnitude with the sign in bit 10; some readers decode it already."""
    return float(v) if v < 1024 else -float(v - 1024)


class Arm:
    rock = 40.0  # wrist rocking size for pull-outs, in encoder counts (about 3.5 degrees)

    def __init__(self, bus, limits: dict[str, tuple[int, int]] | None = None):
        self.bus = bus
        lim = limits or {}
        self.lo = np.array([lim.get(m, (0, 4095))[0] for m in ARM], float)
        self.hi = np.array([lim.get(m, (0, 4095))[1] for m in ARM], float)
        self.goal = self.pos()

    @classmethod
    def from_robot(cls, robot):
        cal = robot.calibration or {}
        return cls(robot.bus, {m: (c.range_min, c.range_max) for m, c in cal.items() if m in ARM})

    def pos(self) -> np.ndarray:
        d = self.bus.sync_read("Present_Position", ARM, normalize=False)
        return np.array([d[m] for m in ARM], float)

    def load(self) -> np.ndarray:
        d = self.bus.sync_read("Present_Load", ARM, normalize=False)
        return np.array([decode_load(d[m]) for m in ARM], float)

    def write(self, q: np.ndarray) -> None:
        q = np.clip(np.round(q), self.lo, self.hi)
        self.bus.sync_write("Goal_Position", {m: int(v) for m, v in zip(ARM, q)}, normalize=False)
        self.goal = q

    def sync_goal(self) -> None:
        """Call after teleop: start the next move from where the arm really is."""
        self.goal = self.pos()

    def goto(self, q: np.ndarray, speed: float = 500.0, settle: float = 0.15) -> None:
        """Smooth move from the last commanded pose. speed = peak counts per second on the busiest joint."""
        q = np.asarray(q, float)
        start = self.goal.copy()
        dist = float(np.abs(q - start).max())
        if dist >= 1:
            seconds = max(0.2, 1.875 * dist / speed)  # 1.875 = peak/mean speed of a min-jerk move
            for f in min_jerk(int(seconds * HZ)):
                t0 = time.monotonic()
                self.write(start + f * (q - start))
                time.sleep(max(0.0, 1.0 / HZ - (time.monotonic() - t0)))
        time.sleep(settle)

    def approach(self, hover: np.ndarray, seat: np.ndarray, speed: float = 500.0, above: float = 0.6) -> None:
        """Reach hover from further up the insertion line, so servo play always sits on the same side."""
        hover, seat = np.asarray(hover, float), np.asarray(seat, float)
        self.goto(hover - above * (seat - hover), speed, settle=0.05)
        self.goto(hover, speed=min(speed, 200.0), settle=0.25)

    def descend(
        self,
        hover: np.ndarray,
        seat: np.ndarray,
        offset: np.ndarray | None = None,
        seconds: float = 1.5,
        lead_max: float = 0.30,
        overshoot: float | None = None,
        seated_at: float = 0.85,
        wiggle: float = 0.0,
        wiggle_hz: float = 3.0,
        timeout: float = 5.0,
        on_tick=None,
    ) -> dict:
        """Push from hover toward seat along a straight line in joint space.

        The commanded point never runs more than lead_max (a fraction of the line) ahead of
        the measured one, and never more than overshoot past the seat, which caps the push
        force of a position servo. The push ends when the arm stops making progress while
        held at that cap.
          seated  stopped at or beyond seated_at of the way
          jammed  stopped earlier
          missed  went past the seat without ever being stopped
        offset shifts the whole line (counts per joint), for probing from a known error.
        wiggle (counts) adds a small circle on pan and elbow once the arm meets resistance.
        """
        overshoot = lead_max if overshoot is None else overshoot  # same push at the seat as anywhere else
        hover, seat = np.asarray(hover, float), np.asarray(seat, float)
        off = np.zeros(len(ARM)) if offset is None else np.asarray(offset, float)
        line = seat - hover
        length2 = float(line @ line)
        if length2 < 1:
            raise ValueError("hover and seat are the same pose")

        def frac(q):
            return float((q - hover - off) @ line / length2)

        rows, history = [], []
        g, outcome, t_start = 0.0, "timeout", time.monotonic()
        while (now := time.monotonic()) - t_start < timeout:
            p = self.pos()
            f = frac(p)
            g = min(g + (1.0 + overshoot) / (seconds * HZ), 1.0 + overshoot, f + lead_max)
            pushing = g - f >= 0.9 * lead_max or g >= 1.0 + overshoot - 1e-9
            cmd = hover + off + g * line
            if wiggle and pushing:
                ph = 2 * math.pi * wiggle_hz * (now - t_start)
                cmd[PAN] += wiggle * math.sin(ph)
                cmd[ELBOW] += wiggle * math.cos(ph)
            self.write(cmd)
            load = self.load()
            rows.append({"t": now, "goal_frac": g, "frac": f, **{f"pos_{m}": v for m, v in zip(ARM, p)},
                         **{f"load_{m}": v for m, v in zip(ARM, load)}})
            if on_tick:
                on_tick(now)
            history.append((now, f, pushing))
            history = [h for h in history if now - h[0] <= 0.4]
            stalled = (now - history[0][0] >= 0.35 and all(h[2] for h in history)
                       and max(h[1] for h in history) - min(h[1] for h in history) < 0.04)
            if f >= 1.0 + overshoot - 0.03:
                outcome = "missed"
                break
            if stalled:
                outcome = "seated" if f >= seated_at else "jammed"
                break
            time.sleep(max(0.0, 1.0 / HZ - (time.monotonic() - now)))
        final = frac(self.pos())
        return {"outcome": outcome, "frac": round(final, 3), "seconds": round(time.monotonic() - t_start, 2),
                "t_start": t_start, "rows": rows}

    def retract(self, hover: np.ndarray, seat: np.ndarray, seconds: float = 1.5, lead_max: float = 0.40,
                rock: float | None = None, rock_hz: float = 3.0, timeout: float = 10.0, to: float = -0.6) -> bool:
        """Pull out along the insertion line to a point above hover. Returns False if the plug stayed stuck.

        The commanded point never runs more than lead_max ahead of the measured one. Whenever the
        arm is held back at that cap (the latch is holding), it rocks the plug front to back with
        the wrist (rock = size in encoder counts), plus a little elbow and pan, which releases a
        USB-C latch with far less force than a straight pull.
        """
        rock = self.rock if rock is None else rock
        hover, seat = np.asarray(hover, float), np.asarray(seat, float)
        line = seat - hover
        length2 = float(line @ line)
        p0 = self.pos()
        f0 = float((p0 - hover) @ line / length2)
        side = p0 - hover - f0 * line  # keep whatever sideways offset the arm has now
        g, t_start = f0, time.monotonic()
        while (now := time.monotonic()) - t_start < timeout:
            f = float((self.pos() - hover - side) @ line / length2)
            if f <= to + 0.05:
                self.goto(hover + side + to * line, speed=200.0, settle=0.2)
                return True
            g = max(g - (f0 - to) / (seconds * HZ), to, f - lead_max)
            cmd = hover + side + g * line
            if rock and f - g >= 0.9 * lead_max:
                ph = 2 * math.pi * rock_hz * (now - t_start)
                cmd[WRIST] += rock * math.sin(ph)
                cmd[ELBOW] += 0.3 * rock * math.sin(ph)
                cmd[PAN] += 0.2 * rock * math.cos(ph)
            self.write(cmd)
            time.sleep(max(0.0, 1.0 / HZ - (time.monotonic() - now)))
        return False
