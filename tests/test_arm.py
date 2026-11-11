"""Checks the motion logic against a simulated arm. No hardware."""
import numpy as np

from plugsense import arm as armmod
from plugsense.arm import ARM, Arm, decode_load, min_jerk


class FakeBus:
    """Servos that follow their goal with a lag, and stop at a wall along the insertion line."""

    def __init__(self, start, hover=None, seat=None, wall=None):
        self.p = np.array(start, float)
        self.goal = self.p.copy()
        self.hover, self.seat, self.wall = hover, seat, wall

    def sync_read(self, name, motors, normalize=True):
        if name == "Present_Position":
            self.p += 0.5 * (self.goal - self.p)
            if self.wall is not None:
                line = self.seat - self.hover
                f = (self.p - self.hover) @ line / (line @ line)
                if f > self.wall:
                    self.p -= (f - self.wall) * line
            return dict(zip(ARM, self.p))
        return dict(zip(ARM, [0.0] * len(ARM)))

    def sync_write(self, name, values, normalize=True):
        self.goal = np.array([values[m] for m in ARM], float)


HOVER = np.array([2000, 2000, 2000, 2000, 2000], float)
SEAT = HOVER + np.array([0, 30, -25, 10, 0], float)


def fast(monkeypatch):
    monkeypatch.setattr(armmod.time, "sleep", lambda s: None)
    clock = iter(np.arange(0, 1e6, 1.0 / armmod.HZ))
    monkeypatch.setattr(armmod.time, "monotonic", lambda: float(next(clock)))


def test_min_jerk_ends_at_one():
    f = min_jerk(50)
    assert abs(f[-1] - 1.0) < 1e-9 and np.all(np.diff(f) >= 0)


def test_decode_load():
    assert decode_load(100) == 100 and decode_load(1024 + 100) == -100 and decode_load(-50) == -50


def test_goto_reaches_target_and_respects_limits(monkeypatch):
    fast(monkeypatch)
    a = Arm(FakeBus(HOVER), {"shoulder_lift": (0, 2010)})
    a.goto(SEAT)
    assert a.goal[0] == SEAT[0] and a.goal[1] == 2010 and a.goal[2] == SEAT[2]


def run_descent(monkeypatch, wall):
    fast(monkeypatch)
    a = Arm(FakeBus(HOVER, HOVER, SEAT, wall))
    return a.descend(HOVER, SEAT)


def test_descent_seated(monkeypatch):
    r = run_descent(monkeypatch, wall=1.0)
    assert r["outcome"] == "seated" and 0.9 < r["frac"] <= 1.05


def test_descent_jammed(monkeypatch):
    r = run_descent(monkeypatch, wall=0.4)
    assert r["outcome"] == "jammed" and r["frac"] < 0.6


def test_descent_missed(monkeypatch):
    r = run_descent(monkeypatch, wall=None)
    assert r["outcome"] == "missed"


def test_push_is_capped(monkeypatch):
    r = run_descent(monkeypatch, wall=0.4)
    assert max(row["goal_frac"] - row["frac"] for row in r["rows"]) <= 0.30 + 1e-6


class LatchBus(FakeBus):
    """Holds the plug at the seat until it has been rocked sideways a few times."""

    def __init__(self, *a, need=4, **k):
        super().__init__(*a, **k)
        self.need, self.swings, self.last = need, 0, 0

    def sync_write(self, name, values, normalize=True):
        super().sync_write(name, values, normalize)
        side = np.sign(round(self.goal[0] - HOVER[0]))
        if side and side != self.last:
            self.swings, self.last = self.swings + 1, side

    def sync_read(self, name, motors, normalize=True):
        if name == "Present_Position" and self.swings < self.need:
            return dict(zip(ARM, SEAT))
        return super().sync_read(name, motors, normalize)


def test_retract_free(monkeypatch):
    fast(monkeypatch)
    a = Arm(FakeBus(SEAT, HOVER, SEAT))
    assert a.retract(HOVER, SEAT) is True
    assert np.allclose(a.goal, HOVER - 0.6 * (SEAT - HOVER), atol=1)


def test_retract_rocks_a_latched_plug_free(monkeypatch):
    fast(monkeypatch)
    bus = LatchBus(SEAT, HOVER, SEAT)
    a = Arm(bus)
    assert a.retract(HOVER, SEAT) is True and bus.swings >= 4


def test_retract_reports_stuck(monkeypatch):
    fast(monkeypatch)
    a = Arm(LatchBus(SEAT, HOVER, SEAT, need=10**9))
    assert a.retract(HOVER, SEAT, timeout=3.0) is False
