"""Checks cutting a demonstration into a push and a pull-out, and replaying them. No hardware."""
import numpy as np

import plugsense.arm as armmod
from plugsense import demo
from plugsense.arm import Arm
from test_arm import HOVER, SEAT, FakeBus, fast

TOP = HOVER - np.array([0, 200, 0, 0, 0.0])
UP = HOVER - np.array([0, 40, 0, 0, 0.0])


def steps(poses, lead=None, t0=0.0):
    """Teleop steps where the command runs `lead` ahead of the measured pose (as when pushing)."""
    lead = np.zeros(5) if lead is None else lead
    return [(t0 + i / 30, p + lead, p) for i, p in enumerate(poses)]


def approach(n=60):
    return [TOP + (SEAT - TOP) * k / (n - 1) for k in range(n)]


def test_build_skill_cuts_push_and_pull():
    down = steps(approach() + [SEAT] * 10, lead=np.array([0, 30, 0, 0, 0.0]))
    up = steps([SEAT] * 15 + [SEAT + (UP - SEAT) * k / 9 for k in range(10)] + [TOP] * 30)
    skill = demo.build_skill(down, up, near=45)
    assert not isinstance(skill, str)
    assert np.abs(skill.hover - SEAT)[:4].max() >= 45 and abs(skill.depth(SEAT) - 1) < 1e-6
    assert 0.9 < skill.push[-1][0] < 2.5 and np.allclose(skill.push[-1][1], SEAT + [0, 30, 0, 0, 0])
    assert len(skill.pull) < 30 and skill.away(skill.pull[-1][1])   # the idle start and most of the drive to TOP are dropped


def test_build_skill_rejects_bad_demos():
    down = steps(approach())
    assert isinstance(demo.build_skill(steps([SEAT] * 20), steps([UP]), 45), str)        # never far from the seat
    assert isinstance(demo.build_skill(down, steps([SEAT] * 20), 45), str)               # plug never pulled clear


def test_push_then_pull_on_a_simulated_arm(monkeypatch, tmp_path):
    fast(monkeypatch)
    monkeypatch.setattr(demo.time, "sleep", lambda s: None)
    down = steps(approach() + [SEAT] * 10)
    up = steps([SEAT] * 5 + [SEAT + (UP - SEAT) * k / 9 for k in range(10)])
    skill = demo.build_skill(down, up, near=45)
    skill.save(tmp_path / "s.json")
    skill = demo.Skill.load(tmp_path / "s.json")
    bus = FakeBus(TOP)
    arm = Arm(bus)
    r = demo.run_push(arm, skill)
    assert r["outcome"] == "seated" and len(r["rows"]) == len(skill.push)
    assert demo.is_free(arm, skill) is False            # resting at the seated pose: treated as still in
    assert demo.run_pull(arm, skill) is True and skill.away(arm.pos())
    assert demo.run_pull(arm, skill) is True            # already out: nothing replayed
    shift = np.array([6, 0, 0, 0, 0.0])
    assert demo.run_push(arm, skill, shift)["outcome"] == "seated"   # depth is measured relative to the shifted push


def test_pull_out_in_another_direction_counts_as_clear():
    down = steps(approach() + [SEAT] * 10)
    wrist_up = SEAT + np.array([0, 0, 0, 120, 0.0])     # raised with the wrist, not back along the push
    shake = [SEAT + np.array([0, 0, 0, 60 * (-1) ** k, 0.0]) for k in range(12)]   # big shakes while still in
    up = steps([SEAT] * 5 + shake + [SEAT] + [SEAT + (wrist_up - SEAT) * k / 9 for k in range(10)] + [wrist_up] * 40)
    skill = demo.build_skill(down, up, near=45)
    assert not isinstance(skill, str) and skill.away(skill.pull[-1][1])
    assert len(skill.pull) > 20      # the shaking is kept, not cut at the first big swing


def test_latched_plug_is_not_called_free(monkeypatch):
    """Pulled 60 counts off the seated pose under load, but it springs back when the arm stops pulling."""
    fast(monkeypatch)
    monkeypatch.setattr(demo.time, "sleep", lambda s: None)
    down = steps(approach() + [SEAT] * 10)
    up = steps([SEAT] * 5 + [SEAT + (UP - SEAT) * k / 9 for k in range(10)])
    skill = demo.build_skill(down, up, near=45)

    class Sprung(FakeBus):
        def sync_read(self, name, motors, normalize=True):
            if name == "Present_Position":       # the latch holds the arm within 60 counts of the seat, more only under load
                pull = np.clip(self.goal - SEAT, -60, 60)
                relaxed = np.abs(self.goal - self.p).max() < 30
                self.p = SEAT + (0 if relaxed else pull)
                return dict(zip(demo.ARM, self.p))
            return super().sync_read(name, motors, normalize)

    bus = Sprung(SEAT)
    arm = Arm(bus)
    arm.write(SEAT + np.array([0, -200, 0, 0, 0.0]))     # pulling hard
    assert skill.away(arm.pos())                          # bent away under load
    assert demo.is_free(arm, skill) is False              # but not free
