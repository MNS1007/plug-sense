"""Checks the push-look-retry loop on a simulated arm. No hardware."""
import numpy as np

from plugsense import demo, retry
from plugsense.arm import Arm
from test_arm import FakeBus, SEAT, fast
from test_demo import TOP, UP, approach, steps


def setup(monkeypatch):
    fast(monkeypatch)
    monkeypatch.setattr(demo.time, "sleep", lambda s: None)
    down = steps(approach() + [SEAT] * 10)
    up = steps([SEAT] * 5 + [SEAT + (UP - SEAT) * k / 9 for k in range(10)])
    return demo.build_skill(down, up, near=45), Arm(FakeBus(TOP))


def test_retries_until_the_display_lights(monkeypatch):
    skill, arm = setup(monkeypatch)
    pan = lambda: arm.commanded()[0] - SEAT[0]
    # the display lights only when the push is 3 counts to the side, and only while the arm is down
    read = lambda: 0.3 if abs(pan() - 3) < 0.5 and not skill.away(arm.pos()) else 0.0
    pulls = []
    t = retry.run_trial(arm, skill, TOP, read, lambda off, lit: (pulls.append((off[0], lit)), arm.goto(TOP)), retry.parse_nudges("0,0,pan+3,pan-3"), 4)
    assert t["lit_on"] == 3 and [p["lit"] for p in t["pushes"]] == [False, False, True] and pulls == [(0, False), (0, False), (3, True)]


def test_gives_up_after_the_allowed_tries(monkeypatch):
    skill, arm = setup(monkeypatch)
    t = retry.run_trial(arm, skill, TOP, lambda: 0.0, lambda off, lit: arm.goto(TOP), retry.parse_nudges("0"), 2)
    assert t["lit_on"] == 0 and len(t["pushes"]) == 2
    assert "within 2 tries: 0 of 1" in retry.summary([t], 2)


def test_report_tally():
    from plugsense.report import tally
    trials = [{"lit_on": 1, "seconds": 12.0, "pushes": [1]}, {"lit_on": 2, "seconds": 30.0, "pushes": [1, 2]}, {"lit_on": 0, "seconds": 50.0, "pushes": [1, 2, 3]}]
    r = tally(trials)
    assert (r["first_try"], r["any_try"], r["trials"], r["seconds_first"], r["seconds_any"], r["pushes"]) == (1, 2, 3, 12.0, 21.0, 6)


def test_harness_summary_counts_fallback_rescues():
    from plugsense.harness import summary
    trials = [{"act_lit": True, "fallback_lit": False}, {"act_lit": False, "fallback_lit": True}, {"act_lit": False, "fallback_lit": False}]
    assert "ACT alone:                    1 of 3" in summary(trials) and "fallback: 2 of 3" in summary(trials)
