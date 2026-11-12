"""Add up retry sessions into one results table.

  python -m plugsense.report                       # every session under data/retry
  python -m plugsense.report 20261111-222019 ...   # only these sessions
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent / "data" / "retry"


def tally(trials: list[dict]) -> dict:
    """Counts and times over a list of trials as saved by plugsense.retry."""
    n = len(trials)
    done = [t for t in trials if t["lit_on"]]
    first = [t for t in trials if t["lit_on"] == 1]
    return {"trials": n, "first_try": len(first), "any_try": len(done),
            "seconds_first": round(float(np.mean([t["seconds"] for t in first])), 1) if first else None,
            "seconds_any": round(float(np.mean([t["seconds"] for t in done])), 1) if done else None,
            "pushes": sum(len(t["pushes"]) for t in trials)}


def main():
    names = sys.argv[1:] or sorted(p.name for p in ROOT.iterdir() if (p / "trials.json").exists())
    everything = []
    for name in names:
        d = json.loads((ROOT / name / "trials.json").read_text())
        r = tally(d["trials"])
        everything += d["trials"]
        print(f"{name}  base {d.get('base', '0'):<8} first try {r['first_try']}/{r['trials']}  within {d['tries']} tries {r['any_try']}/{r['trials']}  {d.get('note', '')}")
    r = tally(everything)
    print(f"\nall: first try {r['first_try']}/{r['trials']}, with retries {r['any_try']}/{r['trials']}, "
          f"mean time first-try {r['seconds_first']} s, mean time to lit {r['seconds_any']} s, {r['pushes']} pushes")


if __name__ == "__main__":
    main()
