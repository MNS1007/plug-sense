"""Trains the fine network on synthetic images where a bright square's position encodes the label."""
import csv
import json

import cv2
import numpy as np

from plugsense import fine
from plugsense.arm import ARM


def make_spot(folder, spot, held, n, rng):
    d = folder / f"t-spot{spot:02d}"
    (d / "img").mkdir(parents=True)
    hover = np.array([2000, 2100, 1900, 2050, 2048], float) + rng.integers(-40, 40, 5)
    rows = []
    for i in range(n):
        off = np.zeros(5)
        off[0], off[2] = rng.integers(-24, 25), rng.integers(-24, 25)
        img = np.full((480, 640, 3), 60, np.uint8)
        cx, cy = int(320 + off[0] * 6), int(240 + off[2] * 6)   # the "port" moves in the image as the arm moves
        cv2.rectangle(img, (cx - 20, cy - 20), (cx + 20, cy + 20), (230, 230, 230), -1)
        cv2.imwrite(str(d / f"img/{i:03d}_wrist.jpg"), img)
        cmd = hover + off
        rows.append({"i": i, **{f"cmd_{m}": v for m, v in zip(ARM, cmd)}, **{f"meas_{m}": v for m, v in zip(ARM, cmd)},
                     "wrist": f"img/{i:03d}_wrist.jpg", "front": ""})
    with open(d / "samples.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (d / "meta.json").write_text(json.dumps({"spot": spot, "held_out": held, "hover": hover.tolist(), "check": "seated"}))


def test_learns_offset_on_unseen_spots(tmp_path):
    rng = np.random.default_rng(0)
    for s in range(1, 7):
        make_spot(tmp_path, s, held=False, n=30, rng=rng)
    make_spot(tmp_path, 8, held=True, n=20, rng=rng)
    report = fine.train(tmp_path, tmp_path / "out", epochs=25, batch=32, lr=1e-3, pretrained=False, heads=3)
    held = report["held_out"]
    assert held["n"] == 20
    # offsets span +-24 counts; guessing zero gives a mean error near 12 on the two moving joints
    assert held["mean_abs_counts"]["shoulder_pan"] < 6 and held["mean_abs_counts"]["elbow_flex"] < 6
