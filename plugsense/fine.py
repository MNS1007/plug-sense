"""Fine alignment network: close-up of the port from the front camera -> joint offset to the hover pose.

The wrist camera on this arm cannot see the plug tip or the port (the jaw is in the way),
so the image is a square cut out of the full-resolution front camera picture around the port.

  python -m plugsense.fine train                 # train on data/collect, test on the held-out spots
  python -m plugsense.fine train --epochs 30

Labels come from self-collection (plugsense.collect): every photo was taken at a known
joint offset from the hover pose that seated the plug. Offsets are in encoder counts.
The joint angles are deliberately NOT an input: at one taped location the offset could be
computed from them alone, and the network would never need to look at the port.
The network has several output heads; how much they disagree is used later to decide
between moving and looking again.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from plugsense.arm import ARM

ROOT = Path(__file__).resolve().parent.parent
SIZE = 224
OUT_SCALE = 30.0   # counts; offsets are divided by this for the loss
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_spots(folder: Path) -> list[dict]:
    """One entry per collected session, with its samples."""
    spots = []
    for d in sorted(folder.glob("*-spot*")):
        if not (d / "meta.json").exists() or not (d / "samples.csv").exists():
            continue
        meta = json.loads((d / "meta.json").read_text())
        hover = np.array(meta["hover"], np.float32)
        view, crop = meta.get("view", "wrist"), meta.get("crop")
        rows = []
        with open(d / "samples.csv") as f:
            for r in csv.DictReader(f):
                if not r.get(view):
                    continue
                cmd = np.array([float(r[f"cmd_{m}"]) for m in ARM], np.float32)
                meas = np.array([float(r[f"meas_{m}"]) for m in ARM], np.float32)
                rows.append({"image": str(d / r[view]), "crop": crop, "pos": meas, "target": hover - meas})  # measured to measured
        spots.append({"name": d.name, "spot": meta["spot"], "held_out": meta["held_out"], "hover": hover, "rows": rows})
    return spots


def cut(image_bgr: np.ndarray, crop) -> np.ndarray:
    """crop = (centre x, centre y, side) in pixels of the full picture; None keeps the whole picture."""
    if not crop:
        return image_bgr
    cx, cy, side = (int(v) for v in crop)
    h, w = image_bgr.shape[:2]
    x0, y0 = min(max(cx - side // 2, 0), w - side), min(max(cy - side // 2, 0), h - side)
    return image_bgr[y0:y0 + side, x0:x0 + side]


def prepare(image_bgr: np.ndarray, crop=None) -> np.ndarray:
    img = cv2.resize(cut(image_bgr, crop), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def augment(img: np.ndarray, rng) -> np.ndarray:
    """Lighting changes and a shift of at most 2 pixels. No flips or rotations: position is the label."""
    img = img * rng.uniform(0.75, 1.25) + rng.uniform(-0.08, 0.08)
    img = img * rng.uniform(0.9, 1.1, size=(1, 1, 3)).astype(np.float32)
    dx, dy = rng.integers(-2, 3, size=2)
    img = np.roll(img, (int(dy), int(dx)), axis=(0, 1))
    return np.clip(img, 0.0, 1.0)


class Samples(torch.utils.data.Dataset):
    def __init__(self, spots, centre, train: bool, seed: int = 0):
        self.rows = [r for s in spots for r in s["rows"]]
        self.centre, self.train = centre, train
        self.rng = np.random.default_rng(seed)
        self.cache: dict[str, np.ndarray] = {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        if r["image"] not in self.cache:
            self.cache[r["image"]] = prepare(cv2.imread(r["image"]), r.get("crop"))
        img = self.cache[r["image"]]
        if self.train:
            img = augment(img, self.rng)
        img = torch.from_numpy(((img - MEAN) / STD).transpose(2, 0, 1).astype(np.float32))
        return img, torch.from_numpy(r["target"] / OUT_SCALE)


class FineNet(nn.Module):
    """ResNet-18 features kept as a grid, reduced to image coordinates of 32 learned points."""

    def __init__(self, heads: int = 5, pretrained: bool = True):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        net = torchvision.models.resnet18(weights=weights)
        self.trunk = nn.Sequential(*list(net.children())[:-3])  # stride 16: a 14 x 14 grid at 224 px
        self.points = nn.Conv2d(256, 32, 1)
        self.heads = nn.ModuleList(
            nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, len(ARM))) for _ in range(heads))

    def forward(self, img):
        f = self.points(self.trunk(img))
        b, c, h, w = f.shape
        p = F.softmax(f.view(b, c, -1), dim=-1).view(b, c, h, w)
        ys = torch.linspace(-1, 1, h, device=f.device)
        xs = torch.linspace(-1, 1, w, device=f.device)
        xy = torch.cat([(p.sum(2) * xs).sum(-1), (p.sum(3) * ys).sum(-1)], dim=1)  # expected x and y of each point
        return torch.stack([head(xy) for head in self.heads], dim=1)  # batch, heads, joints


def device():
    return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, loader, dev) -> dict:
    model.eval()
    errs, spreads = [], []
    with torch.no_grad():
        for img, tgt in loader:
            out = model(img.to(dev)).cpu()
            errs.append((out.mean(1) - tgt).abs() * OUT_SCALE)
            spreads.append(out.std(1) * OUT_SCALE)
    if not errs:
        return {}
    e, s = torch.cat(errs), torch.cat(spreads)
    return {"mean_abs_counts": {m: round(float(v), 2) for m, v in zip(ARM, e.mean(0))},
            "worst_joint_mean": round(float(e.mean(0).max()), 2),
            "p90_worst_joint": round(float(e.max(1).values.quantile(0.9)), 2),
            "head_spread": round(float(s.mean()), 2), "n": len(e)}


def train(folder: Path, out_dir: Path, epochs: int, batch: int, lr: float, pretrained: bool = True, heads: int = 5) -> dict:
    spots = load_spots(folder)
    tr = [s for s in spots if not s["held_out"]]
    te = [s for s in spots if s["held_out"]]
    if not tr:
        raise SystemExit(f"no training spots under {folder}")
    centre = np.mean([s["hover"] for s in tr], axis=0).astype(np.float32)
    dev = device()
    model = FineNet(heads, pretrained).to(dev)
    train_set, test_set = Samples(tr, centre, True), Samples(te, centre, False)
    loader = torch.utils.data.DataLoader(train_set, batch_size=batch, shuffle=True, drop_last=len(train_set) > batch)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64) if len(test_set) else None
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(1, epochs * len(loader)))
    print(f"{len(tr)} training spots ({len(train_set)} images), {len(te)} held-out spots ({len(test_set)} images), device {dev}")
    for ep in range(epochs):
        model.train()
        total = 0.0
        for img, tgt in loader:
            out = model(img.to(dev))
            loss = (out - tgt.to(dev).unsqueeze(1)).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss.detach())
        line = f"epoch {ep + 1:>2}/{epochs}  train L1 {total / max(1, len(loader)) * OUT_SCALE:.2f} counts"
        if test_loader and (ep + 1) % 5 == 0:
            line += f"   held-out worst joint {evaluate(model, test_loader, dev).get('worst_joint_mean')} counts"
        print(line)
    report = {"train_spots": [s["spot"] for s in tr], "held_out_spots": [s["spot"] for s in te],
              "held_out": evaluate(model, test_loader, dev) if test_loader else {}}
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "centre": centre.tolist(), "heads": heads, "report": report,
                "crop": tr[0]["rows"][0].get("crop") if tr[0]["rows"] else None}, out_dir / "fine.pt")
    (out_dir / "fine_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["held_out"], indent=2))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train"])
    ap.add_argument("--data", default=str(ROOT / "data" / "collect"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "fine"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    a = ap.parse_args()
    train(Path(a.data), Path(a.out), a.epochs, a.batch, a.lr)


if __name__ == "__main__":
    main()


class Predictor:
    """Loads a trained network and turns a camera picture into a joint offset in counts."""

    def __init__(self, path: Path, crop=None):
        ck = torch.load(path, map_location="cpu")
        self.crop = crop if crop is not None else ck.get("crop")
        self.dev = device()
        self.model = FineNet(ck["heads"], pretrained=False).to(self.dev)
        self.model.load_state_dict(ck["model"])
        self.model.eval()

    def __call__(self, image_bgr: np.ndarray):
        """Returns (offset per joint, disagreement between heads per joint), both in counts."""
        img = (prepare(image_bgr, self.crop) - MEAN) / STD
        with torch.no_grad():
            out = self.model(torch.from_numpy(img.transpose(2, 0, 1)[None].astype(np.float32)).to(self.dev)).cpu()[0]
        return out.mean(0).numpy() * OUT_SCALE, out.std(0).numpy() * OUT_SCALE
