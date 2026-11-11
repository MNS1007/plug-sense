"""Record labelled insertion attempts by teleop: servo signals and contact-mic audio.

  python -m plugsense.attempt_log --mic-device 1

One run records many trials. For each trial:
  1. Press Enter to start. Drive the leader arm and do the attempt.
  2. Press Enter the moment the plug seats, or when you give up.
  3. Type what happened:
       s   seated (plug fully in)
       j   jammed: touched the port but did not go in. Add the side the plug was
           off toward, seen from the front camera: jl, jr, ju, jd (left/right/up/down)
       m   missed: never touched the port
       f   free motion: arm moved, nothing touched (motor-noise reference)
       x   discard this trial
       q   save nothing and quit

Each trial is a folder under data/trials/ with signals.csv, audio.wav and meta.json.
All times are time.monotonic() seconds. This uses both arms' serial ports, so close
lerobot-teleoperate first.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

FOLLOWER_PORT = "/dev/tty.usbmodem5B790502351"
LEADER_PORT = "/dev/tty.usbmodem5B790499721"
REGISTERS = ["Present_Position", "Present_Load", "Present_Current"]
LABELS = {"s": "seated", "j": "jammed", "m": "missed", "f": "free"}
SIDES = {"l": "left", "r": "right", "u": "up", "d": "down"}
RATE = 16000


class Mic:
    """Continuous recording; each block is stamped with the time it arrived."""

    def __init__(self, device):
        import sounddevice as sd

        self.blocks: list[tuple[float, np.ndarray]] = []
        self.stream = sd.InputStream(
            samplerate=RATE, channels=1, dtype="int16", device=device, blocksize=RATE // 50, callback=self._cb
        )
        self.stream.start()

    def _cb(self, indata, frames, t, status):
        self.blocks.append((time.monotonic(), indata[:, 0].copy()))

    def cut(self, t_start: float, t_stop: float):
        """Audio between two times, and the time of its first sample."""
        picked = [(t, b) for t, b in self.blocks if t_start <= t <= t_stop + 0.05]
        self.blocks = [(t, b) for t, b in self.blocks if t > t_stop]
        if not picked:
            return np.zeros(0, np.int16), t_start
        first_t, first_b = picked[0]
        return np.concatenate([b for _, b in picked]), first_t - len(first_b) / RATE

    def close(self):
        self.stream.stop()
        self.stream.close()


def wait_enter() -> threading.Event:
    done = threading.Event()
    threading.Thread(target=lambda: (sys.stdin.readline(), done.set()), daemon=True).start()
    return done


def ask_label():
    while True:
        raw = input("what happened? [s / j l|r|u|d / m / f / x / q]: ").strip().lower().replace(" ", "")
        if raw in ("x", "q"):
            return raw, None
        if raw[:1] in LABELS and (len(raw) == 1 or (raw[0] == "j" and raw[1:] in SIDES)):
            return LABELS[raw[0]], SIDES.get(raw[1:])
        print("  not understood")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mic-device", default=None, help="sounddevice input index or name; omit for no audio")
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--max-seconds", type=float, default=30.0)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "data" / "trials"))
    ap.add_argument("--note", default="", help="free text saved with every trial, e.g. 'usb-c, port up'")
    a = ap.parse_args()

    robot = SO101Follower(SO101FollowerConfig(port=FOLLOWER_PORT, id="follower", max_relative_target=10.0))
    leader = SO101Leader(SO101LeaderConfig(port=LEADER_PORT, id="leader"))
    robot.connect()
    leader.connect()
    motors = list(robot.bus.motors)
    mic = None
    if a.mic_device is not None:
        mic = Mic(int(a.mic_device) if a.mic_device.isdigit() else a.mic_device)

    fields = ["t"] + [f"cmd_{m}" for m in motors] + [f"{r}_{m}" for r in REGISTERS for m in motors]
    counts: dict[str, int] = {}
    period = 1.0 / a.hz
    try:
        while True:
            input(f"\n{counts or 'no trials yet'}  Enter = start a trial (Ctrl+C = finish) ")
            stop = wait_enter()
            print("RECORDING. Press Enter when seated or when you give up.")
            rows, t_start = [], time.monotonic()
            while not stop.is_set() and (now := time.monotonic()) - t_start < a.max_seconds:
                action = leader.get_action()
                robot.send_action(action)
                row = {"t": now}
                row.update({f"cmd_{m}": action.get(f"{m}.pos") for m in motors})
                for r in REGISTERS:
                    try:
                        vals = robot.bus.sync_read(r, motors, normalize=False)
                    except Exception:
                        vals = {}
                    row.update({f"{r}_{m}": vals.get(m) for m in motors})
                rows.append(row)
                time.sleep(max(0.0, period - (time.monotonic() - now)))
            t_stop = time.monotonic()
            if not stop.is_set():
                print(f"stopped at {a.max_seconds:.0f} s. Press Enter to continue.")
                stop.wait()
            rate = len(rows) / max(t_stop - t_start, 1e-6)
            print(f"{t_stop - t_start:.1f} s, {len(rows)} samples ({rate:.0f} Hz)")

            label, side = ask_label()
            if label == "q":
                break
            if label == "x":
                if mic:
                    mic.cut(t_start, t_stop)
                continue
            name = label + (f"-{side}" if side else "")
            trial = Path(a.out) / f"{time.strftime('%Y%m%d-%H%M%S')}-{name}"
            trial.mkdir(parents=True, exist_ok=True)
            with open(trial / "signals.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
            meta = {
                "label": label, "side": side, "note": a.note, "motors": motors, "registers": REGISTERS,
                "t_start": t_start, "t_stop": t_stop, "samples": len(rows), "hz": round(rate, 1),
            }
            if mic:
                audio, audio_t0 = mic.cut(t_start, t_stop)
                with wave.open(str(trial / "audio.wav"), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(RATE)
                    w.writeframes(audio.tobytes())
                peak = int(np.abs(audio).max()) if len(audio) else 0
                meta.update(audio_t0=audio_t0, audio_rate=RATE, audio_samples=int(len(audio)), audio_peak=peak)
                if peak >= 32000:
                    print("  audio clipped: lower the input volume")
            (trial / "meta.json").write_text(json.dumps(meta, indent=2))
            counts[name] = counts.get(name, 0) + 1
            print(f"saved {trial.name}")
    except KeyboardInterrupt:
        print("\nfinished")
    finally:
        if mic:
            mic.close()
        leader.disconnect()
        robot.disconnect()
    print(f"totals: {counts}")


if __name__ == "__main__":
    main()
