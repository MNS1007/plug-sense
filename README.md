# plug-sense

A self-correction harness for ACT on a USB-C plug-in task, tested on an SO-101 arm.

A trained policy cannot be shown every edge case in advance. This is a small test of
another way to handle them: check whether the task succeeded and recover when it did
not, without retraining.

ACT often reaches the port but does not get the plug to latch, and it has no way of
knowing. The harness reads the power bank's display after each attempt. If the plug
did not latch, the arm backs out and retries with a motion taught from one
teleoperated demonstration. The ACT weights are not changed.

Results page with videos: https://mns1007.github.io/plug-sense/

## Results

| Condition | Latched | Time to latch |
|---|---|---|
| ACT alone | 7 / 20 | 34 s average |
| ACT with the harness | 15 / 20 | about 60 s when a retry was needed |
| Taught motion alone, right after teaching | 5 / 5 | 10.7 s |

The two 20-trial sets were counted by hand, on the same setup with the same ACT
checkpoint (50 demonstrations, 20,000 steps). An ACT-alone trial was stopped and
counted as failed after 90 s. The 5-trial run was logged by `plugsense.retry`.

## How it works

1. **Read the display.** A fixed front camera watches a box around the power bank's
   display. The reading is the share of pixels in the box whose brightest colour
   channel is at least 215 (`plugsense/display.py`).
2. **Decide.** A reading is taken just before the attempt and again for up to 4 s
   after it. The plug counts as latched if the reading rose by at least 0.04.
3. **Retry.** If it did not latch, the arm backs out and replays one demonstration:
   the commanded joint targets at 30 Hz with the original timing, and the same
   gripper squeeze (`plugsense/demo.py`). Commands are replayed, not the measured
   path, so the push at contact matches the demonstration. The display is checked
   again after each try.

Every attempt is saved with its outcome, servo positions, loads and a photo.

## Hardware

- SO-101 follower and leader arms (Feetech STS3215), LeRobot 0.6.x
- Front camera (fixed, looking down at the port) and wrist camera, both 1280x720
- A power bank with a display that lights up when charging starts
- USB-C plug fixed rigidly to the gripper jaw. With a single loose zip tie the plug
  hinged in the gripper and neither ACT nor the replay worked reliably.

## Running it

Record demonstrations with `lerobot-record`, then train ACT on Modal:

    modal run --detach train_act_modal.py --dataset <dataset> --steps 20000

Teach the retry motion once and test it on its own:

    python -m plugsense.retry --trials 5

Run ACT with the harness (uses the last taught motion):

    python -m plugsense.harness --policy checkpoints/act_stiff_50ep --dataset <dataset> --trials 10

Add up logged retry sessions:

    python -m plugsense.report

Keep a hand near the arm's power switch when any of these are running.

## Layout

- `plugsense/harness.py` ACT attempt, display check, retry
- `plugsense/retry.py` the taught motion with the display check, on its own
- `plugsense/demo.py` teaching by teleoperation, replay, pull-out
- `plugsense/display.py` display reading
- `plugsense/arm.py`, `plugsense/cameras.py` motor and camera helpers
- `plugsense/report.py` tally of logged sessions
- `train_act_modal.py` ACT training on a Modal GPU
- `checkpoints/` trained ACT models (Git LFS, run `git lfs pull`)
- `samples/` two teleoperated demonstrations with video and joint data
- `docs/` the results page
- `tests/` checks that run without hardware (`pytest`)

`gate.py`, `collect.py`, `fine.py`, `attempt_log.py` and `probe_link.py` are earlier
experiments (replay tests, self-collected data, a camera-correction network, sound
logging). They are not used in the results above.

## Limitations

- One fixed port location. The retry is a replay, so it does not adapt if the port moves.
- 20 trials per condition, counted by hand.
- Not yet tried: retraining ACT on the saved attempts, or a second latch signal such
  as the contact microphone.
