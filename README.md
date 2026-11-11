# plug-sense

A low-cost arm (SO-101) that plugs in a USB-C connector, confirms the connection
from contact sound and servo load, works out why a failed attempt failed, and
retries.

Status: in progress. Nothing here has been measured yet.

## Idea

- **Electrical ground truth.** The host computer sees the moment the plug connects.
  That gives an exact label for every attempt.
- **Outcome detector.** A contact microphone on the fixed gripper jaw plus servo
  load and position classify each attempt as seated, jammed or missed. It is
  checked against the electrical label.
- **Recovery.** On a jam the arm replays its last half-second in reverse, shifts
  slightly and tries again. After three failed retries it stops and asks for help.
- **Learning from failures.** Attempts the detector flags are corrected by
  teleoperation and added to the training data.

## Hardware

- SO-101 follower and leader (Feetech STS3215), LeRobot 0.6.x
- Front camera and wrist camera
- Korg CM-300 contact microphone into a USB audio adapter (microphone jack)

## Layout

- `plugsense/` logger, detector and runner (to come)
- `data/` attempt logs (not tracked)
- `outputs/` checkpoints and figures (not tracked)
