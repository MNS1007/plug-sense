# Sample teleop demonstrations

Two of the 70 demonstrations used to train the ACT baseline (episodes 61 and 64 of the dataset).
Each is one plug-in by teleoperation, about 10 seconds, ending with the power bank display lit.

- `front.mp4` — fixed camera looking down at the power bank, 1280x720, 30 frames a second
- `wrist.mp4` — camera on the arm, same size and rate
- `joints.csv` — for every frame: the commanded joint position (`action_*`) and the measured one (`state_*`), in degrees

The videos are re-encoded to H.264 so they play in a browser; the training data itself is unchanged.
