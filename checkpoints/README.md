# Trained ACT checkpoints

All were trained with `train_act_modal.py` (batch 8, chunk 100, two 1280x720 cameras). The two older ones ran 30,000 steps on a Modal A100.

- `act_stiff_50ep/` — trained on 50 demonstrations recorded after the plug was fixed rigidly to the gripper (20,000 steps, Modal H100). This is the baseline used in the comparison.
- `act_70ep/` — trained on the first 70 teleop demonstrations, with the plug loosely zip-tied. Scored 0 of 10 on the arm.
- `act_46ep/` — trained on the first 46 demonstrations.

The weight files are stored with Git LFS (`git lfs pull` after cloning). Run one with:

    lerobot-rollout --strategy.type=base --policy.path=checkpoints/act_70ep --policy.device=mps ...
