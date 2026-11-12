# Trained ACT checkpoints

Both were trained with `train_act_modal.py` for 30,000 steps (batch 8, chunk 100, two 1280x720 cameras) on a Modal A100.

- `act_70ep/` — trained on all 70 teleop demonstrations. This is the baseline used in the comparison.
- `act_46ep/` — trained on the first 46 demonstrations.

The weight files are stored with Git LFS (`git lfs pull` after cloning). Run one with:

    lerobot-rollout --strategy.type=base --policy.path=checkpoints/act_70ep --policy.device=mps ...
