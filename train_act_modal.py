"""Train the ACT baseline on a Modal GPU from a dataset recorded on this Mac.

  modal run --detach train_act_modal.py --dataset plug_act --steps 30000
  modal volume get plug-sense act_plug_act/checkpoints/last/pretrained_model outputs/act_plug_act

The dataset is read from ~/.cache/huggingface/lerobot/local/<dataset> and uploaded to the
Modal volume "plug-sense". Checkpoints are written to the same volume under act_<dataset>/,
so they survive if this laptop sleeps. --detach keeps the job running after the terminal closes.
LeRobot is pinned to the commit installed on this Mac so the dataset format and policy match.
"""
from pathlib import Path

import modal

LEROBOT = "3f797de5952432284a7c77ad2760218cb74e2f26"
LOCAL = Path.home() / ".cache/huggingface/lerobot/local"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(f"lerobot @ git+https://github.com/huggingface/lerobot@{LEROBOT}")
)
app = modal.App("plug-sense-act", image=image)
vol = modal.Volume.from_name("plug-sense", create_if_missing=True)


@app.function(gpu="A10G", volumes={"/vol": vol}, timeout=8 * 3600, cpu=8, memory=32768)
def train(dataset: str, steps: int, batch_size: int, save_freq: int, chunk: int):
    import subprocess
    import threading
    import time

    out = f"/vol/act_{dataset}"
    stop = threading.Event()

    def keep_saving():  # make checkpoints visible outside the job while it runs
        while not stop.wait(120):
            vol.commit()

    threading.Thread(target=keep_saving, daemon=True).start()
    cmd = [
        "lerobot-train",
        f"--dataset.repo_id=local/{dataset}", f"--dataset.root=/vol/datasets/{dataset}",
        "--policy.type=act", "--policy.device=cuda", "--policy.push_to_hub=false",
        f"--policy.chunk_size={chunk}", f"--policy.n_action_steps={chunk}",
        f"--output_dir={out}", f"--job_name=act_{dataset}",
        f"--batch_size={batch_size}", f"--steps={steps}", f"--save_freq={save_freq}",
        "--num_workers=8", "--log_freq=100", "--wandb.enable=false",
    ]
    t0 = time.time()
    try:
        subprocess.run(cmd, check=True)
    finally:
        stop.set()
        vol.commit()
    print(f"done: {steps} steps in {(time.time() - t0) / 60:.1f} min -> volume plug-sense:{out[5:]}")


@app.local_entrypoint()
def main(dataset: str, steps: int = 30000, batch_size: int = 8, save_freq: int = 5000, chunk: int = 100,
         upload: bool = True):
    src = LOCAL / dataset
    if upload:
        if not (src / "meta" / "info.json").exists():
            raise SystemExit(f"no dataset at {src}")
        print(f"uploading {src} ...")
        with vol.batch_upload(force=True) as batch:
            batch.put_directory(str(src), f"/datasets/{dataset}")
    train.remote(dataset, steps, batch_size, save_freq, chunk)
