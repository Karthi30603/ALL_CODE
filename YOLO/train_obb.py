#!/usr/bin/env python3
"""train_obb.py

Full-featured training script for Ultralytics YOLO *Oriented Bounding-Box* (OBB) models.

Features
--------
• Trains one or multiple OBB models (e.g. `yolov8m-obb.pt`, `yolo11m-obb.pt`).
• Supports parallel training on multiple GPUs or sequentially on CPU / single GPU.
• Automatic WandB logging (metrics table, final metrics dict, metric plots).
• Generates beautiful training-progress plots from `results.csv`.
• Pure Python – run with:  ``python train_obb.py --data /path/to/data.yaml``

Dataset
-------
Pass the YAML created by ``data_prepare_obb.py``, e.g.  ``.../data_anklel_obb/data.yaml``.
The YAML can optionally contain ``task: obb`` (otherwise the model’s weights determine the task).

Example
-------
```bash
python train_obb.py \
  --models yolov8m-obb.pt yolo11m-obb.pt \
  --data /home/ai-user/efficientdet/YOLO/data_anklel_obb/data.yaml \
  --project ankle_fracture_obb \
  --epochs 150 --batch 16
```
"""
from __future__ import annotations

import os
import argparse
import multiprocessing as mp
import sys
from pathlib import Path
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
# ----------------------------- Optional Weights & Biases ------------------------------------
try:
    import wandb
    _WANDB_AVAILABLE = hasattr(wandb, "init")
except ImportError:  # wandb not installed
    wandb = None  # type: ignore
    _WANDB_AVAILABLE = False
import torch

# ---------------------------------------------------------------------------------------------
# Plot styling ---------------------------------------------------------------------------------
plt.style.use("ggplot")

# ---------------------------------------------------------------------------------------------
# Core training util ---------------------------------------------------------------------------

def train_and_log_model(model_name: str, weight_path: str, data_yaml: str, project: str, run_name: str,
                        device: str | int, epochs: int, imgsz: int, batch: int) -> None:
    """Train a single OBB model and log everything to WandB."""

    # ----------------------- WandB initialisation --------------------------------------------
    if _WANDB_AVAILABLE:
        wandb.init(project=project, name=run_name, config=dict(
            model=model_name,
            weights=weight_path,
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            task="obb",
        ))
    else:
        print("[WARN] wandb unavailable or missing `.init`; proceeding without WandB logging.")

    # ----------------------- Model & Training -------------------------------------------------
    model = YOLO(weight_path, task="obb")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=run_name,
        save=True,
        plots=True,
        val=True,
    )

    # ----------------------- Metrics extraction ----------------------------------------------
    run_dir = Path(results.save_dir)
    results_csv = run_dir / "results.csv"

    if results_csv.exists():
        metrics_df = pd.read_csv(results_csv).reset_index().rename(columns={"index": "epoch"})
        metrics_df["epoch"] = metrics_df.index + 1
        metrics_df.to_csv(run_dir / "full_metrics.csv", index=False)
        print(f"[✔] {model_name} metrics saved → {run_dir / 'full_metrics.csv'}")

        plot_path = run_dir / "metrics_plot.png"
        _plot_metrics(metrics_df, model_name, plot_path)
        print(f"[✔] Metrics plot → {plot_path}")

        # ---------------- WandB Logging -------------------------------------------------------
        if _WANDB_AVAILABLE:
            wandb.log({f"{model_name}_metrics_plot": wandb.Image(str(plot_path))})
            wandb.log({f"{model_name}_metrics_table": wandb.Table(dataframe=metrics_df)})
            wandb.log({f"{model_name}_final_metrics": metrics_df.iloc[-1].to_dict()})
    else:
        print(f"[!] results.csv not found for {model_name} (looked in {results_csv})")

    if _WANDB_AVAILABLE:
        wandb.finish()


# ---------------------------------------------------------------------------------------------
# Plot helper ----------------------------------------------------------------------------------

def _plot_metrics(df: pd.DataFrame, title: str, save_path: Path):
    """Generate a 2×3 grid of key training curves."""
    plt.figure(figsize=(16, 10))

    # Box loss ---------------------------------------------------------------------------
    ax = plt.subplot(2, 3, 1)
    for col, label in [("train/box_loss", "Train"), ("val/box_loss", "Val")]:
        if col in df:
            ax.plot(df["epoch"], df[col], label=label)
    ax.set_title("Box Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend(); ax.grid(True)

    # Classification loss ----------------------------------------------------------------
    ax = plt.subplot(2, 3, 2)
    for col, label in [("train/cls_loss", "Train"), ("val/cls_loss", "Val")]:
        if col in df:
            ax.plot(df["epoch"], df[col], label=label)
    ax.set_title("Classification Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend(); ax.grid(True)

    # DFL loss ---------------------------------------------------------------------------
    ax = plt.subplot(2, 3, 3)
    for col, label in [("train/dfl_loss", "Train"), ("val/dfl_loss", "Val")]:
        if col in df:
            ax.plot(df["epoch"], df[col], label=label)
    ax.set_title("DFL Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend(); ax.grid(True)

    # mAP -------------------------------------------------------------------------------
    ax = plt.subplot(2, 3, 4)
    for col, label in [("metrics/mAP50(B)", "mAP@50"), ("metrics/mAP50-95(B)", "mAP@50:95")]:
        if col in df:
            ax.plot(df["epoch"], df[col], label=label)
    ax.set_title("Validation mAP")
    ax.set_xlabel("Epoch"); ax.set_ylabel("mAP"); ax.legend(); ax.grid(True)

    # Precision / Recall ----------------------------------------------------------------
    ax = plt.subplot(2, 3, 5)
    for col, label in [("metrics/precision(B)", "Precision"), ("metrics/recall(B)", "Recall")]:
        if col in df:
            ax.plot(df["epoch"], df[col], label=label)
    ax.set_title("Precision & Recall")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score"); ax.legend(); ax.grid(True)

    # Learning rate ----------------------------------------------------------------------
    ax = plt.subplot(2, 3, 6)
    if "lr/pg0" in df:
        ax.plot(df["epoch"], df["lr/pg0"], label="LR")
    ax.set_title("Learning Rate")
    ax.set_xlabel("Epoch"); ax.set_ylabel("LR"); ax.legend(); ax.grid(True)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------------------------
# Multiprocessing wrapper ----------------------------------------------------------------------

def _train_wrapper(args_tuple):
    """Unpack arguments for multiprocessing."""
    train_and_log_model(*args_tuple)


# ---------------------------------------------------------------------------------------------
# Main -----------------------------------------------------------------------------------------

def parse_args(args: List[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Train Ultralytics OBB models with WandB logging (all arguments optional)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["yolo11m-obb.pt"],
        help="OBB weight files to train (space-separated). default: %(default)s",
    )
    parser.add_argument(
        "--data",
        default="/home/ai-user/efficientdet/YOLO/data_anklel_obb/data.yaml",
        help="Path to OBB dataset YAML. default: %(default)s",
    )
    parser.add_argument(
        "--project",
        default="hand_wrist_fracture_obb-11",
        help="WandB project & directory to store runs. default: %(default)s",
    )
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs. default: %(default)s")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size. default: %(default)s")
    parser.add_argument("--batch", type=int, default=72, help="Batch size. default: %(default)s")
    return parser.parse_args(args)


def main(cli_args: List[str] | None = None):
    args = parse_args(cli_args)

    # GPU availability ------------------------------------------------------------------
    num_gpus = torch.cuda.device_count()
    devices: List[int | str]
    if num_gpus == 0:
        devices = ["cpu"] * len(args.models)
        print("[!] No GPUs detected – training will run on CPU.")
    elif num_gpus == 1:
        devices = [0] * len(args.models)
        if len(args.models) > 1:
            print("[!] Only 1 GPU – models will train sequentially.")
    else:
        # Cycle through available GPUs for each model
        devices = [i % num_gpus for i in range(len(args.models))]

    # Build training argument tuples -----------------------------------------------------
    train_args = []
    for idx, (model_weight, device) in enumerate(zip(args.models, devices)):
        model_name = Path(model_weight).stem
        run_name = f"{model_name}_run"
        train_args.append((
            model_name,
            model_weight,
            args.data,
            args.project,
            run_name,
            device,
            args.epochs,
            args.imgsz,
            args.batch,
        ))

    # Decide on parallel / sequential execution -----------------------------------------
    if num_gpus >= 2 and len(args.models) > 1:
        print(f"[✔] Starting parallel training on {len(train_args)} processes…")
        mp.set_start_method("spawn", force=True)
        procs = [mp.Process(target=_train_wrapper, args=(t,)) for t in train_args]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
    else:
        print("[✔] Starting sequential training…")
        for t in train_args:
            _train_wrapper(t)

    print("\n🏁 All training runs completed.")


if __name__ == "__main__":
    main() 