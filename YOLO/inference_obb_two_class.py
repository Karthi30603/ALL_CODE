import argparse
import os
from pathlib import Path
import math

import yaml
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from ultralytics.utils import ops
import numpy as np
from tqdm import tqdm

# --------------------------------------------------
# Utility functions
# --------------------------------------------------

def draw_obb(image: Image.Image, obb_list: list, names: list, color_pred: bool) -> Image.Image:
    """Draw oriented bounding boxes on a PIL image.

    Args:
        image (PIL.Image): Image to annotate.
        obb_list (list): List of [cls, xc, yc, w, h, theta, conf] (all pixel units except `theta`).
        names (list): Class names.
        color_pred (bool): If True use red/green for predictions, else fixed green (GT).
    Returns:
        PIL.Image with drawn OBBs.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    draw = ImageDraw.Draw(image)

    for obb in obb_list:
        cls, xc, yc, w, h, theta, *rest = obb
        conf = rest[0] if rest else None

        # Compute 4 corner points (xyxyxyxy)
        corners = ops.xywhr2xyxyxyxy(np.array([[xc, yc, w, h, theta]], dtype=float))[0]  # (4,2)
        corners = [(int(x), int(y)) for x, y in corners]

        # Choose colour
        if color_pred:
            color = (255, 0, 0) if int(cls) == 0 else (0, 255, 0)
        else:
            color = (0, 255, 0)  # GT always green

        # Draw polygon
        draw.polygon(corners, outline=color)

        # Prepare label text
        if conf is not None:
            label = f"{names[int(cls)]} {conf:.2f}"
        else:
            label = f"GT {names[int(cls)]}"

        # Determine text size
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except IOError:
            font = ImageFont.load_default()
        try:
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
        except AttributeError:  # Older Pillow fallback
            text_w, text_h = font.getsize(label)

        # Position label near first corner
        x1, y1 = corners[0]
        draw.rectangle([x1, y1 - text_h - 4, x1 + text_w + 4, y1], fill=color)
        draw.text((x1 + 2, y1 - text_h - 2), label, fill="white", font=font)

    return image


# --------------------------------------------------
# Ground-truth helpers
# --------------------------------------------------

def parse_label_obb(label_path: Path, img_size: tuple) -> list:
    """Parse a YOLO-OBB txt label file.

    Returns list of [cls, xc_px, yc_px, w_px, h_px, theta_rad]
    """
    w_img, h_img = img_size
    boxes = []
    if not label_path.exists():
        return boxes

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 6:
                continue
            cls, xc, yc, bw, bh, theta = map(float, parts)
            # Convert to pixel units
            xc *= w_img
            yc *= h_img
            bw *= w_img
            bh *= h_img
            # Theta: convert deg->rad if appears in degrees
            if abs(theta) > math.pi:  # assume degrees
                theta = math.radians(theta)
            boxes.append([int(cls), xc, yc, bw, bh, theta])
    return boxes


def load_gt_obb(img_path: str, names: list) -> list:
    """Load ground-truth OBB boxes corresponding to an image path."""
    path = Path(img_path)
    try:
        idx = path.parts.index("images")
        label_parts = list(path.parts)
        label_parts[idx] = "labels"
        label_path = Path(*label_parts).with_suffix(".txt")
    except ValueError:
        return []  # Not under images folder

    img = Image.open(img_path)
    boxes = parse_label_obb(label_path, img.size)
    return boxes


# --------------------------------------------------
# Prediction helper
# --------------------------------------------------

def predict(model: YOLO, img_path: str, names: list, conf_thres: float):
    pil_img = Image.open(img_path).convert("RGB")
    res = model(pil_img, conf=conf_thres)[0]

    preds = []
    if res.obb is not None:
        obb_data = res.obb.data.cpu().numpy()  # (N,7)
        for xc, yc, bw, bh, theta, conf, cls in obb_data:
            preds.append([int(cls), xc, yc, bw, bh, theta, conf])
    # Visualise predictions only
    vis_img = draw_obb(pil_img.copy(), preds, names, color_pred=True) if preds else pil_img
    return preds, vis_img


# --------------------------------------------------
# Main routine
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="YOLO-OBB two-class (Fracture/Normal) inference script"
    )
    parser.add_argument("--weights", required=True,
                        help="Path to trained OBB weights (e.g. yolov8m-obb.pt)")
    parser.add_argument("--data_yaml", required=True, help="Dataset YAML used for class names")
    parser.add_argument("--input", required=True, help="Image file or directory containing images")
    parser.add_argument("--output", default="results_obb", help="Folder for annotated images & CSV")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--map_dir", default="map_results_obb", help="Folder to save mAP metrics & visualisations")
    args = parser.parse_args()

    # Load class names
    with open(args.data_yaml, "r") as f:
        names = yaml.safe_load(f)["names"]
    if len(names) != 2:
        print("[WARN] Expected 2 classes but data YAML has {}".format(len(names)))

    # Prepare directories
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "predictions.csv"

    map_dir = Path(args.map_dir)
    map_dir.mkdir(parents=True, exist_ok=True)

    # Create TP/TN/FP/FN sub-folders
    case_dirs = {}
    for c in ["TP", "TN", "FP", "FN"]:
        subdir = out_dir / c
        subdir.mkdir(exist_ok=True)
        case_dirs[c] = subdir

    # Load model
    model = YOLO(args.weights, task="obb")

    # Collect images
    if os.path.isdir(args.input):
        img_paths = [str(p) for p in Path(args.input).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    else:
        img_paths = [args.input]

    if not img_paths:
        print(f"[!] No images found in {args.input}")
        return

    results_rows = []

    def infer_gt_label(path_str: str):
        return "Fracture" if "fracture" in path_str.lower() else "Normal"

    for img_path in tqdm(img_paths, desc="Images"):
        preds, vis_pred_img = predict(model, img_path, names, args.conf)

        # Ground truth boxes
        gt_boxes = load_gt_obb(img_path, names)

        if gt_boxes:
            gt_label = "Fracture" if any(b[0] == 0 for b in gt_boxes) else "Normal"
        else:
            gt_label = infer_gt_label(img_path)

        pred_positive = any(p[0] == 0 for p in preds)
        gt_positive = gt_label.lower().startswith("fracture")

        if gt_positive and pred_positive:
            case = "TP"
        elif gt_positive and not pred_positive:
            case = "FN"
        elif not gt_positive and pred_positive:
            case = "FP"
        else:
            case = "TN"

        # Combine GT and predictions for a single visualisation
        raw_img = Image.open(img_path).convert("RGB")
        combined_img = draw_obb(raw_img, gt_boxes, names, color_pred=False)
        combined_img = draw_obb(combined_img, preds, names, color_pred=True)

        # Prediction summary string
        if preds:
            pred_str = "; ".join([f"{names[p[0]]}:{p[-1]:.2f}" for p in preds])
        else:
            pred_str = "Normal"

        results_rows.append({
            "image": os.path.basename(img_path),
            "ground_truth": gt_label,
            "predictions": pred_str,
            "case": case
        })

        vis_name = Path(img_path).stem + "_gt_pred.jpg"
        combined_img.save(case_dirs[case] / vis_name)
        combined_img.save(map_dir / vis_name)  # optional copy

    # Save CSV
    df = pd.DataFrame(results_rows)
    df.to_csv(csv_path, index=False)
    print(f"[✔] Saved CSV results → {csv_path}")

    # Classification metrics
    def is_frac(label: str):
        return label.lower().startswith("fracture")

    df["gt_binary"] = df["ground_truth"].apply(is_frac)
    df["pred_binary"] = df["predictions"].apply(is_frac)

    TP = int(((df.gt_binary) & (df.pred_binary)).sum())
    TN = int(((~df.gt_binary) & (~df.pred_binary)).sum())
    FP = int((~df.gt_binary & df.pred_binary).sum())
    FN = int((df.gt_binary & ~df.pred_binary).sum())

    accuracy = (TP + TN) / len(df) if len(df) else 0
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    metrics = {
        "Total": len(df),
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4)
    }

    metrics_df = pd.DataFrame([metrics])
    metrics_path = out_dir / "metrics_summary.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print("Classification metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"[✔] Metrics saved → {metrics_path}")

    # ---------------- mAP metrics ------------------
    print("\nRunning YOLO validation to compute mAP …")
    val_metrics = model.val(data=args.data_yaml, split="test", conf=args.conf)

    map50_95 = getattr(val_metrics.box, "map", None)
    map50 = getattr(val_metrics.box, "map50", None)

    map_results = {
        "mAP50-95": round(map50_95, 4) if map50_95 is not None else None,
        "mAP50": round(map50, 4) if map50 is not None else None
    }

    map_df = pd.DataFrame([map_results])
    map_csv = map_dir / "map_metrics.csv"
    map_df.to_csv(map_csv, index=False)
    print("mAP metrics:")
    for k, v in map_results.items():
        print(f"  {k}: {v}")
    print(f"[✔] mAP metrics saved → {map_csv}")


if __name__ == "__main__":
    main() 