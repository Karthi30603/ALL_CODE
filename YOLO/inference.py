import argparse
import os
from pathlib import Path

import yaml
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from tqdm import tqdm

# ---------------------------------------
# Utility functions
# ---------------------------------------

def draw_boxes(image: Image.Image, boxes: list, names: list) -> Image.Image:
    """Draw bounding boxes on an image using Pillow."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    draw = ImageDraw.Draw(image)

    for det in boxes:
        cls, x1, y1, x2, y2, conf = det
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        if x2 <= x1 or y2 <= y1:
            continue

        # Rectangle
        color = (255, 0, 0) if int(cls) == 0 else (0, 255, 0)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        label = f"{names[int(cls)]} {conf:.2f}"
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except IOError:
            font = ImageFont.load_default()
        try:
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
        except AttributeError:  # Fall back for very old Pillow
            text_w, text_h = font.getsize(label)
        draw.rectangle([
            x1, y1 - text_h - 4,
            x1 + text_w + 4, y1
        ], fill=color)
        draw.text((x1 + 2, y1 - text_h - 2), label, fill="white", font=font)

    return image


# ---------------------------------------
# Main inference routine
# ---------------------------------------

def predict(model: YOLO, img_path: str, names: list, conf_thres: float):
    pil_img = Image.open(img_path).convert("RGB")

    # Run inference directly on the PIL image
    res = model(pil_img, conf=conf_thres)[0]
    preds = []
    for *xyxy, conf, cls in res.boxes.data.cpu().numpy():
        x1, y1, x2, y2 = xyxy
        preds.append([int(cls), x1, y1, x2, y2, float(conf)])

    vis_img = draw_boxes(pil_img.copy(), preds, names) if preds else pil_img
    return preds, vis_img


def main():
    parser = argparse.ArgumentParser(description="YOLO two-class (Fracture/Normal) inference script")
    parser.add_argument("--weights", required=True, help="/home/ai-user/efficientdet/YOLO/code/H&W_YOLO/yolo8m_run_new/weights/best.pt")
    parser.add_argument("--data_yaml", required=True, help="/home/ai-user/efficientdet/YOLO/data_anklel/data.yaml")
    parser.add_argument("--input", required=True, help="/home/ai-user/efficientdet/MYHIL/mydata")
    parser.add_argument("--output", default="/home/ai-user/efficientdet/YOLO/results/yolo8m-myhil-0.1", help="Folder for annotated images & CSV")
    parser.add_argument("--conf", type=float, default=0.1, help="Confidence threshold")
    args = parser.parse_args()

    # Load class names
    with open(args.data_yaml, "r") as f:
        names = yaml.safe_load(f)["names"]
    if len(names) != 2:
        print("Warning: data YAML does not contain 2 classes as expected.")

    # Prepare output paths
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "predictions.csv"

    # New: create sub-folders for TP, TN, FP, FN visualisations
    case_dirs = {}
    for c in ["TP", "TN", "FP", "FN"]:
        subdir = out_dir / c
        subdir.mkdir(exist_ok=True)
        case_dirs[c] = subdir

    # Load model
    model = YOLO(args.weights)

    # Gather images
    if os.path.isdir(args.input):
        img_paths = [str(p) for p in Path(args.input).rglob("*") if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
    else:
        img_paths = [args.input]

    if not img_paths:
        print(f"No images found in {args.input}")
        return

    results_rows = []

    # Helper to decide ground-truth label from path
    def infer_gt(path: str):
        """Return 'Fracture' if the file path contains 'fracture' (case-insensitive) else 'Normal'."""
        return "Fracture" if "fracture" in path.lower() else "Normal"

    for img_path in tqdm(img_paths, desc="Images"):
        preds, vis_img = predict(model, img_path, names, args.conf)

        # Ground truth from directory name
        gt_label = infer_gt(img_path)

        # Determine prediction presence (any detection => Fracture)
        pred_positive = bool(preds)
        gt_positive = gt_label.lower().startswith("fracture")

        if gt_positive and pred_positive:
            case = "TP"
        elif gt_positive and not pred_positive:
            case = "FN"
        elif not gt_positive and pred_positive:
            case = "FP"
        else:
            case = "TN"

        # Save visualisation in the appropriate sub-folder
        vis_name = Path(img_path).stem + "_pred.jpg"
        vis_img.save(case_dirs[case] / vis_name)

        # Summarise prediction as a string for CSV
        pred_str = "; ".join([f"{names[p[0]]}:{p[5]:.2f}" for p in preds]) if preds else "Normal"

        results_rows.append({
            "image": os.path.basename(img_path),
            "ground_truth": gt_label,
            "predictions": pred_str,
            "case": case
        })

    df = pd.DataFrame(results_rows)
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")

    # ---------------------
    # Compute classification metrics
    # ---------------------
    def is_fracture(label: str):
        return label.lower().startswith("fracture")

    df["gt_binary"] = df["ground_truth"].apply(is_fracture)
    df["pred_binary"] = df["predictions"].apply(is_fracture)

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
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main() 
