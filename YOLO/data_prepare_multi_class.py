import os
import pandas as pd
import json
import shutil
import random
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import ast

# Define paths
csv_path = "/home/ai-user/Knee_det/group_csv/group1_postop_implants.csv"
source_img_dir = "/home/ai-user/AUTONOMOUS/Data_cropped"
dest_root = "/home/ai-user/Knee_det/DATA_YOLO/postop_implants_multi"

# Create destination directories
for split in ['train', 'test', 'val']:
    os.makedirs(os.path.join(dest_root, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(dest_root, 'labels', split), exist_ok=True)

# Read CSV file
df = pd.read_csv(csv_path)
# df = df.head(200)

# Robust parse of bbox strings
def parse_bboxes(bbox_str):
    if pd.isna(bbox_str):
        return []
    cleaned = bbox_str.replace('""', '"') if isinstance(bbox_str, str) else bbox_str
    try:
        data = json.loads(cleaned) if isinstance(cleaned, str) else cleaned
    except Exception:
        try:
            data = ast.literal_eval(cleaned)
        except Exception:
            return []
    return data if isinstance(data, list) else []

# Extract labels and build mapping
all_labels = []
for _, row in df.iterrows():
    boxes = parse_bboxes(row.get('bbox'))
    for box in boxes:
        labels = box.get('rectanglelabels')
        if isinstance(labels, list) and len(labels) > 0:
            all_labels.append(str(labels[0]).strip())
        elif isinstance(labels, str) and labels.strip():
            all_labels.append(labels.strip())

unique_labels = sorted(set([l for l in all_labels if l]))
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

# Save class names
with open(os.path.join(dest_root, 'classes.txt'), 'w') as f:
    for label in unique_labels:
        f.write(f"{label}\n")

print(f"Found classes ({len(unique_labels)}): {unique_labels}")

# Convert one row to YOLO annotations
def convert_to_yolo(row, img_dir):
    raw_path = row['image_path']
    if os.path.isabs(raw_path):
        image_path = raw_path
        rel_image_path = os.path.basename(raw_path)
    else:
        rel_image_path = raw_path
        image_path = os.path.join(img_dir, raw_path)

    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        return [], rel_image_path
    img_width, img_height = img.size

    boxes = parse_bboxes(row.get('bbox'))
    yolo_annotations = []
    for box in boxes:
        try:
            x_center = float(box['x']) + float(box['width']) / 2.0
            y_center = float(box['y']) + float(box['height']) / 2.0
            width = float(box['width'])
            height = float(box['height'])
        except Exception:
            continue

        # Normalize to 0-1
        x_center_norm = x_center / 100.0
        y_center_norm = y_center / 100.0
        width_norm = width / 100.0
        height_norm = height / 100.0

        # Map label
        labels = box.get('rectanglelabels')
        label = None
        if isinstance(labels, list) and len(labels) > 0:
            label = str(labels[0]).strip()
        elif isinstance(labels, str):
            label = labels.strip()
        if not label or label not in label_to_idx:
            continue
        class_idx = label_to_idx[label]

        yolo_annotations.append(f"{class_idx} {x_center_norm} {y_center_norm} {width_norm} {height_norm}")

    return yolo_annotations, rel_image_path

# Build list of items
data = []
for _, row in df.iterrows():
    yolo_annotations, rel_path = convert_to_yolo(row, source_img_dir)
    data.append({
        'image_path': row['image_path'],
        'rel_path': rel_path,
        'annotations': yolo_annotations,
    })

# Split data
train_ratio = 0.84
test_ratio = 0.01
val_ratio = 0.15
train_data, temp_data = train_test_split(data, test_size=(test_ratio + val_ratio), random_state=42)
val_fraction_of_temp = val_ratio / (test_ratio + val_ratio)
test_data, val_data = train_test_split(temp_data, test_size=val_fraction_of_temp, random_state=42)

# Copy and write labels
def process_split(split_data, split_name):
    for item in split_data:
        src_img = item['image_path'] if os.path.isabs(item['image_path']) else os.path.join(source_img_dir, item['image_path'])
        if not os.path.exists(src_img):
            print(f"[Warning] Source image not found, skipping: {src_img}")
            continue
        dst_img = os.path.join(dest_root, 'images', split_name, item['rel_path'])
        os.makedirs(os.path.dirname(dst_img), exist_ok=True)
        if os.path.abspath(src_img) != os.path.abspath(dst_img):
            shutil.copy2(src_img, dst_img)
        label_rel_path = os.path.splitext(item['rel_path'])[0] + '.txt'
        label_path = os.path.join(dest_root, 'labels', split_name, label_rel_path)
        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        with open(label_path, 'w') as f:
            f.write('\n'.join(item['annotations']))

process_split(train_data, 'train')
process_split(test_data, 'test')
process_split(val_data, 'val')

# Write data.yaml
yaml_content = """
train: {train}
val: {val}
test: {test}

nc: {nc}
names: {names}
""".format(
    train=os.path.join(dest_root, 'images', 'train'),
    val=os.path.join(dest_root, 'images', 'val'),
    test=os.path.join(dest_root, 'images', 'test'),
    nc=len(unique_labels),
    names=unique_labels,
)

with open(os.path.join(dest_root, 'data.yaml'), 'w') as f:
    f.write(yaml_content)

print(f"Dataset prepared with {len(unique_labels)} classes: {unique_labels}")
print(f"Train: {len(train_data)} images")
print(f"Test: {len(test_data)} images")
print(f"Val: {len(val_data)} images")

# Optional: simple validation plot

def validate_yolo_dataset(yolo_root_dir, num_samples=5):
    with open(os.path.join(yolo_root_dir, 'classes.txt'), 'r') as f:
        class_names = [line.strip() for line in f.readlines()]

    train_img_dir = os.path.join(yolo_root_dir, 'images', 'train')
    train_label_dir = os.path.join(yolo_root_dir, 'labels', 'train')

    all_images = []
    for root, _, files in os.walk(train_img_dir):
        for fname in files:
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                rel_path = os.path.relpath(os.path.join(root, fname), start=train_img_dir)
                all_images.append(rel_path)

    if len(all_images) == 0:
        print("No images found for validation plot.")
        return
    if len(all_images) < num_samples:
        num_samples = len(all_images)
        print(f"Warning: Only {num_samples} images available for validation")

    sample_images = random.sample(all_images, num_samples)

    plt.figure(figsize=(15, 15))
    for i, img_file in enumerate(sample_images):
        img_path = os.path.join(train_img_dir, img_file)
        img = Image.open(img_path)
        img_width, img_height = img.size
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(train_label_dir, label_file)

        plt.subplot(2, 3, i + 1)
        plt.imshow(np.array(img))
        plt.title(f"Image: {img_file}")

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * img_width
                    y_center = float(parts[2]) * img_height
                    width = float(parts[3]) * img_width
                    height = float(parts[4]) * img_height
                    x1 = x_center - width / 2
                    y1 = y_center - height / 2
                    rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
                    plt.gca().add_patch(rect)
                    class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
                    plt.text(x1, y1 - 5, class_name, color='red', fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
        else:
            plt.text(10, 10, "No label file found", color='red', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(yolo_root_dir, 'validation_samples.png'))
    plt.show()
    print(f"Validation image saved to {os.path.join(yolo_root_dir, 'validation_samples.png')}")

validate_yolo_dataset(dest_root, num_samples=5) 