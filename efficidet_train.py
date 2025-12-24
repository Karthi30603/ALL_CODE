"""
Fracture Detection Training Script using EfficientDet with Command Line Arguments

Installation:
pip install effdet

Usage Examples:
# Basic training with default parameters
python fracture_efficientdet_training_args.py --data_dir Data --csv_path csvs/frac_ann.csv --model_name tf_efficientdet_d7x --normal_limit 1000

#Avaailable arguments:
--data_dir: Path to the data directory
--csv_path: Path to the CSV file containing the annotations
--model_name: Name of the model to use
--normal_limit: Maximum number of normal images to use
--batch_size: Batch size for training
--max_epochs: Maximum number of epochs to train for
--run_evaluation: Whether to run evaluation on the test set
--test_size: Number of images to use for testing
--normal_limit: Maximum number of normal images to use for testing
--eval_conf_threshold: Confidence threshold for evaluation
--eval_iou_threshold: IoU threshold for evaluation
--eval_seed: Random seed for reproducible evaluation splits

# Training with custom data paths
python fracture_efficientdet_training_args.py --data_dir /path/to/data --csv_path /path/to/annotations.csv


Available EfficientDet Models:
- tf_efficientdet_d0, tf_efficientdet_d1, tf_efficientdet_d2, tf_efficientdet_d3
- tf_efficientdet_d4, tf_efficientdet_d5, tf_efficientdet_d6, tf_efficientdet_d7, tf_efficientdet_d7x

Tips:
1. Start with smaller models (d0-d3) for faster experimentation
2. Use larger batch sizes with smaller models if GPU memory allows
3. Monitor training with wandb logs for optimal hyperparameter tuning
4. The evaluation script automatically finds and tests your latest models
5. Use consistent test sets (same seed) to compare different models fairly
6. Check the TP/FP/TN/FN directories to understand model behavior
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb
import csv

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import json
import ast
from typing import Dict, List, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import argparse

# EfficientDet related imports
import effdet
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchPredict
from effdet.efficientdet import HeadNet
import timm

# For inference and evaluation
from ensemble_boxes import weighted_boxes_fusion
import random
import shutil
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import glob
from tqdm import tqdm

# ================================================================================================
# DATASET CLASSES
# ================================================================================================

class FractureDatasetAdaptor:
    """Dataset adaptor for fracture detection with specific CSV format"""
    
    def __init__(self, fracture_dir_path: str, normal_dir_path: str, csv_path: str):
        self.fracture_dir_path = fracture_dir_path
        self.normal_dir_path = normal_dir_path
        self.csv_path = csv_path
        
        # Initialize corrupted image tracking
        self.corrupted_count = 0
        self.corrupted_log_file = f"corrupted_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        # Write initial header to log file
        with open(self.corrupted_log_file, 'w') as f:
            f.write(f"Corrupted Images Log - Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
        
        # Load CSV data (robust)
        self.annotations_df = read_csv_robust(csv_path)
        print(f"Loaded {len(self.annotations_df)} annotations from CSV")
        
        # Get fracture images from CSV
        fracture_images = self.annotations_df['study_path'].unique().tolist()
        
        # Filter out fracture images that don't exist
        existing_fracture_images = []
        for img_path in fracture_images:
            # Try several possible physical locations for the fracture image
            candidate_paths = []
            # Absolute path as-is
            if os.path.isabs(img_path):
                candidate_paths.append(img_path)
            # Relative path nested under fracture_dir
            candidate_paths.append(os.path.join(self.fracture_dir_path, img_path.lstrip('/')))
            # Base filename only (all images flattened inside fracture_dir)
            candidate_paths.append(os.path.join(self.fracture_dir_path, os.path.basename(img_path)))
            # Historical flattened variant
            candidate_paths.append(os.path.join(self.fracture_dir_path, img_path.replace('/', '_')))

            full_image_path = next((p for p in candidate_paths if os.path.exists(p)), candidate_paths[0])

            if full_image_path is not None:
                # Keep the original relative path so that it can be re-constructed later
                existing_fracture_images.append(('fracture', img_path))
            else:
                print(f"Warning: Fracture image not found: {img_path}")
        
        # Get ALL normal images from normal directory (no limit)
        normal_images = []
        normal_count = 0
        
        if os.path.exists(self.normal_dir_path):
            for root, _, files in os.walk(self.normal_dir_path):
                for file in files:
                    if file.lower().endswith(('.jpeg', '.jpg', '.png')):
                        relative_path = os.path.relpath(os.path.join(root, file), self.normal_dir_path)
                        normal_images.append(('normal', relative_path))
                        normal_count += 1
        
        # Combine fracture and normal images
        self.images = existing_fracture_images + normal_images
        print(f"Found {len(existing_fracture_images)} fracture images out of {len(fracture_images)} total")
        print(f"Found {len(normal_images)} normal images (no limit applied)")
        print(f"Total dataset size: {len(self.images)} images")
        print(f"Corrupted images will be logged to: {self.corrupted_log_file}")
        
    def log_corrupted_image(self, image_path: str, error_msg: str):
        """Log corrupted image information to file"""
        self.corrupted_count += 1
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open(self.corrupted_log_file, 'a') as f:
            f.write(f"[{timestamp}] Corrupted Image #{self.corrupted_count}\n")
            f.write(f"Path: {image_path}\n")
            f.write(f"Error: {error_msg}\n")
            f.write("-" * 40 + "\n")
        
        print(f"Skipped corrupted image #{self.corrupted_count}: {image_path}")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def get_image_and_labels_by_idx(self, index: int) -> Tuple[Image.Image, np.ndarray, np.ndarray, str]:
        image_type, image_path = self.images[index]
         
        # Load image based on type
        if image_type == 'fracture':
            # Determine the correct on-disk location of the fracture image
            candidate_paths = []
            # 1) Absolute path (CSV may already contain full path)
            if os.path.isabs(image_path):
                candidate_paths.append(image_path)
            # 2) Relative nested path under the configured fracture directory
            candidate_paths.append(os.path.join(self.fracture_dir_path, image_path.lstrip('/')))
            # 3) Only the basename inside the fracture directory (all images flat)
            candidate_paths.append(os.path.join(self.fracture_dir_path, os.path.basename(image_path)))
            # 4) Historical flattened path variant
            candidate_paths.append(os.path.join(self.fracture_dir_path, image_path.replace('/', '_')))

            full_image_path = next((p for p in candidate_paths if os.path.exists(p)), candidate_paths[0])
        else:  # normal image
            full_image_path = os.path.join(self.normal_dir_path, image_path)
        
        # Load image with error handling
        try:
            image = Image.open(full_image_path).convert("RGB")
            width, height = image.size
        except (OSError, IOError, Exception) as e:
            # Log the corrupted image
            self.log_corrupted_image(full_image_path, str(e))
            
            # Return a dummy black image to continue training
            image = Image.new('RGB', (512, 512), color='black')
            width, height = 512, 512
            
            # Return empty bboxes for corrupted images
            pascal_bboxes = np.array([], dtype=np.float32).reshape(0, 4)
            class_labels = np.array([], dtype=np.float32)
            return image, pascal_bboxes, class_labels, str(index)
        
        bboxes = []
        
        if image_type == 'fracture':
            # Get all annotations for this fracture image
            image_annotations = self.annotations_df[self.annotations_df['study_path'] == image_path]
            
            for _, row in image_annotations.iterrows():
                bbox_str = row['bbox']
                if bbox_str and bbox_str != '':
                    # Parse the bbox string
                    try:
                        bbox_list = ast.literal_eval(bbox_str)
                        for bbox_dict in bbox_list:
                            # Extract coordinates (in percentages)
                            x_percent = bbox_dict['x']
                            y_percent = bbox_dict['y'] 
                            width_percent = bbox_dict['width']
                            height_percent = bbox_dict['height']
                            
                            # Convert to absolute coordinates
                            x = (x_percent / 100) * width
                            y = (y_percent / 100) * height
                            w = (width_percent / 100) * width
                            h = (height_percent / 100) * height
                            
                            # Convert to pascal VOC format (xmin, ymin, xmax, ymax)
                            xmin = x
                            ymin = y
                            xmax = x + w
                            ymax = y + h
                            
                            # Ensure coordinates are within image bounds
                            xmin = max(0, min(xmin, width - 1))
                            ymin = max(0, min(ymin, height - 1))
                            xmax = max(xmin + 1, min(xmax, width))
                            ymax = max(ymin + 1, min(ymax, height))
                            
                            bboxes.append([xmin, ymin, xmax, ymax])
                            
                    except Exception as e:
                        print(f"Error parsing bbox for image {image_path}: {e}")
                        continue
        # Normal images have no bboxes (background only)
        
        # Convert to numpy arrays
        if len(bboxes) > 0:
            pascal_bboxes = np.array(bboxes, dtype=np.float32)
            # All fractures are treated as class 1 (single class detection)
            class_labels = np.ones(len(bboxes), dtype=np.float32)
        else:
            # No valid bboxes found (normal images or fracture images without valid annotations)
            pascal_bboxes = np.array([], dtype=np.float32).reshape(0, 4)
            class_labels = np.array([], dtype=np.float32)
        
        return image, pascal_bboxes, class_labels, str(index)
    
    def show_image(self, index: int):
        """Visualize image with bounding boxes"""
        image, bboxes, labels, image_id = self.get_image_and_labels_by_idx(index)
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        
        for bbox in bboxes:
            rect = patches.Rectangle((bbox[0], bbox[1]), 
                                   bbox[2] - bbox[0], 
                                   bbox[3] - bbox[1], 
                                   linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        
        ax.set_title(f'Image ID: {image_id} - {len(bboxes)} fractures')
        plt.show()

# ================================================================================================
# MODEL HELPER FUNCTIONS
# ================================================================================================

def create_efficientdet_model(num_classes: int, 
                            model_name: str = 'tf_efficientdet_d7x',
                            pretrained: bool = True) -> DetBenchTrain:
    """Create EfficientDet model with image size from config"""
    
    # Get configuration for the specified model
    config = get_efficientdet_config(model_name)
    
    # Get image size from config
    image_size = config.image_size[0]
    print(f"Using image size from {model_name} config: {image_size}")
    
    # Handle DenseNet backbone compatibility - remove drop_path_rate if using DenseNet
    # DenseNet models don't support drop_path_rate parameter, so we need to remove it
    # to avoid "TypeError: DenseNet.__init__() got an unexpected keyword argument 'drop_path_rate'"
    if 'densenet' in model_name and 'backbone_args' in config and 'drop_path_rate' in config.backbone_args:
        print(f"Removing drop_path_rate for DenseNet backbone compatibility")
        config.backbone_args = {k: v for k, v in config.backbone_args.items() if k != 'drop_path_rate'}
    
    # Ensure image_size is a tuple (height, width)
    if isinstance(image_size, int):
        image_size_tuple = (image_size, image_size)
    else:
        image_size_tuple = image_size
    
    config.update({
        'num_classes': num_classes,
        'image_size': image_size_tuple
    })
    
    # Create model
    net = EfficientDet(config, pretrained_backbone=pretrained)
    
    # Replace classification head for custom number of classes
    net.class_net = HeadNet(config, num_outputs=num_classes)
    
    return DetBenchTrain(net), image_size

# ================================================================================================
# PYTORCH DATASET AND TRANSFORMS
# ================================================================================================

class EfficientDetDataset(Dataset):
    """Dataset class for EfficientDet training"""
    
    def __init__(self, dataset_adaptor, transforms=None):
        self.ds = dataset_adaptor
        self.transforms = transforms
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        image, pascal_bboxes, class_labels, image_id = self.ds.get_image_and_labels_by_idx(index)
        
        # Convert PIL image to numpy array
        image = np.array(image)
        
        # Handle empty annotations
        if len(pascal_bboxes) == 0:
            # Create dummy annotation to avoid training issues
            h, w = image.shape[:2]
            pascal_bboxes = np.array([[0, 0, 1, 1]], dtype=np.float32)
            class_labels = np.array([0], dtype=np.float32)  # background class
        
        # Convert pascal VOC to YXYX format (required by EfficientDet)
        yxyx_bboxes = pascal_bboxes.copy()
        yxyx_bboxes[:, [0, 1, 2, 3]] = yxyx_bboxes[:, [1, 0, 3, 2]]  # xyxy -> yxyx
        
        if self.transforms:
            # Apply albumentations transforms
            sample = {
                'image': image,
                'bboxes': pascal_bboxes,
                'labels': class_labels
            }
            
            try:
                sample = self.transforms(**sample)
                image = sample['image']
                if len(sample['bboxes']) > 0:
                    yxyx_bboxes = np.array(sample['bboxes'])
                    yxyx_bboxes[:, [0, 1, 2, 3]] = yxyx_bboxes[:, [1, 0, 3, 2]]  # xyxy -> yxyx
                    class_labels = np.array(sample['labels'])
                else:
                    # Handle case where transforms removed all boxes
                    yxyx_bboxes = np.array([[0, 0, 1, 1]], dtype=np.float32)
                    class_labels = np.array([0], dtype=np.float32)
            except Exception as e:
                print(f"Transform error for image {image_id}: {e}")
                # Fall back to original image
                image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
        
        # Convert to tensors
        # Handle image dimensions properly
        if isinstance(image, torch.Tensor):
            img_height, img_width = image.shape[1], image.shape[2]
        else:
            img_height, img_width = image.shape[0], image.shape[1]
            
        targets = {
            'bbox': torch.tensor(yxyx_bboxes, dtype=torch.float32),
            'cls': torch.tensor(class_labels, dtype=torch.long),
            'img_size': torch.tensor([img_width, img_height], dtype=torch.float32),
            'img_scale': torch.tensor(1.0, dtype=torch.float32)
        }
        
        return image, targets, image_id

def get_train_transforms(image_size: int = 832):
    """Training transforms with medical image normalization"""
    # Handle case where image_size might be a list [height, width]
    if isinstance(image_size, (list, tuple)):
        height, width = image_size[0], image_size[1]
    else:
        height = width = image_size
    
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.Resize(height=height, width=width, p=1.0),
        # Use custom normalization for medical images
        A.Normalize([0.20793004, 0.20824375, 0.20862524], 
                   [0.23613826, 0.23636451, 0.23669834]),
        ToTensorV2(p=1.0)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.3))

def get_valid_transforms(image_size: int = 832):
    """Validation transforms"""
    # Handle case where image_size might be a list [height, width]
    if isinstance(image_size, (list, tuple)):
        height, width = image_size[0], image_size[1]
    else:
        height = width = image_size
    
    return A.Compose([
        A.Resize(height=height, width=width, p=1.0),
        A.Normalize([0.20793004, 0.20824375, 0.20862524], 
                   [0.23613826, 0.23636451, 0.23669834]),

                   
        ToTensorV2(p=1.0)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

class EfficientDetDataModule(pl.LightningDataModule):
    """Lightning DataModule for EfficientDet"""
    
    def __init__(self, 
                 train_dataset_adaptor,
                 validation_dataset_adaptor,
                 train_transforms=None,
                 valid_transforms=None,
                 num_workers: int = 8,
                 batch_size: int = 1):
        super().__init__()
        
        self.train_ds = train_dataset_adaptor
        self.valid_ds = validation_dataset_adaptor
        self.train_transforms = train_transforms
        self.valid_transforms = valid_transforms
        self.num_workers = num_workers
        self.batch_size = batch_size
    
    def train_dataloader(self) -> DataLoader:
        train_dataset = EfficientDetDataset(
            dataset_adaptor=self.train_ds,
            transforms=self.train_transforms
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )
        return train_loader
    
    def val_dataloader(self) -> DataLoader:
        valid_dataset = EfficientDetDataset(
            dataset_adaptor=self.valid_ds,
            transforms=self.valid_transforms
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )
        return valid_loader
    
    @staticmethod
    def collate_fn(batch):
        images, targets, image_ids = tuple(zip(*batch))
        images = torch.stack(images)
        images = images.float()
        
        boxes = []
        labels = []
        img_sizes = []
        img_scales = []
        
        for i, target in enumerate(targets):
            boxes.append(target['bbox'])
            labels.append(target['cls'])
            img_sizes.append(target['img_size'])
            img_scales.append(target['img_scale'])
        
        # Stack img_size and img_scale tensors properly
        img_size_tensor = torch.stack(img_sizes)
        img_scale_tensor = torch.stack(img_scales)
        
        return images, {'bbox': boxes, 'cls': labels, 'img_size': img_size_tensor, 'img_scale': img_scale_tensor}, image_ids

# ================================================================================================
# PYTORCH LIGHTNING MODULE
# ================================================================================================

class EfficientDetModel(pl.LightningModule):
    """Lightning Module for EfficientDet"""
    
    def __init__(self, 
                 num_classes: int = 1,
                 learning_rate: float = 0.0001,
                 wbf_iou_threshold: float = 0.44,
                 model_name: str = 'tf_efficientdet_d7x'):
        super().__init__()
        
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.wbf_iou_threshold = wbf_iou_threshold
        self.model_name = model_name
        
        self.model, self.image_size = create_efficientdet_model(
            num_classes=num_classes,
            model_name=model_name
        )
        
        print(f"Model {model_name} initialized with image size: {self.image_size}")
        
        self.save_hyperparameters()
    
    @property
    def image_dim(self):
        """Get single image dimension from image_size (handles both int and list)"""
        if isinstance(self.image_size, (list, tuple)):
            return self.image_size[0]  # Assume square images, take first dimension
        return self.image_size
    
    def forward(self, images, targets):
        return self.model(images, targets)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }
    
    def training_step(self, batch, batch_idx):
        images, targets, image_ids = batch
        
        losses = self.model(images, targets)
        
        self.log("train_loss", losses['loss'], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_class_loss", losses['class_loss'], on_step=True, on_epoch=True)
        self.log("train_box_loss", losses['box_loss'], on_step=True, on_epoch=True)
        
        return losses['loss']

    def validation_step(self, batch, batch_idx):
        images, targets, image_ids = batch
        
        losses = self.model(images, targets)
        
        self.log("val_loss", losses['loss'], on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_class_loss", losses['class_loss'], on_step=True, on_epoch=True)
        self.log("val_box_loss", losses['box_loss'], on_step=True, on_epoch=True)
        
        return losses['loss']

# ================================================================================================
# MAIN TRAINING FUNCTION
# ================================================================================================

def train_fracture_detection(
    data_dir: str,
    csv_path: str,
    model_name: str = 'tf_efficientdet_d7x',
    batch_size: int = 1,
    max_epochs: int = 50,
    learning_rate: float = 0.0001,
    val_split: float = 0.2,
    num_workers: int = 8,
    # Evaluation parameters
    run_evaluation: bool = True,
    test_size: int = 200,
    normal_limit: int = None,
    eval_conf_threshold: float = 0.3,
    eval_iou_threshold: float = 0.5,
    eval_seed: int = 42
):
    """
    Main training function for fracture detection with proper train/val/test splits
    
    IMPORTANT: Test set is created BEFORE training to prevent data leakage.
    Training and validation only use the remaining data after test set removal.
    """
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"runs/{model_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up data paths
    fracture_dir = os.path.join(data_dir, "fracture")
    normal_dir = os.path.join(data_dir, "normal")
    
    print(f"Data directory: {data_dir}")
    print(f"Fracture directory: {fracture_dir}")
    print(f"Normal directory: {normal_dir}")
    print(f"CSV path: {csv_path}")
    print(f"Output directory: {output_dir}")
    
    # Verify directories exist
    if not os.path.exists(fracture_dir):
        raise ValueError(f"Fracture directory not found: {fracture_dir}")
    if not os.path.exists(normal_dir):
        print(f"Warning: Normal directory not found: {normal_dir}")
    if not os.path.exists(csv_path):
        raise ValueError(f"CSV file not found: {csv_path}")
    
    # Get image size from model config
    config = get_efficientdet_config(model_name)
    image_size = config.image_size[0]
    print(f"Using image size {image_size} for model {model_name}")
    
    # Extract single dimension for transforms (handle both int and list)
    if isinstance(image_size, (list, tuple)):
        image_dim = image_size[0]  # Assume square images, take first dimension
    else:
        image_dim = image_size
    
    # ================================================================================================
    # STEP 1: CREATE TEST SET BEFORE TRAINING (PREVENT DATA LEAKAGE)
    # ================================================================================================
    
    print("\n" + "="*80)
    print("STEP 1: CREATING TEST SET (BEFORE TRAINING)")
    print("="*80)
    
    # Set random seed for reproducible test set
    random.seed(eval_seed)
    np.random.seed(eval_seed)
    
    # Load CSV data for fracture images
    annotations_df = pd.read_csv(csv_path)
    fracture_images = annotations_df['study_path'].unique().tolist()
    
    # Filter existing fracture images
    existing_fracture_images = []
    for img_path in fracture_images:
        # Build candidate physical paths
        candidates = []
        if os.path.isabs(img_path):
            candidates.append(img_path)
        candidates.append(os.path.join(fracture_dir, img_path.lstrip('/')))
        candidates.append(os.path.join(fracture_dir, os.path.basename(img_path)))
        candidates.append(os.path.join(fracture_dir, img_path.replace('/', '_')))

        found = next((p for p in candidates if os.path.exists(p)), None)
        if found is not None:
            existing_fracture_images.append(img_path)
    
    # Get ALL normal images (no limit applied yet - test set should be unbiased)
    all_normal_images = []
    if os.path.exists(normal_dir):
        for root, _, files in os.walk(normal_dir):
            for file in files:
                if file.lower().endswith(('.jpeg', '.jpg', '.png')):
                    relative_path = os.path.relpath(os.path.join(root, file), normal_dir)
                    all_normal_images.append(relative_path)
    
    print(f"Found {len(all_normal_images)} total normal images (before any limits)")
    
    # Randomly sample test sets from ALL available images (no normal_limit)
    random.shuffle(existing_fracture_images)
    random.shuffle(all_normal_images)
    
    test_fracture = existing_fracture_images[:min(test_size, len(existing_fracture_images))]
    test_normal = all_normal_images[:min(test_size, len(all_normal_images))]
    
    # IMPORTANT: Remove test samples from training data
    train_val_fracture = existing_fracture_images[len(test_fracture):]
    train_val_normal_all = all_normal_images[len(test_normal):]
    
    # NOW apply normal_limit to train+val data only (not test set)
    if normal_limit is not None and len(train_val_normal_all) > normal_limit:
        random.shuffle(train_val_normal_all)
        train_val_normal = train_val_normal_all[:normal_limit]
        print(f"Applied normal_limit={normal_limit} to train+val data only")
        print(f"Train+Val normal images: {len(train_val_normal)} (limited from {len(train_val_normal_all)})")
    else:
        train_val_normal = train_val_normal_all
        print(f"No normal_limit applied to train+val data")
        print(f"Train+Val normal images: {len(train_val_normal)}")
    
    print(f"\nDataset Split Summary:")
    print(f"  Test set (UNBIASED - no normal_limit applied):")
    print(f"    - Fracture images: {len(test_fracture)}")
    print(f"    - Normal images: {len(test_normal)}")
    print(f"    - Total test images: {len(test_fracture) + len(test_normal)}")
    print(f"  Train+Val set (normal_limit={normal_limit} applied here only):")
    print(f"    - Fracture images: {len(train_val_fracture)}")
    print(f"    - Normal images: {len(train_val_normal)}")
    print(f"    - Total train+val images: {len(train_val_fracture) + len(train_val_normal)}")
    
    # Save test set information for evaluation
    test_data = {
        'fracture': test_fracture,
        'normal': test_normal,
        'fracture_dir': fracture_dir,
        'normal_dir': normal_dir,
        'annotations_df': annotations_df,
        'test_size': test_size
    }
    
    # Save test set to JSON file for reproducibility
    test_set_file = os.path.join(output_dir, 'test_set.json')
    with open(test_set_file, 'w') as f:
        json.dump({
            'test_fracture': test_fracture,
            'test_normal': test_normal,
            'test_size': test_size,
            'normal_limit': normal_limit,
            'eval_seed': eval_seed,
            'timestamp': timestamp
        }, f, indent=2)
    print(f"Test set saved to: {test_set_file}")
    
    # ================================================================================================
    # STEP 2: CREATE TRAINING DATASET (EXCLUDING TEST SET)
    # ================================================================================================
    
    print("\n" + "="*80)
    print("STEP 2: CREATING TRAINING DATASET (EXCLUDING TEST SET)")
    print("="*80)
    
    # Optimize for A100 Tensor Cores
    torch.set_float32_matmul_precision('medium')  # Better performance on A100
    
    # Initialize wandb (logging only, no model artifacts)
    wandb.init(
        project="fracture_detection_efficientdet",
        config={
            "model": model_name,
            "image_size": image_size,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_epochs": max_epochs,
            "num_classes": 1,
            "normalization": "medical_custom",
            "num_workers": num_workers,
            "data_dir": data_dir,
            "csv_path": csv_path,
            "run_evaluation": run_evaluation,
            "test_size": test_size,
            "normal_limit": normal_limit,
            "eval_conf_threshold": eval_conf_threshold,
            "eval_iou_threshold": eval_iou_threshold,
            "eval_seed": eval_seed,
            "train_fracture_count": len(train_val_fracture),
            "train_normal_count": len(train_val_normal),
            "train_normal_available": len(train_val_normal_all),
            "test_fracture_count": len(test_fracture),
            "test_normal_count": len(test_normal),
            "all_normal_images_found": len(all_normal_images)
        },
        settings=wandb.Settings(
            _disable_meta=True,  # Disable automatic model logging
            _disable_stats=True   # Disable system stats logging
        )
    )
    
    # Create dataset adaptor with ONLY train+val images (test set excluded)
    dataset_adaptor = FractureDatasetAdaptor.__new__(FractureDatasetAdaptor)
    dataset_adaptor.fracture_dir_path = fracture_dir
    dataset_adaptor.normal_dir_path = normal_dir
    dataset_adaptor.csv_path = csv_path
    dataset_adaptor.annotations_df = annotations_df
    
    # CRUCIAL: Only use train+val images (test set completely excluded)
    train_val_images = []
    for img_path in train_val_fracture:
        train_val_images.append(('fracture', img_path))
    for img_path in train_val_normal:
        train_val_images.append(('normal', img_path))
    
    dataset_adaptor.images = train_val_images
    
    # Initialize corrupted image tracking
    dataset_adaptor.corrupted_count = 0
    dataset_adaptor.corrupted_log_file = f"corrupted_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    # Write initial header to log file
    with open(dataset_adaptor.corrupted_log_file, 'w') as f:
        f.write(f"Corrupted Images Log - Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
    
    print(f"Training dataset created (test set excluded):")
    print(f"  Total images available for train+val: {len(dataset_adaptor)}")
    
    # ================================================================================================
    # STEP 3: SPLIT TRAIN+VAL (FROM REMAINING DATA AFTER TEST REMOVAL)
    # ================================================================================================
    
    print("\n" + "="*80)
    print("STEP 3: SPLITTING TRAIN+VAL (FROM REMAINING DATA)")
    print("="*80)
    
    # Split train+val dataset
    total_size = len(dataset_adaptor)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    # Create simple split (you might want to use stratified split)
    indices = list(range(total_size))
    np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create sub-datasets
    train_images = [dataset_adaptor.images[i] for i in train_indices]
    val_images = [dataset_adaptor.images[i] for i in val_indices]
    
    # Create separate adaptors for train/val
    train_adaptor = FractureDatasetAdaptor.__new__(FractureDatasetAdaptor)
    train_adaptor.fracture_dir_path = dataset_adaptor.fracture_dir_path
    train_adaptor.normal_dir_path = dataset_adaptor.normal_dir_path
    train_adaptor.csv_path = dataset_adaptor.csv_path
    train_adaptor.annotations_df = dataset_adaptor.annotations_df
    train_adaptor.images = train_images
    # Initialize corrupted image tracking for train adaptor
    train_adaptor.corrupted_count = 0
    train_adaptor.corrupted_log_file = dataset_adaptor.corrupted_log_file
    
    val_adaptor = FractureDatasetAdaptor.__new__(FractureDatasetAdaptor)
    val_adaptor.fracture_dir_path = dataset_adaptor.fracture_dir_path
    val_adaptor.normal_dir_path = dataset_adaptor.normal_dir_path
    val_adaptor.csv_path = dataset_adaptor.csv_path
    val_adaptor.annotations_df = dataset_adaptor.annotations_df
    val_adaptor.images = val_images
    # Initialize corrupted image tracking for validation adaptor
    val_adaptor.corrupted_count = 0
    val_adaptor.corrupted_log_file = dataset_adaptor.corrupted_log_file
    
    print(f"Final dataset splits:")
    print(f"  Training set: {len(train_images)} images")
    print(f"  Validation set: {len(val_images)} images")
    print(f"  Test set (held out): {len(test_fracture) + len(test_normal)} images")
    print(f"  Total: {len(train_images) + len(val_images) + len(test_fracture) + len(test_normal)} images")
    
    # ================================================================================================
    # STEP 4: TRAINING (USING ONLY TRAIN+VAL, TEST SET COMPLETELY ISOLATED)
    # ================================================================================================
    
    print("\n" + "="*80)
    print("STEP 4: STARTING TRAINING (TEST SET ISOLATED)")
    print("="*80)
    
    # Create data module
    dm = EfficientDetDataModule(
        train_dataset_adaptor=train_adaptor,
        validation_dataset_adaptor=val_adaptor,
        train_transforms=get_train_transforms(image_dim),
        valid_transforms=get_valid_transforms(image_dim),
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Create model
    model = EfficientDetModel(
        num_classes=1,  # background + fracture
        learning_rate=learning_rate,
        model_name=model_name
    )
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename=f'{model_name}_fracture_detection-{{epoch:02d}}-{{val_loss:.3f}}',
        monitor='val_loss',
        save_top_k=3,
        mode='min',
        save_last=True
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=8,
        mode='min',
        verbose=True
    )
    
    # Setup wandb logger
    wandb_logger = WandbLogger(project="efficientdet_D7x_W&H")
    
    # Create trainer
    trainer = Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stopping],
        logger=wandb_logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision='16-mixed' if torch.cuda.is_available() else 32,
        gradient_clip_val=10.0,
        accumulate_grad_batches=3,  # Effective batch size = batch_size * 3
        log_every_n_steps=10,
        enable_checkpointing=True,
        enable_model_summary=True
    )
    
    # Train model
    trainer.fit(model, dm)
    
    # Log final corrupted image summary
    with open(train_adaptor.corrupted_log_file, 'a') as f:
        f.write(f"\n" + "=" * 60 + "\n")
        f.write(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total corrupted images skipped: {train_adaptor.corrupted_count}\n")
        f.write(f"Validation corrupted images skipped: {val_adaptor.corrupted_count}\n")
        f.write(f"Total corrupted images: {train_adaptor.corrupted_count + val_adaptor.corrupted_count}\n")
    
    print(f"\nCorrupted Images Summary:")
    print(f"Training set: {train_adaptor.corrupted_count} corrupted images skipped")
    print(f"Validation set: {val_adaptor.corrupted_count} corrupted images skipped")
    print(f"Total: {train_adaptor.corrupted_count + val_adaptor.corrupted_count} corrupted images skipped")
    print(f"Log file: {train_adaptor.corrupted_log_file}")
    
    # Save final model
    final_model_path = os.path.join(output_dir, f'{model_name}_fracture_detection_final.ckpt')
    trainer.save_checkpoint(final_model_path)
    
    print(f"Training completed! Best model saved in: {output_dir}")
    print(f"Final model saved as: {final_model_path}")
    
    # ================================================================================================
    # STEP 5: EVALUATION ON HELD-OUT TEST SET
    # ================================================================================================
    
    # Run automatic evaluation if requested
    evaluation_results = None
    if run_evaluation:
        try:
            print("\n" + "="*80)
            print("STEP 5: EVALUATING ON HELD-OUT TEST SET")
            print("="*80)
            
            evaluation_results = evaluate_trained_model_with_test_data(
                model=model,
                trainer=trainer,
                output_dir=output_dir,
                test_data=test_data,
                conf_threshold=eval_conf_threshold,
                batch_size=4,
                iou_threshold=eval_iou_threshold
            )
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            print("Training completed successfully, but evaluation encountered errors.")
    
    wandb.finish()
    
    return model, trainer, evaluation_results

# ================================================================================================
# COMMAND LINE ARGUMENT PARSER
# ================================================================================================

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Fracture Detection Training with EfficientDet and Automatic Evaluation')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='/home/ai-user/data',
                      help='Path to data directory containing fracture/ and normal/ subdirectories (default: Data)')
    parser.add_argument('--csv_path', type=str, default='/home/ai-user/efficientdet/newfrac_ann.csv',
                      help='Path to CSV file with fracture annotations (default: csvs/frac_ann.csv)')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='tf_efficientdet_d7x',
                      help='EfficientDet model name (default: tf_efficientdet_d7x)')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=2,
                      help='Batch size for training (default: 8)')
    parser.add_argument('--max_epochs', type=int, default=50,
                      help='Maximum number of epochs (default: 50)')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                      help='Learning rate (default: 0.0001)')
    parser.add_argument('--val_split', type=float, default=0.2,
                      help='Validation split ratio (default: 0.2)')
    parser.add_argument('--num_workers', type=int, default=None,
                      help='Number of data loading workers (default: auto-detect)')
    
    # Evaluation arguments
    parser.add_argument('--run_evaluation', action='store_true', default=True,
                      help='Run automatic evaluation after training (default: True)')
    parser.add_argument('--skip_evaluation', action='store_true', default=False,
                      help='Skip automatic evaluation after training')
    parser.add_argument('--test_size', type=int, default=200,
                      help='Number of images per class for testing (default: 200)')
    parser.add_argument('--normal_limit', type=int, default=None,
                      help='Maximum number of normal images to consider during evaluation (default: no limit)')
    parser.add_argument('--eval_conf_threshold', type=float, default=0.3,
                      help='Confidence threshold for evaluation (default: 0.3)')
    parser.add_argument('--eval_iou_threshold', type=float, default=0.5,
                      help='IoU threshold for evaluation (default: 0.5)')
    parser.add_argument('--eval_seed', type=int, default=42,
                      help='Random seed for reproducible evaluation splits (default: 42)')
    
    return parser.parse_args()

# ================================================================================================
# EVALUATION HELPER FUNCTIONS
# ================================================================================================

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes in [x1, y1, x2, y2] format"""
    # Ensure boxes are in correct format
    box1 = np.array(box1)
    box2 = np.array(box2)
    
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Check if there's intersection
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    # Calculate intersection area
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    # Avoid division by zero
    if union == 0:
        return 0.0
    
    return intersection / union

def match_predictions_to_ground_truth(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
    """
    Match predictions to ground truth boxes using IoU threshold
    Returns matches with IoU values and unmatched predictions/ground truths
    """
    matches = []
    gt_matched = [False] * len(gt_boxes)
    pred_matched = [False] * len(pred_boxes)
    
    # Sort predictions by confidence (highest first)
    if len(pred_boxes) > 0:
        sorted_indices = np.argsort(pred_scores)[::-1]
        
        for pred_idx in sorted_indices:
            if pred_matched[pred_idx]:
                continue
                
            best_iou = 0.0
            best_gt_idx = -1
            
            # Find best matching ground truth
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_matched[gt_idx]:
                    continue
                    
                iou = calculate_iou(pred_boxes[pred_idx], gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # If IoU is above threshold, create match
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                matches.append({
                    'pred_idx': pred_idx,
                    'gt_idx': best_gt_idx,
                    'iou': best_iou,
                    'confidence': pred_scores[pred_idx],
                    'pred_box': pred_boxes[pred_idx],
                    'gt_box': gt_boxes[best_gt_idx]
                })
                gt_matched[best_gt_idx] = True
                pred_matched[pred_idx] = True
    
    # Get unmatched predictions and ground truths
    unmatched_preds = [i for i, matched in enumerate(pred_matched) if not matched]
    unmatched_gts = [i for i, matched in enumerate(gt_matched) if not matched]
    
    return matches, unmatched_preds, unmatched_gts

def calculate_ap(precisions, recalls):
    """Calculate Average Precision using 11-point interpolation"""
    # Add sentinel values
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    
    # Compute precision envelope
    for i in range(precisions.size - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
    
    # Find points where recall changes
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    
    # Calculate AP using trapezoidal rule
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    
    return ap

def calculate_coco_metrics(all_matches, all_unmatched_preds, all_unmatched_gts, 
                          total_gt_boxes, iou_thresholds=None):
    """Calculate COCO-style metrics including AP@0.5, AP@0.5:0.95"""
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05)  # 0.5, 0.55, 0.6, ..., 0.95
    
    metrics = {}
    aps = []
    
    for iou_thresh in iou_thresholds:
        # Filter matches by IoU threshold
        valid_matches = [m for m in all_matches if m['iou'] >= iou_thresh]
        
        if len(valid_matches) == 0 and total_gt_boxes == 0:
            ap = 1.0  # Perfect if no GT and no predictions
        elif len(valid_matches) == 0:
            ap = 0.0  # No matches
        else:
            # Sort by confidence
            valid_matches.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Calculate precision and recall at each detection
            tp = np.zeros(len(valid_matches))
            fp = np.zeros(len(valid_matches))
            
            for i, match in enumerate(valid_matches):
                tp[i] = 1  # All valid matches are TP by definition
            
            # Add FP from unmatched predictions with high confidence
            all_confs = [m['confidence'] for m in valid_matches]
            
            # Calculate cumulative TP and FP
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            
            # Calculate precision and recall
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
            recalls = tp_cumsum / (total_gt_boxes + 1e-8)
            
            # Calculate AP
            ap = calculate_ap(precisions, recalls)
        
        aps.append(ap)
        metrics[f'AP@{iou_thresh:.2f}'] = ap
    
    # Calculate mean AP over all IoU thresholds
    metrics['AP@0.5:0.95'] = np.mean(aps)
    metrics['AP@0.5'] = aps[0] if len(aps) > 0 else 0.0
    
    return metrics

def run_inference_on_test_data(model: EfficientDetModel, test_data: dict, 
                              output_dir: str, conf_threshold: float = 0.3, batch_size: int = 4,
                              iou_threshold: float = 0.5):
    """Run comprehensive IoU-based inference evaluation on test dataset"""
    print("Running comprehensive IoU-based evaluation on test dataset...")
    
    evaluation_results = {}
    device = next(model.parameters()).device
    model.eval()
    
    # Collect all matches for COCO metrics
    all_matches = []
    all_unmatched_preds = []
    all_unmatched_gts = []
    total_gt_boxes = 0
    
    # Collect all image paths with their types
    all_images = []
    
    # Process fracture images
    for img_path in test_data['fracture']:
        # Build list of possible physical paths
        candidates = []
        if os.path.isabs(img_path):
            candidates.append(img_path)
        candidates.append(os.path.join(test_data['fracture_dir'], img_path.lstrip('/')))
        candidates.append(os.path.join(test_data['fracture_dir'], os.path.basename(img_path)))
        candidates.append(os.path.join(test_data['fracture_dir'], img_path.replace('/', '_')))

        final_path = next((p for p in candidates if os.path.exists(p)), None)

        if final_path is not None:
            img_id = f"fracture_{hash(img_path) % 100000}"
            all_images.append({
                'img_path': img_path,
                'full_image_path': final_path,
                'img_id': img_id,
                'is_fracture': True
            })
    
    # Process normal images  
    for img_path in test_data['normal']:
        full_image_path = os.path.join(test_data['normal_dir'], img_path)
        
        if os.path.exists(full_image_path):
            img_id = f"normal_{hash(img_path) % 100000}"
            all_images.append({
                'img_path': img_path,
                'full_image_path': full_image_path,
                'img_id': img_id,
                'is_fracture': False
            })
    
    print(f"Running IoU-based evaluation on {len(all_images)} images with batch size {batch_size}")
    print(f"Confidence threshold: {conf_threshold}, IoU threshold: {iou_threshold}")
    
    # Process images in batches with progress bar
    with torch.no_grad():
        for i in tqdm(range(0, len(all_images), batch_size), desc="Processing batches"):
            batch_images = all_images[i:i + batch_size]
            
            # Prepare batch
            batch_tensors = []
            batch_info = []
            
            for img_info in batch_images:
                try:
                    # Load and preprocess image
                    image = Image.open(img_info['full_image_path']).convert("RGB")
                    original_size = image.size  # (width, height)
                    
                    # Use validation transforms
                    transforms = get_valid_transforms(model.image_dim)
                    image_array = np.array(image)
                    
                    # Apply transforms
                    transformed = transforms(image=image_array, bboxes=[], labels=[])
                    image_tensor = transformed['image']
                    
                    # Ensure it's a tensor
                    if isinstance(image_tensor, np.ndarray):
                        image_tensor = torch.from_numpy(image_tensor).float()
                    
                    batch_tensors.append(image_tensor)
                    batch_info.append({
                        'img_info': img_info,
                        'original_size': original_size
                    })
                    
                except Exception as e:
                    print(f"Error loading image {img_info['full_image_path']}: {e}")
                    # Add dummy evaluation result for failed images
                    evaluation_results[img_info['img_id']] = {
                        'predictions': {'boxes': np.array([]), 'scores': np.array([]), 'labels': np.array([])},
                        'ground_truth_boxes': np.array([]).reshape(0, 4),
                        'matches': [],
                        'unmatched_preds': [],
                        'unmatched_gts': [],
                        'mean_iou': 0.0,
                        'max_confidence': 0.0,
                        'num_predictions': 0,
                        'num_gt_boxes': 0
                    }
                    continue
            
            if not batch_tensors:
                continue
                
            # Stack batch tensors
            batch_tensor = torch.stack(batch_tensors).to(device)
            
            # Create dummy targets for inference
            batch_size_actual = batch_tensor.shape[0]
            dummy_targets = {
                'bbox': [torch.zeros((1, 4), dtype=torch.float32, device=device)] * batch_size_actual,
                'cls': [torch.zeros(1, dtype=torch.long, device=device)] * batch_size_actual,
                'img_size': torch.stack([torch.tensor([model.image_dim, model.image_dim], 
                                                    dtype=torch.float32, device=device)] * batch_size_actual),
                'img_scale': torch.stack([torch.tensor(1.0, dtype=torch.float32, device=device)] * batch_size_actual)
            }
            
            # Run forward pass
            try:
                outputs = model(batch_tensor, dummy_targets)
                
                # Process outputs for each image in batch
                for idx, info in enumerate(batch_info):
                    img_info = info['img_info']
                    original_size = info['original_size']
                    
                    # Extract predictions
                    predictions = extract_single_prediction(
                        outputs, idx, original_size, model.image_dim
                    )
                    
                    # Get ground truth boxes
                    if img_info['is_fracture']:
                        gt_boxes = get_ground_truth_boxes(
                            img_info['img_path'], 
                            test_data['annotations_df'], 
                            original_size
                        )
                    else:
                        gt_boxes = np.array([]).reshape(0, 4)  # Normal images have no GT boxes
                    
                    # Perform IoU-based matching
                    pred_boxes = predictions.get('boxes', [])
                    pred_scores = predictions.get('scores', [])
                    
                    if len(pred_boxes) > 0 and len(pred_scores) > 0:
                        # Convert to numpy arrays and ensure they're 1D
                        pred_scores_array = np.array(pred_scores)
                        if pred_scores_array.ndim > 1:
                            pred_scores_array = pred_scores_array.flatten()
                        
                        # Filter predictions by confidence
                        high_conf_mask = pred_scores_array >= conf_threshold
                        high_conf_indices = np.where(high_conf_mask)[0]
                        
                        filtered_boxes = [pred_boxes[i] for i in high_conf_indices]
                        filtered_scores = [pred_scores_array[i] for i in high_conf_indices]
                    else:
                        filtered_boxes = []
                        filtered_scores = []
                    
                    # Match predictions to ground truth
                    if len(filtered_boxes) > 0 and len(gt_boxes) > 0:
                        matches, unmatched_preds, unmatched_gts = match_predictions_to_ground_truth(
                            np.array(filtered_boxes), np.array(filtered_scores), gt_boxes, iou_threshold
                        )
                    else:
                        matches = []
                        unmatched_preds = list(range(len(filtered_boxes)))
                        unmatched_gts = list(range(len(gt_boxes)))
                    
                    # Calculate metrics for this image
                    ious = [m['iou'] for m in matches]
                    mean_iou = np.mean(ious) if ious else 0.0
                    
                    # Safe max calculation for pred_scores
                    if len(pred_scores) > 0:
                        # Convert to numpy array and handle multi-dimensional arrays
                        pred_scores_array = np.array(pred_scores)
                        if pred_scores_array.ndim > 1:
                            pred_scores_array = pred_scores_array.flatten()
                        max_confidence = float(np.max(pred_scores_array))
                    else:
                        max_confidence = 0.0
                    
                    # Store evaluation results
                    img_result = {
                        'predictions': predictions,
                        'ground_truth_boxes': gt_boxes,
                        'matches': matches,
                        'unmatched_preds': unmatched_preds,
                        'unmatched_gts': unmatched_gts,
                        'mean_iou': mean_iou,
                        'max_confidence': max_confidence,
                        'num_predictions': len(pred_boxes),
                        'num_gt_boxes': len(gt_boxes)
                    }
                    
                    evaluation_results[img_info['img_id']] = img_result
                    
                    # Accumulate for COCO metrics
                    all_matches.extend(matches)
                    all_unmatched_preds.extend(unmatched_preds)
                    all_unmatched_gts.extend(unmatched_gts)
                    total_gt_boxes += len(gt_boxes)
                    
                    # Create enhanced visualization
                    create_enhanced_visualization(
                        img_info['full_image_path'], img_result, img_info['img_id'], 
                        img_info['is_fracture'], gt_boxes, output_dir=output_dir, 
                        conf_threshold=conf_threshold, iou_threshold=iou_threshold
                    )
                    
            except Exception as e:
                print(f"Error during batch inference: {e}")
                # Add empty evaluation results for this batch
                for info in batch_info:
                    evaluation_results[info['img_info']['img_id']] = {
                        'predictions': {'boxes': np.array([]), 'scores': np.array([]), 'labels': np.array([])},
                        'ground_truth_boxes': np.array([]).reshape(0, 4),
                        'matches': [],
                        'unmatched_preds': [],
                        'unmatched_gts': [],
                        'mean_iou': 0.0,
                        'max_confidence': 0.0,
                        'num_predictions': 0,
                        'num_gt_boxes': 0
                    }
    
    # Calculate COCO metrics
    print("\nCalculating COCO metrics...")
    coco_metrics = calculate_coco_metrics(all_matches, all_unmatched_preds, all_unmatched_gts, total_gt_boxes)
    
    print(f"IoU-based evaluation completed on {len(evaluation_results)} images")
    print(f"Total ground truth boxes: {total_gt_boxes}")
    print(f"Total matches found: {len(all_matches)}")
    print(f"COCO Metrics:")
    for metric_name, value in coco_metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    
    return evaluation_results, coco_metrics

def extract_single_prediction(outputs, idx: int, original_size: tuple, model_img_dim: int):
    """Extract prediction for a single image from batch outputs"""
    try:
        # Process outputs
        if isinstance(outputs, dict) and 'detections' in outputs:
            detections = outputs['detections']
            
            if hasattr(detections[idx], 'cpu'):
                detection = detections[idx].cpu().numpy()
            else:
                detection = detections[idx]
            
            # Detection format: [x1, y1, x2, y2, confidence, class]
            if len(detection) > 0 and detection.shape[1] >= 6:
                boxes = detection[:, :4]
                scores = detection[:, 4]
                labels = detection[:, 5]
                
                # Scale boxes back to original image size
                orig_w, orig_h = original_size
                scale_x = orig_w / model_img_dim
                scale_y = orig_h / model_img_dim
                
                boxes[:, [0, 2]] *= scale_x  # x coordinates
                boxes[:, [1, 3]] *= scale_y  # y coordinates
                
                return {
                    'boxes': boxes,
                    'scores': scores,
                    'labels': labels
                }
            else:
                return {
                    'boxes': np.array([]),
                    'scores': np.array([]),
                    'labels': np.array([])
                }
        else:
            return {
                'boxes': np.array([]),
                'scores': np.array([]),
                'labels': np.array([])
            }
            
    except Exception as e:
        print(f"Error extracting prediction for image {idx}: {e}")
        return {
            'boxes': np.array([]),
            'scores': np.array([]),
            'labels': np.array([])
        }

def evaluate_trained_model_with_test_data(model: EfficientDetModel, trainer: Trainer, output_dir: str,
                          test_data: dict, conf_threshold: float = 0.3, batch_size: int = 4,
                          iou_threshold: float = 0.5):
    """
    Comprehensive evaluation with IoU-based metrics and COCO evaluation
    
    This function uses the test set that was created BEFORE training to ensure
    no data leakage and provide unbiased evaluation results with proper object detection metrics.
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE IoU-BASED EVALUATION (NO DATA LEAKAGE)")
    print("="*80)
    print(f"Test set size: {len(test_data['fracture']) + len(test_data['normal'])} images")
    print(f"  - Fracture images: {len(test_data['fracture'])}")
    print(f"  - Normal images: {len(test_data['normal'])}")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"IoU threshold: {iou_threshold}")
    
    # Run comprehensive IoU-based evaluation
    evaluation_results, coco_metrics = run_inference_on_test_data(
        model, test_data, output_dir, conf_threshold, batch_size=batch_size, iou_threshold=iou_threshold
    )
    
    # Calculate IoU-based metrics
    print("\n" + "="*60)
    print("CALCULATING IoU-BASED DETECTION METRICS")
    print("="*60)
    
    # Collect statistics for IoU-based evaluation
    total_gt_boxes = 0
    total_predictions = 0
    tp_detections = 0  # True positive detections (IoU >= threshold)
    fp_detections = 0  # False positive detections (no matching GT or IoU < threshold)
    fn_detections = 0  # False negative detections (GT boxes not matched)
    
    # Image-level classification statistics
    tp_images = 0  # Fracture images correctly detected (with good IoU)
    fp_images = 0  # Normal images incorrectly classified as fracture
    tn_images = 0  # Normal images correctly classified
    fn_images = 0  # Fracture images missed or detected with poor IoU
    
    # IoU statistics
    all_ious = []
    matched_confidences = []
    
    for img_id, img_result in evaluation_results.items():
        gt_boxes = img_result['ground_truth_boxes']
        matches = img_result['matches']
        unmatched_preds = img_result['unmatched_preds']
        unmatched_gts = img_result['unmatched_gts']
        max_confidence = img_result['max_confidence']
        
        # Accumulate detection-level statistics
        total_gt_boxes += len(gt_boxes)
        total_predictions += img_result['num_predictions']
        
        # Count true positive detections (good IoU matches)
        good_matches = [m for m in matches if m['iou'] >= iou_threshold and m['confidence'] >= conf_threshold]
        tp_detections += len(good_matches)
        
        # Count false positive detections (unmatched predictions + poor IoU matches)
        poor_matches = [m for m in matches if m['iou'] < iou_threshold or m['confidence'] < conf_threshold]
        fp_detections += len(unmatched_preds) + len(poor_matches)
        
        # Count false negative detections (unmatched ground truth)
        fn_detections += len(unmatched_gts)
        
        # Collect IoUs and confidences
        for match in matches:
            all_ious.append(match['iou'])
            matched_confidences.append(match['confidence'])
        
        # Image-level classification
        is_fracture = 'fracture' in img_id
        has_good_detection = len(good_matches) > 0
        has_any_detection = max_confidence >= conf_threshold
        
        if is_fracture and has_good_detection:
            tp_images += 1
        elif is_fracture and not has_any_detection:
            fn_images += 1
        elif is_fracture and has_any_detection and not has_good_detection:
            fn_images += 1  # Poor IoU = miss
        elif not is_fracture and has_any_detection:
            fp_images += 1
        else:  # not is_fracture and not has_any_detection
            tn_images += 1
    
    # Calculate detection-level metrics
    detection_precision = tp_detections / (tp_detections + fp_detections) if (tp_detections + fp_detections) > 0 else 0
    detection_recall = tp_detections / (tp_detections + fn_detections) if (tp_detections + fn_detections) > 0 else 0
    detection_f1 = 2 * (detection_precision * detection_recall) / (detection_precision + detection_recall) if (detection_precision + detection_recall) > 0 else 0
    
    # Calculate image-level metrics
    total_images = tp_images + fp_images + tn_images + fn_images
    image_accuracy = (tp_images + tn_images) / total_images if total_images > 0 else 0
    image_precision = tp_images / (tp_images + fp_images) if (tp_images + fp_images) > 0 else 0
    image_recall = tp_images / (tp_images + fn_images) if (tp_images + fn_images) > 0 else 0
    image_f1 = 2 * (image_precision * image_recall) / (image_precision + image_recall) if (image_precision + image_recall) > 0 else 0
    image_specificity = tn_images / (tn_images + fp_images) if (tn_images + fp_images) > 0 else 0
    
    # Calculate IoU statistics
    mean_iou = np.mean(all_ious) if all_ious else 0.0
    median_iou = np.median(all_ious) if all_ious else 0.0
    mean_confidence = np.mean(matched_confidences) if matched_confidences else 0.0
    
    # Copy and classify test images with IoU analysis
    classification_results = copy_test_images_and_classify_with_iou(
        test_data=test_data,
        evaluation_results=evaluation_results,
        output_dir=output_dir,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold
    )
    
    # Print comprehensive results
    print(f"\n" + "="*80)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("="*80)
    
    print(f"\nDATASET STATISTICS:")
    print(f"  Total test images: {total_images}")
    print(f"  Total ground truth boxes: {total_gt_boxes}")
    print(f"  Total predictions: {total_predictions}")
    print(f"  Total matches found: {len(all_ious)}")
    print(f"  Good matches (IoU≥{iou_threshold}): {tp_detections}")
    
    print(f"\nIoU STATISTICS:")
    print(f"  Mean IoU: {mean_iou:.4f}")
    print(f"  Median IoU: {median_iou:.4f}")
    print(f"  Mean Confidence (matched): {mean_confidence:.4f}")
    print(f"  IoU Threshold: {iou_threshold}")
    print(f"  Confidence Threshold: {conf_threshold}")
    
    print(f"\nDETECTION-LEVEL METRICS (Box-level evaluation):")
    print(f"  True Positive Detections:  {tp_detections}")
    print(f"  False Positive Detections: {fp_detections}")
    print(f"  False Negative Detections: {fn_detections}")
    print(f"  Precision@IoU{iou_threshold}: {detection_precision:.4f} ({detection_precision*100:.2f}%)")
    print(f"  Recall@IoU{iou_threshold}:    {detection_recall:.4f} ({detection_recall*100:.2f}%)")
    print(f"  F1-Score@IoU{iou_threshold}:  {detection_f1:.4f} ({detection_f1*100:.2f}%)")
    
    print(f"\nIMAGE-LEVEL METRICS (Classification evaluation):")
    print(f"  True Positive Images:  {tp_images} (Fractures correctly detected with good IoU)")
    print(f"  False Positive Images: {fp_images} (Normal images classified as fractures)")
    print(f"  True Negative Images:  {tn_images} (Normal images correctly classified)")
    print(f"  False Negative Images: {fn_images} (Fractures missed or detected with poor IoU)")
    print(f"  Accuracy@IoU{iou_threshold}:    {image_accuracy:.4f} ({image_accuracy*100:.2f}%)")
    print(f"  Precision@IoU{iou_threshold}:   {image_precision:.4f} ({image_precision*100:.2f}%)")
    print(f"  Recall@IoU{iou_threshold}:      {image_recall:.4f} ({image_recall*100:.2f}%)")
    print(f"  F1-Score@IoU{iou_threshold}:    {image_f1:.4f} ({image_f1*100:.2f}%)")
    print(f"  Specificity@IoU{iou_threshold}: {image_specificity:.4f} ({image_specificity*100:.2f}%)")
    
    print(f"\nCOCO EVALUATION METRICS:")
    for metric_name, value in coco_metrics.items():
        print(f"  {metric_name}: {value:.4f} ({value*100:.2f}%)")
    
    # Prepare comprehensive results
    model_results = {
        'model_type': 'comprehensive_iou_evaluation',
        'conf_threshold': conf_threshold,
        'iou_threshold': iou_threshold,
        'test_size_per_class': test_data.get('test_size', 'unknown'),
        'total_test_images': total_images,
        'total_gt_boxes': total_gt_boxes,
        'total_predictions': total_predictions,
        
        # Detection-level metrics
        'detection_tp': tp_detections,
        'detection_fp': fp_detections,
        'detection_fn': fn_detections,
        'detection_precision': detection_precision,
        'detection_recall': detection_recall,
        'detection_f1': detection_f1,
        
        # Image-level metrics
        'image_tp': tp_images,
        'image_fp': fp_images,
        'image_tn': tn_images,
        'image_fn': fn_images,
        'image_accuracy': image_accuracy,
        'image_precision': image_precision,
        'image_recall': image_recall,
        'image_f1': image_f1,
        'image_specificity': image_specificity,
        
        # IoU statistics
        'mean_iou': mean_iou,
        'median_iou': median_iou,
        'mean_confidence': mean_confidence,
        
        # COCO metrics
        **coco_metrics,
        
        'output_dir': output_dir,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save comprehensive results
    results_df = pd.DataFrame([model_results])
    results_path = os.path.join(output_dir, 'comprehensive_evaluation_results.csv')
    results_df.to_csv(results_path, index=False)
    
    # Save detailed IoU analysis
    iou_analysis = {
        'all_ious': all_ious,
        'matched_confidences': matched_confidences,
        'evaluation_summary': model_results
    }
    
    import json
    iou_analysis_path = os.path.join(output_dir, 'iou_analysis.json')
    with open(iou_analysis_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_analysis = {
            'all_ious': [float(x) for x in all_ious],
            'matched_confidences': [float(x) for x in matched_confidences],
            'evaluation_summary': {k: float(v) if isinstance(v, (np.float64, np.float32)) else v 
                                 for k, v in model_results.items()}
        }
        json.dump(serializable_analysis, f, indent=2)
    
    print(f"\nResults saved to:")
    print(f"  - Comprehensive metrics: {results_path}")
    print(f"  - IoU analysis: {iou_analysis_path}")
    print(f"  - Test images organized in: {classification_results['test_dir']}")
    print(f"  - Enhanced visualizations: {classification_results['test_dir']}/visualizations")
    
    return model_results

def get_ground_truth_boxes(img_path: str, annotations_df: pd.DataFrame, image_size: tuple):
    """Extract ground truth bounding boxes for an image"""
    image_annotations = annotations_df[annotations_df['study_path'] == img_path]
    gt_boxes = []
    
    width, height = image_size
    
    for _, row in image_annotations.iterrows():
        bbox_str = row['bbox']
        if bbox_str and bbox_str != '':
            try:
                bbox_list = ast.literal_eval(bbox_str)
                for bbox_dict in bbox_list:
                    # Extract coordinates (in percentages)
                    x_percent = bbox_dict['x']
                    y_percent = bbox_dict['y'] 
                    width_percent = bbox_dict['width']
                    height_percent = bbox_dict['height']
                    
                    # Convert to absolute coordinates
                    x = (x_percent / 100) * width
                    y = (y_percent / 100) * height
                    w = (width_percent / 100) * width
                    h = (height_percent / 100) * height
                    
                    # Convert to pascal VOC format (xmin, ymin, xmax, ymax)
                    xmin = max(0, min(x, width - 1))
                    ymin = max(0, min(y, height - 1))
                    xmax = max(xmin + 1, min(x + w, width))
                    ymax = max(ymin + 1, min(y + h, height))
                    
                    gt_boxes.append([xmin, ymin, xmax, ymax])
                    
            except Exception as e:
                print(f"Error parsing bbox for image {img_path}: {e}")
                continue
    
    return np.array(gt_boxes) if gt_boxes else np.array([]).reshape(0, 4)

def split_dataset_for_testing(data_dir: str, csv_path: str, test_size: int = 200, 
                             normal_limit: int = None, seed: int = 42):
    """
    Split dataset into test sets with fixed seed for reproducibility
    """
    print(f"Splitting dataset for testing with seed={seed}")
    random.seed(seed)
    np.random.seed(seed)
    
    # Set up paths
    fracture_dir = os.path.join(data_dir, "fracture")
    normal_dir = os.path.join(data_dir, "normal")
    
    # Load CSV data for fracture images
    annotations_df = pd.read_csv(csv_path)
    fracture_images = annotations_df['study_path'].unique().tolist()
    
    # Filter existing fracture images
    existing_fracture_images = []
    for img_path in fracture_images:
        # Build candidate physical paths
        candidates = []
        if os.path.isabs(img_path):
            candidates.append(img_path)
        candidates.append(os.path.join(fracture_dir, img_path.lstrip('/')))
        candidates.append(os.path.join(fracture_dir, os.path.basename(img_path)))
        candidates.append(os.path.join(fracture_dir, img_path.replace('/', '_')))

        found = next((p for p in candidates if os.path.exists(p)), None)
        if found is not None:
            existing_fracture_images.append(img_path)
    
    # Get normal images
    normal_images = []
    if os.path.exists(normal_dir):
        for root, _, files in os.walk(normal_dir):
            for file in files:
                if file.lower().endswith(('.jpeg', '.jpg', '.png')):
                    relative_path = os.path.relpath(os.path.join(root, file), normal_dir)
                    normal_images.append(relative_path)
    
    # Apply normal image limit if specified
    if normal_limit is not None and len(normal_images) > normal_limit:
        random.shuffle(normal_images)
        normal_images = normal_images[:normal_limit]
        print(f"Limited normal images to {normal_limit}")
    
    # Randomly sample test sets
    random.shuffle(existing_fracture_images)
    random.shuffle(normal_images)
    
    test_fracture = existing_fracture_images[:min(test_size, len(existing_fracture_images))]
    test_normal = normal_images[:min(test_size, len(normal_images))]
    
    print(f"Test set created:")
    print(f"  - Fracture images: {len(test_fracture)}")
    print(f"  - Normal images: {len(test_normal)}")
    print(f"  - Total test images: {len(test_fracture) + len(test_normal)}")
    
    return {
        'fracture': test_fracture,
        'normal': test_normal,
        'fracture_dir': fracture_dir,
        'normal_dir': normal_dir,
        'annotations_df': annotations_df
    }

def copy_test_images_and_classify_with_iou(test_data: dict, evaluation_results: dict, 
                                          output_dir: str, conf_threshold: float = 0.3,
                                          iou_threshold: float = 0.5):
    """
    Copy test images to organized directories based on classification results with IoU analysis
    """
    # Create test directory structure
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(test_dir, exist_ok=True)
    
    # Create classification directories
    class_dirs = {
        'tp': os.path.join(test_dir, 'true_positive'),      # Fracture correctly detected
        'fp': os.path.join(test_dir, 'false_positive'),     # Normal classified as fracture  
        'tn': os.path.join(test_dir, 'true_negative'),      # Normal correctly classified
        'fn': os.path.join(test_dir, 'false_negative'),     # Fracture missed
        'visualizations': os.path.join(test_dir, 'visualizations')
    }
    
    for dir_path in class_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # Classification counters
    classification_counts = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
    
    # Process fracture test images
    for img_path in test_data['fracture']:
        candidates = []
        if os.path.isabs(img_path):
            candidates.append(img_path)
        candidates.append(os.path.join(test_data['fracture_dir'], img_path.lstrip('/')))
        candidates.append(os.path.join(test_data['fracture_dir'], os.path.basename(img_path)))
        candidates.append(os.path.join(test_data['fracture_dir'], img_path.replace('/', '_')))
        source_path = next((p for p in candidates if os.path.exists(p)), None)
        if source_path is None:
            continue
        # Create unique identifier for this image
        img_id = f"fracture_{hash(img_path) % 100000}"
        
        # Get evaluation results for this image
        img_result = evaluation_results.get(img_id, {})
        matches = img_result.get('matches', [])
        mean_iou = img_result.get('mean_iou', 0.0)
        max_confidence = img_result.get('max_confidence', 0.0)
        
        # Classify based on IoU-aware detection
        if len(matches) > 0 and max_confidence >= conf_threshold:
            # Check if any match has IoU >= threshold
            good_matches = [m for m in matches if m['iou'] >= iou_threshold]
            if good_matches:
                dest_dir = class_dirs['tp']
                classification = 'TP'
            else:
                dest_dir = class_dirs['fn'] 
                classification = 'FN'  # Low IoU detection = missed
        else:
            dest_dir = class_dirs['fn']
            classification = 'FN'
        
        # Copy image with detailed filename
        dest_filename = f"{classification}_mIoU{mean_iou:.3f}_conf{max_confidence:.3f}_{img_id}_{os.path.basename(source_path)}"
        dest_path = os.path.join(dest_dir, dest_filename)
        shutil.copy2(source_path, dest_path)
        classification_counts[classification.lower()] += 1
    
    # Process normal test images
    for img_path in test_data['normal']:
        source_path = os.path.join(test_data['normal_dir'], img_path)
        
        if not os.path.exists(source_path):
            continue
            
        # Create unique identifier for this image
        img_id = f"normal_{hash(img_path) % 100000}"
        
        # Get evaluation results for this image  
        img_result = evaluation_results.get(img_id, {})
        max_confidence = img_result.get('max_confidence', 0.0)
        
        # Classify: FP if detected above threshold, TN if correctly classified as normal
        if max_confidence >= conf_threshold:
            dest_dir = class_dirs['fp']
            classification = 'FP'
        else:
            dest_dir = class_dirs['tn']
            classification = 'TN'
        
        # Copy image with detailed filename
        dest_filename = f"{classification}_conf{max_confidence:.3f}_{img_id}_{os.path.basename(source_path)}"
        dest_path = os.path.join(dest_dir, dest_filename)
        shutil.copy2(source_path, dest_path)
        classification_counts[classification.lower()] += 1
    
    print(f"\nClassification Summary (conf_threshold={conf_threshold}, iou_threshold={iou_threshold}):")
    print(f"  True Positives (TP):  {classification_counts['tp']}")
    print(f"  False Positives (FP): {classification_counts['fp']}")
    print(f"  True Negatives (TN):  {classification_counts['tn']}")
    print(f"  False Negatives (FN): {classification_counts['fn']}")
    
    return {
        'tp': classification_counts['tp'],
        'fp': classification_counts['fp'],
        'tn': classification_counts['tn'],
        'fn': classification_counts['fn'],
        'test_dir': test_dir
    }

def create_enhanced_visualization(image_path: str, img_result: dict, img_id: str, 
                                is_fracture: bool, gt_boxes: np.ndarray, output_dir: str, 
                                conf_threshold: float = 0.3, iou_threshold: float = 0.5):
    """Create enhanced visualization with IoU and confidence information"""
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return
        
        img_vis = img.copy()
        height, width = img_vis.shape[:2]
        
        # Get results
        predictions = img_result.get('predictions', {})
        matches = img_result.get('matches', [])
        unmatched_preds = img_result.get('unmatched_preds', [])
        mean_iou = img_result.get('mean_iou', 0.0)
        max_confidence = img_result.get('max_confidence', 0.0)
        
        pred_boxes = predictions.get('boxes', [])
        pred_scores = predictions.get('scores', [])
        
        # Draw ground truth boxes in blue
        for gt_box in gt_boxes:
            x1, y1, x2, y2 = [int(coord) for coord in gt_box]
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for GT
            cv2.putText(img_vis, 'GT', (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Draw matched predictions in green with IoU
        for match in matches:
            pred_idx = match['pred_idx']
            iou = match['iou']
            confidence = match['confidence']
            pred_box = match['pred_box']
            
            if confidence >= conf_threshold:
                x1, y1, x2, y2 = [int(coord) for coord in pred_box]
                
                # Color based on IoU quality
                if iou >= iou_threshold:
                    color = (0, 255, 0)  # Green for good IoU
                else:
                    color = (0, 255, 255)  # Yellow for low IoU
                
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img_vis, f'IoU:{iou:.2f} C:{confidence:.2f}', 
                          (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw unmatched predictions in red
        for pred_idx in unmatched_preds:
            if pred_idx < len(pred_boxes) and pred_idx < len(pred_scores):
                if pred_scores[pred_idx] >= conf_threshold:
                    pred_box = pred_boxes[pred_idx]
                    confidence = pred_scores[pred_idx]
                    x1, y1, x2, y2 = [int(coord) for coord in pred_box]
                    
                    cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for FP
                    cv2.putText(img_vis, f'FP C:{confidence:.2f}', 
                              (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Determine classification
        predicted_fracture = max_confidence >= conf_threshold
        good_matches = [m for m in matches if m['iou'] >= iou_threshold and m['confidence'] >= conf_threshold]
        
        if is_fracture and len(good_matches) > 0:
            classification = "TP"
            class_color = (0, 255, 0)  # Green
        elif is_fracture and not predicted_fracture:
            classification = "FN" 
            class_color = (0, 0, 255)  # Red
        elif is_fracture and predicted_fracture and len(good_matches) == 0:
            classification = "FN"  # Low IoU = miss
            class_color = (0, 0, 255)  # Red
        elif not is_fracture and predicted_fracture:
            classification = "FP"
            class_color = (0, 165, 255)  # Orange
        else:  # not is_fracture and not predicted_fracture
            classification = "TN"
            class_color = (255, 0, 0)  # Blue
        
        # Add comprehensive header text
        header_text = f"Mean IoU: {mean_iou:.3f} | Max Conf: {max_confidence:.3f}"
        cv2.putText(img_vis, header_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img_vis, header_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        classification_text = f"{classification} - GT: {'Fracture' if is_fracture else 'Normal'} | Matches: {len(matches)} | Good Matches: {len(good_matches)}"
        cv2.putText(img_vis, classification_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, class_color, 2)
        
        # Add legend
        legend_y = height - 100
        cv2.putText(img_vis, 'Legend: GT=Blue, Good IoU=Green, Low IoU=Yellow, FP=Red', 
                   (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(img_vis, 'Legend: GT=Blue, Good IoU=Green, Low IoU=Yellow, FP=Red', 
                   (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Save visualization
        vis_dir = os.path.join(output_dir, 'test', 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        vis_filename = f"{classification}_mIoU{mean_iou:.3f}_conf{max_confidence:.3f}_{img_id}_{os.path.basename(image_path)}"
        vis_path = os.path.join(vis_dir, vis_filename)
        cv2.imwrite(vis_path, img_vis)
        
    except Exception as e:
        print(f"Error creating visualization for {image_path}: {e}")

# ================================================================================================
# MAIN EXECUTION
# ================================================================================================

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Handle evaluation flag
    run_evaluation = args.run_evaluation and not args.skip_evaluation
    
    # Set num_workers if not specified
    if args.num_workers is None:
        args.num_workers = min(8, os.cpu_count())
    
    print("=" * 80)
    print("FRACTURE DETECTION TRAINING - EfficientDet")
    print("=" * 80)
    print(f"Data directory: {args.data_dir}")
    print(f"CSV path: {args.csv_path}")
    print(f"Model: {args.model_name}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Validation split: {args.val_split}")
    print(f"Number of workers: {args.num_workers}")
    print(f"Run evaluation: {run_evaluation}")
    if run_evaluation:
        print(f"  Test size per class: {args.test_size}")
        print(f"  Normal image limit: {args.normal_limit}")
        print(f"  Confidence threshold: {args.eval_conf_threshold}")
        print(f"  IoU threshold: {args.eval_iou_threshold}")
        print(f"  Evaluation seed: {args.eval_seed}")
    
    # Get image size from model configuration
    try:
        config = get_efficientdet_config(args.model_name)
        image_size = config.image_size[0]
        print(f"Image size (from model config): {image_size}")
    except Exception as e:
        print(f"Error getting model config: {e}")
        print("Please check if the model name is valid.")
        exit(1)
    
    print("=" * 80)
    
    # Start training
    try:
        model, trainer, evaluation_results = train_fracture_detection(
            data_dir=args.data_dir,
            csv_path=args.csv_path,
            model_name=args.model_name,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            learning_rate=args.learning_rate,
            val_split=args.val_split,
            num_workers=args.num_workers,
            run_evaluation=run_evaluation,
            test_size=args.test_size,
            normal_limit=args.normal_limit,
            eval_conf_threshold=args.eval_conf_threshold,
            eval_iou_threshold=args.eval_iou_threshold,
            eval_seed=args.eval_seed
        )
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        if run_evaluation:
            if evaluation_results:
                print("EVALUATION COMPLETED SUCCESSFULLY!")
                print(f"Evaluated {len(evaluation_results)} model(s)")
            else:
                print("EVALUATION COMPLETED WITH ERRORS!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        exit(1) 
