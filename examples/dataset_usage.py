#!/usr/bin/env python3
"""
Example: How to use the YOLOv5 format dataset
"""

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from config import Config
from data.dataset import DroneDataset, DroneDatasetWithAugmentation


def example_1_load_dataset():
    """Example 1: Tải dataset"""
    print("=" * 60)
    print("Example 1: Loading Dataset")
    print("=" * 60)
    
    config = Config()
    
    # Load training dataset với augmentation
    train_dataset = DroneDatasetWithAugmentation(config, split='train', augment=True)
    print(f"Training dataset: {len(train_dataset)} samples")
    
    # Load validation dataset không augmentation
    val_dataset = DroneDataset(config, split='val')
    print(f"Validation dataset: {len(val_dataset)} samples")
    
    # Load test dataset
    test_dataset = DroneDataset(config, split='test')
    print(f"Test dataset: {len(test_dataset)} samples")
    
    return config, train_dataset, val_dataset, test_dataset


def example_2_get_single_sample(dataset, config):
    """Example 2: Lấy một sample từ dataset"""
    print("\n" + "=" * 60)
    print("Example 2: Get Single Sample")
    print("=" * 60)
    
    # Lấy sample thứ 0
    spectrogram, target = dataset[0]
    
    print(f"Spectrogram shape: {spectrogram.shape}")  # (1, 256, 256)
    print(f"Target shape: {target.shape}")             # (16, 16, 20)
    print(f"Spectrogram dtype: {spectrogram.dtype}")
    print(f"Spectrogram range: [{spectrogram.min():.3f}, {spectrogram.max():.3f}]")
    
    # Analyze target
    print("\nTarget analysis:")
    total_objects = 0
    for i in range(config.GRID_SIZE):
        for j in range(config.GRID_SIZE):
            cell = target[i, j, :]
            # Check each box in cell
            for b in range(config.NUM_BOXES):
                if cell[b * 5] > 0:  # Object confidence > 0
                    total_objects += 1
                    x_offset = cell[b * 5 + 1].item()
                    y_offset = cell[b * 5 + 2].item()
                    w = cell[b * 5 + 3].item()
                    h = cell[b * 5 + 4].item()
                    print(f"  Object at grid ({i}, {j}), box {b}:")
                    print(f"    x_offset={x_offset:.3f}, y_offset={y_offset:.3f}")
                    print(f"    width={w:.3f}, height={h:.3f}")
    
    print(f"Total objects: {total_objects}")


def example_3_batch_loading(train_loader):
    """Example 3: Tải batch data"""
    print("\n" + "=" * 60)
    print("Example 3: Batch Loading")
    print("=" * 60)
    
    for batch_idx, (images, targets) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Images shape: {images.shape}")        # (batch_size, 1, 256, 256)
        print(f"  Targets shape: {targets.shape}")      # (batch_size, 16, 16, 20)
        print(f"  Images dtype: {images.dtype}")
        print(f"  Targets dtype: {targets.dtype}")
        
        # Count objects per image
        batch_size = images.shape[0]
        for img_idx in range(batch_size):
            n_objects = (targets[img_idx, :, :, 0] > 0).sum().item()
            if batch_idx == 0:
                print(f"  Image {img_idx}: {n_objects} objects")
        
        if batch_idx == 1:
            break


def example_4_coordinate_conversion(dataset, config):
    """Example 4: Convert coordinates"""
    print("\n" + "=" * 60)
    print("Example 4: Coordinate Conversion")
    print("=" * 60)
    
    # Get image with annotations
    image, annotations = dataset.get_image_with_annotations(0)
    print(f"Image shape: {image.shape}")
    print(f"Number of annotations: {len(annotations)}")
    
    if annotations:
        print(f"\nFirst annotation (YOLO format):")
        ann = annotations[0]
        print(f"  class_id: {ann['class_id']}")
        print(f"  x_center (norm): {ann['x_center']:.3f}")
        print(f"  y_center (norm): {ann['y_center']:.3f}")
        print(f"  width (norm): {ann['width']:.3f}")
        print(f"  height (norm): {ann['height']:.3f}")
        
        # Convert to xyxy
        image_size = image.shape
        xyxy = dataset.convert_yolo_to_xyxy(ann, image_size)
        print(f"\nConverted to xyxy (pixel):")
        print(f"  x1={xyxy[0]}, y1={xyxy[1]}")
        print(f"  x2={xyxy[2]}, y2={xyxy[3]}")
        
        # Convert back to YOLO
        yolo_back = dataset.convert_xyxy_to_yolo(xyxy, image_size)
        print(f"\nConverted back to YOLO (normalized):")
        print(f"  x_center={yolo_back['x_center']:.3f}")
        print(f"  y_center={yolo_back['y_center']:.3f}")
        print(f"  width={yolo_back['width']:.3f}")
        print(f"  height={yolo_back['height']:.3f}")


def example_5_visualize():
    """Example 5: Visualize samples"""
    print("\n" + "=" * 60)
    print("Example 5: Visualization")
    print("=" * 60)
    
    config = Config()
    dataset = DroneDataset(config, split='train')
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for idx in range(6):
        row = idx // 3
        col = idx % 3
        
        image, annotations = dataset.get_image_with_annotations(idx)
        
        # Draw image
        ax = axes[row, col]
        ax.imshow(image, cmap='gray')
        ax.set_title(f'Sample {idx} ({len(annotations)} objects)')
        
        # Draw bounding boxes
        for ann in annotations:
            class_id = ann['class_id']
            class_name = config.CLASSES[class_id]
            
            x1, y1, x2, y2 = dataset.convert_yolo_to_xyxy(
                ann, image.shape
            )
            
            # Draw rectangle
            rect = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            ax.text(x1, y1 - 5, class_name, fontsize=8,
                   color='red', bbox=dict(facecolor='yellow', alpha=0.7))
        
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('visualization_example.png', dpi=100, bbox_inches='tight')
    print("Saved to visualization_example.png")
    plt.close()


def main():
    """Run all examples"""
    
    try:
        # Example 1: Load dataset
        config, train_dataset, val_dataset, test_dataset = example_1_load_dataset()
        
        # Example 2: Get single sample
        example_2_get_single_sample(train_dataset, config)
        
        # Example 3: Batch loading
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        example_3_batch_loading(train_loader)
        
        # Example 4: Coordinate conversion
        example_4_coordinate_conversion(train_dataset, config)
        
        # Example 5: Visualization
        example_5_visualize()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure your dataset structure is:")
        print("data/")
        print("├── images/")
        print("│   ├── train/")
        print("│   ├── val/")
        print("│   └── test/")
        print("└── labels/")
        print("    ├── train/")
        print("    ├── val/")
        print("    └── test/")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
