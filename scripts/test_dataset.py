#!/usr/bin/env python3
"""
Ví dụ sử dụng dataset loader
"""

import torch
from pathlib import Path
from torch.utils.data import DataLoader

from config import Config
from data.dataset import DroneDataset, DroneDatasetWithAugmentation
import matplotlib.pyplot as plt
import numpy as np


def visualize_batch(images, targets, num_samples=4):
    """Visualize batch of samples"""
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 10))
    
    for i in range(num_samples):
        # Image
        img = images[i].squeeze().numpy()
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f'Sample {i}')
        axes[i, 0].axis('off')
        
        # Target heatmap (sum of all channels for visualization)
        target = targets[i].numpy()
        
        # Create heatmap: show object confidence in each grid cell
        S = target.shape[0]
        heatmap = np.zeros((S, S))
        
        for grid_i in range(S):
            for grid_j in range(S):
                # Get max object confidence from all boxes
                cell = target[grid_i, grid_j, :]
                max_conf = 0
                for b in range(2):  # 2 boxes per cell
                    conf = cell[b * 5]
                    max_conf = max(max_conf, conf)
                heatmap[grid_i, grid_j] = max_conf
        
        im = axes[i, 1].imshow(heatmap, cmap='hot')
        axes[i, 1].set_title(f'Object Confidence Heatmap')
        axes[i, 1].axis('off')
        plt.colorbar(im, ax=axes[i, 1])
    
    plt.tight_layout()
    return fig


def main():
    config = Config()
    
    print("Testing Dataset Loading...")
    print(f"Classes: {config.CLASSES}")
    print(f"Number of classes: {config.NUM_CLASSES}")
    print()
    
    # Test train dataset
    try:
        print("Loading TRAIN dataset...")
        train_dataset = DroneDatasetWithAugmentation(config, split='train', augment=True)
        print(f"✓ Train dataset loaded: {len(train_dataset)} samples")
        
        # Create dataloader
        train_loader = DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0
        )
        
        # Get a batch
        for batch_idx, (images, targets) in enumerate(train_loader):
            print(f"\nBatch {batch_idx}:")
            print(f"  Images shape: {images.shape}")
            print(f"  Targets shape: {targets.shape}")
            print(f"  Image dtype: {images.dtype}, range: [{images.min():.3f}, {images.max():.3f}]")
            print(f"  Target range: [{targets.min():.3f}, {targets.max():.3f}]")
            
            # Count objects in batch
            total_objects = 0
            for i in range(targets.shape[0]):
                for j in range(config.GRID_SIZE):
                    for k in range(config.GRID_SIZE):
                        cell = targets[i, j, k, :]
                        # Count boxes with objects
                        for b in range(config.NUM_BOXES):
                            if cell[b * 5] > 0:
                                total_objects += 1
            
            print(f"  Total objects in batch: {total_objects}")
            
            if batch_idx == 0:
                # Visualize first batch
                fig = visualize_batch(images, targets, num_samples=min(4, images.shape[0]))
                plt.savefig('sample_batch.png', dpi=100, bbox_inches='tight')
                print(f"\n✓ Saved visualization to sample_batch.png")
                plt.close()
            
            if batch_idx == 2:
                break
    
    except FileNotFoundError as e:
        print(f"✗ Train dataset not found: {e}")
    except Exception as e:
        print(f"✗ Error loading train dataset: {e}")
    
    # Test val dataset
    try:
        print("\n" + "="*60)
        print("Loading VAL dataset...")
        val_dataset = DroneDataset(config, split='val')
        print(f"✓ Val dataset loaded: {len(val_dataset)} samples")
        
        # Get one sample
        image, target = val_dataset[0]
        print(f"  Image shape: {image.shape}")
        print(f"  Target shape: {target.shape}")
    
    except FileNotFoundError as e:
        print(f"✗ Val dataset not found: {e}")
    except Exception as e:
        print(f"✗ Error loading val dataset: {e}")
    
    # Test test dataset
    try:
        print("\n" + "="*60)
        print("Loading TEST dataset...")
        test_dataset = DroneDataset(config, split='test')
        print(f"✓ Test dataset loaded: {len(test_dataset)} samples")
        
        # Get one sample
        image, target = test_dataset[0]
        print(f"  Image shape: {image.shape}")
        print(f"  Target shape: {target.shape}")
    
    except FileNotFoundError as e:
        print(f"✗ Test dataset not found: {e}")
    except Exception as e:
        print(f"✗ Error loading test dataset: {e}")
    
    print("\n" + "="*60)
    print("Dataset test completed!")


if __name__ == '__main__':
    main()
