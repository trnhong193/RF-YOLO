"""
Visualize detection results
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from config import Config
from inference import DroneDetector
from data.dataset import DroneDataset
from utils.visualization import Visualizer

def visualize_random_samples(detector, dataset, num_samples=10, save_dir='results'):
    """Visualize random samples với predictions"""
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    indices = np.random.choice(len(dataset), 
                              size=min(num_samples, len(dataset)),
                              replace=False)
    
    visualizer = Visualizer(detector.config)
    
    for idx in indices:
        print(f"\nProcessing sample {idx}...")
        
        spectrogram, target = dataset[idx]
        
        # Predict
        detections = detector.predict(spectrogram)
        
        # Ground truth
        gt_detections = detector.decode_ground_truth(target)
        
        # Print detections
        print(f"  Ground Truth: {len(gt_detections)} objects")
        for det in gt_detections:
            print(f"    - {detector.config.CLASSES[det['class']]}")
        
        print(f"  Predictions: {len(detections)} objects")
        for det in detections:
            print(f"    - {det['class_name']}: {det['confidence']:.3f}")
        
        # Visualize
        spec_np = spectrogram.squeeze().numpy()
        
        # Comparison plot
        save_path = save_dir / f'comparison_{idx}.png'
        visualizer.create_comparison_plot(
            spec_np, gt_detections, detections, save_path=save_path
        )
        
        # Individual prediction plot
        save_path = save_dir / f'prediction_{idx}.png'
        visualizer.draw_boxes_on_spectrogram(
            spec_np, detections, save_path=save_path, show=False
        )

def visualize_by_class(detector, dataset, class_name, num_samples=5, save_dir='results'):
    """Visualize samples của một class cụ thể"""
    
    save_dir = Path(save_dir) / class_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    class_id = detector.config.CLASSES.index(class_name)
    
    # Find samples chứa class này
    matching_indices = []
    for idx in range(len(dataset)):
        _, target = dataset[idx]
        gt_detections = detector.decode_ground_truth(target)
        
        for det in gt_detections:
            if det['class'] == class_id:
                matching_indices.append(idx)
                break
        
        if len(matching_indices) >= num_samples * 2:
            break
    
    if len(matching_indices) == 0:
        print(f"No samples found for class {class_name}")
        return
    
    # Random select
    selected = np.random.choice(matching_indices,
                               size=min(num_samples, len(matching_indices)),
                               replace=False)
    
    visualizer = Visualizer(detector.config)
    
    for i, idx in enumerate(selected):
        spectrogram, target = dataset[idx]
        detections = detector.predict(spectrogram)
        gt_detections = detector.decode_ground_truth(target)
        
        spec_np = spectrogram.squeeze().numpy()
        save_path = save_dir / f'{class_name}_{i}.png'
        
        visualizer.create_comparison_plot(
            spec_np, gt_detections, detections, save_path=save_path
        )
        
        print(f"Saved {save_path}")

def visualize_errors(detector, dataset, num_samples=10, save_dir='results/errors'):
    """Visualize detection errors (false positives, false negatives)"""
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    visualizer = Visualizer(detector.config)
    
    errors = []
    
    # Find errors
    for idx in range(len(dataset)):
        spectrogram, target = dataset[idx]
        detections = detector.predict(spectrogram)
        gt_detections = detector.decode_ground_truth(target)
        
        # Compute error
        num_gt = len(gt_detections)
        num_pred = len(detections)
        
        if num_gt != num_pred:
            error_score = abs(num_gt - num_pred)
            errors.append((idx, error_score, num_gt, num_pred))
        
        if len(errors) >= num_samples * 2:
            break
    
    # Sort by error score
    errors.sort(key=lambda x: x[1], reverse=True)
    
    # Visualize top errors
    for i, (idx, error_score, num_gt, num_pred) in enumerate(errors[:num_samples]):
        spectrogram, target = dataset[idx]
        detections = detector.predict(spectrogram)
        gt_detections = detector.decode_ground_truth(target)
        
        spec_np = spectrogram.squeeze().numpy()
        save_path = save_dir / f'error_{i}_gt{num_gt}_pred{num_pred}.png'
        
        visualizer.create_comparison_plot(
            spec_np, gt_detections, detections, save_path=save_path
        )
        
        print(f"Saved error case: GT={num_gt}, Pred={num_pred} -> {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pth',
                       help='Path to checkpoint')
    parser.add_argument('--split', type=str, default='test',
                       help='Dataset split to visualize')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to visualize')
    parser.add_argument('--mode', type=str, default='random',
                       choices=['random', 'by_class', 'errors'],
                       help='Visualization mode')
    parser.add_argument('--class_name', type=str, default='DJI_Phantom',
                       help='Class name for by_class mode')
    parser.add_argument('--save_dir', type=str, default='results/visualizations',
                       help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # Load config and detector
    config = Config()
    detector = DroneDetector(config, args.checkpoint)
    
    # Load dataset
    dataset = DroneDataset(config, split=args.split)
    
    print(f"\nVisualization mode: {args.mode}")
    print(f"Dataset split: {args.split}")
    print(f"Number of samples: {args.num_samples}")
    
    if args.mode == 'random':
        visualize_random_samples(detector, dataset, 
                                args.num_samples, args.save_dir)
    
    elif args.mode == 'by_class':
        visualize_by_class(detector, dataset, args.class_name,
                          args.num_samples, args.save_dir)
    
    elif args.mode == 'errors':
        visualize_errors(detector, dataset,
                        args.num_samples, args.save_dir)
    
    print(f"\n✓ Visualizations saved to {args.save_dir}")

if __name__ == '__main__':
    main()