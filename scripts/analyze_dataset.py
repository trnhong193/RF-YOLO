"""
Phân tích và visualize dataset
"""

import h5py
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import seaborn as sns

from config import Config

class DatasetAnalyzer:
    def __init__(self, config):
        self.config = config
        self.data_dir = Path(config.DATA_DIR)
    
    def analyze_split(self, split='train'):
        """Phân tích một split của dataset"""
        
        print(f"\n{'='*60}")
        print(f"Analyzing {split} split")
        print(f"{'='*60}")
        
        # Load data
        h5_path = self.data_dir / f'{split}.h5'
        json_path = self.data_dir / f'{split}_annotations.json'
        
        with h5py.File(h5_path, 'r') as f:
            spectrograms = f['spectrograms'][:]
        
        with open(json_path, 'r') as f:
            annotations = json.load(f)
        
        # Basic statistics
        print(f"\nBasic Statistics:")
        print(f"  Number of samples: {len(spectrograms)}")
        print(f"  Spectrogram shape: {spectrograms[0].shape}")
        print(f"  Memory size: {spectrograms.nbytes / 1e6:.2f} MB")
        
        # Analyze annotations
        self.analyze_annotations(annotations)
        
        # Analyze spectrograms
        self.analyze_spectrograms(spectrograms, split)
        
        return spectrograms, annotations
    
    def analyze_annotations(self, annotations):
        """Phân tích annotations"""
        
        # Count classes
        all_classes = []
        num_objects_per_image = []
        
        for annots in annotations:
            num_objects_per_image.append(len(annots))
            for annot in annots:
                all_classes.append(annot['class_name'])
        
        class_counts = Counter(all_classes)
        
        print(f"\nAnnotation Statistics:")
        print(f"  Total objects: {len(all_classes)}")
        print(f"  Avg objects per image: {np.mean(num_objects_per_image):.2f}")
        print(f"  Max objects per image: {max(num_objects_per_image)}")
        
        print(f"\nClass Distribution:")
        for class_name, count in class_counts.most_common():
            percentage = count / len(all_classes) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        # Visualize class distribution
        self.plot_class_distribution(class_counts)
        
        # Analyze bounding boxes
        self.analyze_bboxes(annotations)
    
    def analyze_bboxes(self, annotations):
        """Phân tích bounding boxes"""
        
        widths = []
        heights = []
        areas = []
        
        for annots in annotations:
            for annot in annots:
                bbox = annot['bbox']
                widths.append(bbox[2])
                heights.append(bbox[3])
                areas.append(bbox[2] * bbox[3])
        
        print(f"\nBounding Box Statistics:")
        print(f"  Width - Mean: {np.mean(widths):.3f}, Std: {np.std(widths):.3f}")
        print(f"  Height - Mean: {np.mean(heights):.3f}, Std: {np.std(heights):.3f}")
        print(f"  Area - Mean: {np.mean(areas):.3f}, Std: {np.std(areas):.3f}")
        
        # Plot distributions
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        axes[0].hist(widths, bins=50, edgecolor='black')
        axes[0].set_xlabel('Width (normalized)')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Bounding Box Width Distribution')
        
        axes[1].hist(heights, bins=50, edgecolor='black')
        axes[1].set_xlabel('Height (normalized)')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Bounding Box Height Distribution')
        
        axes[2].hist(areas, bins=50, edgecolor='black')
        axes[2].set_xlabel('Area (normalized)')
        axes[2].set_ylabel('Count')
        axes[2].set_title('Bounding Box Area Distribution')
        
        plt.tight_layout()
        plt.savefig('bbox_distributions.png', dpi=150)
        print(f"  Saved bbox distributions to bbox_distributions.png")
        plt.close()
    
    def analyze_spectrograms(self, spectrograms, split):
        """Phân tích spectrograms"""
        
        print(f"\nSpectrogram Statistics:")
        print(f"  Mean: {np.mean(spectrograms):.4f}")
        print(f"  Std: {np.std(spectrograms):.4f}")
        print(f"  Min: {np.min(spectrograms):.4f}")
        print(f"  Max: {np.max(spectrograms):.4f}")
        
        # Plot sample spectrograms
        self.plot_sample_spectrograms(spectrograms, split)
    
    def plot_class_distribution(self, class_counts):
        """Vẽ class distribution"""
        
        plt.figure(figsize=(12, 6))
        
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        plt.bar(classes, counts, edgecolor='black')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Class Distribution')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plt.savefig('class_distribution.png', dpi=150)
        print(f"  Saved class distribution to class_distribution.png")
        plt.close()
    
    def plot_sample_spectrograms(self, spectrograms, split, num_samples=9):
        """Vẽ sample spectrograms"""
        
        indices = np.random.choice(len(spectrograms), 
                                  size=min(num_samples, len(spectrograms)),
                                  replace=False)
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flatten()
        
        for i, idx in enumerate(indices):
            spec = spectrograms[idx]
            axes[i].imshow(spec, aspect='auto', origin='lower', 
                          cmap='viridis', interpolation='nearest')
            axes[i].set_title(f'Sample {idx}')
            axes[i].set_xlabel('Time Steps')
            axes[i].set_ylabel('Frequency Bins')
        
        plt.tight_layout()
        plt.savefig(f'sample_spectrograms_{split}.png', dpi=150)
        print(f"  Saved sample spectrograms to sample_spectrograms_{split}.png")
        plt.close()
    
    def compare_splits(self):
        """So sánh các splits"""
        
        splits = ['train', 'val', 'test']
        split_stats = {}
        
        for split in splits:
            h5_path = self.data_dir / f'{split}.h5'
            if not h5_path.exists():
                continue
            
            with h5py.File(h5_path, 'r') as f:
                spectrograms = f['spectrograms'][:]
            
            split_stats[split] = {
                'num_samples': len(spectrograms),
                'mean': np.mean(spectrograms),
                'std': np.std(spectrograms)
            }
        
        print(f"\n{'='*60}")
        print(f"Split Comparison")
        print(f"{'='*60}")
        
        for split, stats in split_stats.items():
            print(f"\n{split.upper()}:")
            for key, value in stats.items():
                print(f"  {key}: {value}")

def main():
    config = Config()
    analyzer = DatasetAnalyzer(config)
    
    # Analyze each split
    for split in ['train', 'val', 'test']:
        try:
            analyzer.analyze_split(split)
        except Exception as e:
            print(f"Error analyzing {split}: {e}")
    
    # Compare splits
    analyzer.compare_splits()

if __name__ == '__main__':
    main()