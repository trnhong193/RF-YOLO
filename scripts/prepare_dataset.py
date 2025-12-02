#!/usr/bin/env python3
"""
Script để validate và prepare YOLOv5 format dataset
"""

import argparse
from pathlib import Path
import sys

from config import Config
from data.data_utils import YOLOv5DatasetValidator, create_dataset_yaml


def main():
    parser = argparse.ArgumentParser(
        description='Validate YOLOv5 format dataset'
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='./data/generated',
        help='Root directory of dataset'
    )
    parser.add_argument(
        '--output-yaml',
        type=str,
        default='./data.yaml',
        help='Path to save data.yaml'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed information'
    )
    
    args = parser.parse_args()
    
    config = Config()
    dataset_dir = Path(args.dataset_dir)
    
    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found: {dataset_dir}")
        sys.exit(1)
    
    print(f"Validating dataset at: {dataset_dir}")
    print(f"Classes: {config.CLASSES}")
    
    # Validate dataset
    validator = YOLOv5DatasetValidator(dataset_dir, config.CLASSES)
    results = validator.validate(verbose=args.verbose)
    
    if not results['valid']:
        print(f"\n✗ Dataset validation FAILED")
        print(f"Found {len(results['errors'])} errors")
        sys.exit(1)
    
    print(f"\n✓ Dataset validation PASSED")
    
    # Create data.yaml
    create_dataset_yaml(
        str(dataset_dir.absolute()),
        config.CLASSES,
        output_path=args.output_yaml
    )
    
    print(f"✓ Data yaml created at: {args.output_yaml}")
    
    # Print summary
    print("\n" + "="*60)
    print("Dataset Summary:")
    print("="*60)
    
    total_images = 0
    total_annotations = 0
    
    for split in ['train', 'val', 'test']:
        if split in results['stats']:
            stats = results['stats'][split]
            print(f"\n{split.upper()} SET:")
            print(f"  Images: {stats['total_images']}")
            print(f"  Annotations: {stats['total_annotations']}")
            print(f"  Class distribution:")
            for cls, count in stats['class_distribution'].items():
                if count > 0:
                    print(f"    - {cls}: {count}")
            
            total_images += stats['total_images']
            total_annotations += stats['total_annotations']
    
    print(f"\nTOTAL:")
    print(f"  Images: {total_images}")
    print(f"  Annotations: {total_annotations}")
    
    print("\n" + "="*60)
    print("Next steps:")
    print("1. Run training: python train.py")
    print("2. Evaluate: python inference.py")
    print("="*60)


if __name__ == '__main__':
    main()
