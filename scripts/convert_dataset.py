#!/usr/bin/env python3
"""
Script để convert dataset từ các format khác sang YOLOv5 format
"""

import argparse
from pathlib import Path
import sys
from tqdm import tqdm
import cv2
import json

from data.data_utils import YOLOv5Converter


def convert_from_pascal_voc(xml_dir, image_dir, output_dir, classes_list):
    """Convert from PASCAL VOC XML format"""
    
    xml_files = list(Path(xml_dir).glob('*.xml'))
    
    if not xml_files:
        print(f"No XML files found in {xml_dir}")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting {len(xml_files)} PASCAL VOC annotations...")
    
    for xml_file in tqdm(xml_files):
        # Find corresponding image
        image_file = None
        for ext in ['.jpg', '.png', '.jpeg']:
            potential_image = Path(image_dir) / (xml_file.stem + ext)
            if potential_image.exists():
                image_file = potential_image
                break
        
        if image_file is None:
            print(f"Warning: No image found for {xml_file}")
            continue
        
        # Get image size
        img = cv2.imread(str(image_file))
        if img is None:
            print(f"Warning: Cannot read image {image_file}")
            continue
        
        h, w = img.shape[:2]
        
        # Convert
        output_txt = output_dir / (xml_file.stem + '.txt')
        try:
            n_anns = YOLOv5Converter.pascal_voc_to_yolo(
                xml_file, output_txt, (h, w), classes_list
            )
        except Exception as e:
            print(f"Error converting {xml_file}: {e}")
            continue
    
    print(f"Conversion complete. Output: {output_dir}")


def convert_from_coco(coco_json, image_dir, output_dir, classes_list):
    """Convert from COCO JSON format"""
    
    print(f"Converting COCO format from {coco_json}...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        n_images, n_anns = YOLOv5Converter.coco_to_yolo(
            coco_json, image_dir, output_dir, classes_list
        )
        print(f"Conversion complete. Images: {n_images}, Annotations: {n_anns}")
        print(f"Output: {output_dir}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def setup_yolov5_structure(dataset_dir, splits=['train', 'val', 'test']):
    """Create YOLOv5 directory structure"""
    
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = dataset_dir / 'images'
    labels_dir = dataset_dir / 'labels'
    
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)
    
    for split in splits:
        (images_dir / split).mkdir(exist_ok=True)
        (labels_dir / split).mkdir(exist_ok=True)
    
    print(f"Created YOLOv5 directory structure at {dataset_dir}")
    
    return dataset_dir


def main():
    parser = argparse.ArgumentParser(
        description='Convert dataset to YOLOv5 format'
    )
    parser.add_argument(
        'format',
        choices=['pascal-voc', 'coco', 'create-structure'],
        help='Source annotation format'
    )
    parser.add_argument(
        '--xml-dir',
        type=str,
        help='Directory containing XML files (for PASCAL VOC)'
    )
    parser.add_argument(
        '--coco-json',
        type=str,
        help='Path to COCO JSON file'
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        help='Directory containing images'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data/labels/converted',
        help='Output directory for converted labels'
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='./data',
        help='Root dataset directory (for create-structure)'
    )
    parser.add_argument(
        '--classes',
        type=str,
        nargs='+',
        help='List of class names'
    )
    
    args = parser.parse_args()
    
    # Default classes if not provided
    if args.classes is None:
        from config import Config
        config = Config()
        classes_list = config.CLASSES
    else:
        classes_list = args.classes
    
    if args.format == 'create-structure':
        setup_yolov5_structure(args.dataset_dir)
    
    elif args.format == 'pascal-voc':
        if not args.xml_dir or not args.image_dir:
            print("Error: --xml-dir and --image-dir required for PASCAL VOC format")
            sys.exit(1)
        
        convert_from_pascal_voc(
            args.xml_dir,
            args.image_dir,
            args.output_dir,
            classes_list
        )
    
    elif args.format == 'coco':
        if not args.coco_json or not args.image_dir:
            print("Error: --coco-json and --image-dir required for COCO format")
            sys.exit(1)
        
        convert_from_coco(
            args.coco_json,
            args.image_dir,
            args.output_dir,
            classes_list
        )


if __name__ == '__main__':
    main()
