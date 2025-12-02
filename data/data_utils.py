"""
Utility functions cho YOLOv5 format dataset
"""

import numpy as np
import cv2
import xml.etree.ElementTree as ET
from pathlib import Path
import json
from tqdm import tqdm


class YOLOv5Converter:
    """Convert various annotation formats to YOLOv5 format"""
    
    @staticmethod
    def pascal_voc_to_yolo(xml_file, output_txt, image_size, classes_list):
        """
        Convert PASCAL VOC XML format to YOLOv5 format
        
        Args:
            xml_file: Path to XML file
            output_txt: Path to output txt file
            image_size: (height, width) of image
            classes_list: List of class names
        """
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Get image size from XML
        size_elem = root.find('size')
        if size_elem is not None:
            width = int(size_elem.find('width').text)
            height = int(size_elem.find('height').text)
        else:
            height, width = image_size
        
        annotations = []
        
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in classes_list:
                continue
            
            class_id = classes_list.index(class_name)
            
            bbox = obj.find('bndbox')
            x_min = int(bbox.find('xmin').text)
            y_min = int(bbox.find('ymin').text)
            x_max = int(bbox.find('xmax').text)
            y_max = int(bbox.find('ymax').text)
            
            # Convert to YOLO format
            x_center = (x_min + x_max) / 2.0 / width
            y_center = (y_min + y_max) / 2.0 / height
            w = (x_max - x_min) / width
            h = (y_max - y_min) / height
            
            # Clip values
            x_center = np.clip(x_center, 0, 1)
            y_center = np.clip(y_center, 0, 1)
            w = np.clip(w, 0, 1)
            h = np.clip(h, 0, 1)
            
            annotations.append((class_id, x_center, y_center, w, h))
        
        # Write to file
        with open(output_txt, 'w') as f:
            for class_id, x_center, y_center, w, h in annotations:
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
        
        return len(annotations)
    
    @staticmethod
    def coco_to_yolo(coco_file, image_dir, output_label_dir, classes_list):
        """
        Convert COCO JSON format to YOLOv5 format
        
        Args:
            coco_file: Path to COCO JSON file
            image_dir: Directory containing images
            output_label_dir: Directory to save label files
            classes_list: List of class names
        """
        with open(coco_file, 'r') as f:
            coco_data = json.load(f)
        
        # Build image_id to image mapping
        images_map = {img['id']: img for img in coco_data['images']}
        
        # Build category_id to class_id mapping
        category_map = {}
        for cat in coco_data['categories']:
            cat_name = cat['name']
            if cat_name in classes_list:
                category_map[cat['id']] = classes_list.index(cat_name)
        
        # Group annotations by image
        annotations_by_image = {}
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(ann)
        
        # Process each image
        converted_count = 0
        for image_id, annotations in annotations_by_image.items():
            img_info = images_map[image_id]
            img_name = img_info['file_name']
            width = img_info['width']
            height = img_info['height']
            
            # Create output file
            label_file = Path(output_label_dir) / (Path(img_name).stem + '.txt')
            
            # Convert annotations
            yolo_annotations = []
            for ann in annotations:
                cat_id = ann['category_id']
                if cat_id not in category_map:
                    continue
                
                class_id = category_map[cat_id]
                
                # COCO format: [x, y, width, height]
                x, y, w, h = ann['bbox']
                
                # Convert to YOLO format (center coordinates)
                x_center = (x + w / 2.0) / width
                y_center = (y + h / 2.0) / height
                w_norm = w / width
                h_norm = h / height
                
                # Clip values
                x_center = np.clip(x_center, 0, 1)
                y_center = np.clip(y_center, 0, 1)
                w_norm = np.clip(w_norm, 0, 1)
                h_norm = np.clip(h_norm, 0, 1)
                
                yolo_annotations.append((class_id, x_center, y_center, w_norm, h_norm))
            
            # Write to file
            with open(label_file, 'w') as f:
                for class_id, x_center, y_center, w, h in yolo_annotations:
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
            
            converted_count += len(yolo_annotations)
        
        return len(annotations_by_image), converted_count
    
    @staticmethod
    def custom_to_yolo(annotations, image_size, classes_list, output_txt):
        """
        Convert custom format annotations to YOLOv5 format
        
        Args:
            annotations: List of dicts with keys: class_name, x1, y1, x2, y2 (pixel coords)
            image_size: (height, width)
            classes_list: List of class names
            output_txt: Path to output txt file
        """
        height, width = image_size
        
        yolo_annotations = []
        
        for ann in annotations:
            class_name = ann.get('class_name', ann.get('class'))
            if class_name not in classes_list:
                continue
            
            class_id = classes_list.index(class_name)
            
            # Get coordinates (assume xyxy format in pixels)
            if 'x1' in ann and 'y1' in ann and 'x2' in ann and 'y2' in ann:
                x1, y1, x2, y2 = ann['x1'], ann['y1'], ann['x2'], ann['y2']
            elif 'bbox' in ann:
                x1, y1, x2, y2 = ann['bbox']
            else:
                continue
            
            # Convert to YOLO format
            x_center = (x1 + x2) / 2.0 / width
            y_center = (y1 + y2) / 2.0 / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height
            
            # Clip values
            x_center = np.clip(x_center, 0, 1)
            y_center = np.clip(y_center, 0, 1)
            w = np.clip(w, 0, 1)
            h = np.clip(h, 0, 1)
            
            if w > 0 and h > 0:
                yolo_annotations.append((class_id, x_center, y_center, w, h))
        
        # Write to file
        with open(output_txt, 'w') as f:
            for class_id, x_center, y_center, w, h in yolo_annotations:
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
        
        return len(yolo_annotations)


class YOLOv5DatasetValidator:
    """Validate YOLOv5 format dataset"""
    
    def __init__(self, dataset_dir, classes_list):
        """
        Args:
            dataset_dir: Root directory containing images/ and labels/
            classes_list: List of class names
        """
        self.dataset_dir = Path(dataset_dir)
        self.classes_list = classes_list
        self.num_classes = len(classes_list)
    
    def validate(self, verbose=True):
        """
        Validate dataset structure and annotations
        
        Returns:
            dict with validation results
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        for split in ['train', 'val', 'test']:
            # Support both layouts: images/{split} & labels/{split} OR {split}/images & {split}/labels
            images_candidate_1 = self.dataset_dir / 'images' / split
            labels_candidate_1 = self.dataset_dir / 'labels' / split
            images_candidate_2 = self.dataset_dir / split / 'images'
            labels_candidate_2 = self.dataset_dir / split / 'labels'

            if images_candidate_2.exists() and labels_candidate_2.exists():
                images_dir = images_candidate_2
                labels_dir = labels_candidate_2
            elif images_candidate_1.exists() and labels_candidate_1.exists():
                images_dir = images_candidate_1
                labels_dir = labels_candidate_1
            else:
                results['warnings'].append(
                    f"Images/labels directories not found for split '{split}'. "
                    f"Checked: {images_candidate_1}, {labels_candidate_1}, {images_candidate_2}, {labels_candidate_2}"
                )
                continue
            
            # Get image and label files
            image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
            label_files = list(labels_dir.glob('*.txt'))
            
            split_stats = {
                'total_images': len(image_files),
                'total_labels': len(label_files),
                'total_annotations': 0,
                'class_distribution': {cls: 0 for cls in self.classes_list},
                'image_issues': [],
                'annotation_issues': []
            }
            
            if verbose:
                print(f"\n{'='*60}")
                print(f"Validating {split.upper()} set...")
                print(f"{'='*60}")
            
            pbar = tqdm(image_files, disable=not verbose)
            
            for image_file in pbar:
                pbar.set_description(f"Checking {image_file.name}")
                
                # Check if label exists
                label_file = labels_dir / (image_file.stem + '.txt')
                
                # Validate image
                try:
                    img = cv2.imread(str(image_file))
                    if img is None:
                        results['errors'].append(f"Cannot read image: {image_file}")
                        split_stats['image_issues'].append(str(image_file))
                        continue
                    
                    h, w = img.shape[:2]
                
                except Exception as e:
                    results['errors'].append(f"Error reading image {image_file}: {e}")
                    split_stats['image_issues'].append(str(image_file))
                    continue
                
                # Validate annotations
                if not label_file.exists():
                    results['warnings'].append(f"No label for image: {image_file}")
                    continue
                
                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                    
                    for line_num, line in enumerate(lines):
                        line = line.strip()
                        if not line:
                            continue
                        
                        parts = line.split()
                        
                        if len(parts) != 5:
                            results['errors'].append(
                                f"Invalid annotation in {label_file}:{line_num + 1} "
                                f"(expected 5 values, got {len(parts)})"
                            )
                            continue
                        
                        try:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            w_norm = float(parts[3])
                            h_norm = float(parts[4])
                        except ValueError as e:
                            results['errors'].append(
                                f"Invalid values in {label_file}:{line_num + 1}: {e}"
                            )
                            continue
                        
                        # Validate class ID
                        if not (0 <= class_id < self.num_classes):
                            results['errors'].append(
                                f"Invalid class ID {class_id} in {label_file}:{line_num + 1} "
                                f"(should be 0-{self.num_classes - 1})"
                            )
                            continue
                        
                        # Validate coordinates
                        if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
                            results['errors'].append(
                                f"Coordinates out of range in {label_file}:{line_num + 1} "
                                f"({x_center}, {y_center})"
                            )
                        
                        if not (0 < w_norm <= 1 and 0 < h_norm <= 1):
                            results['errors'].append(
                                f"Box size out of range in {label_file}:{line_num + 1} "
                                f"({w_norm}, {h_norm})"
                            )
                        
                        split_stats['total_annotations'] += 1
                        split_stats['class_distribution'][self.classes_list[class_id]] += 1
                
                except Exception as e:
                    results['errors'].append(f"Error reading annotations {label_file}: {e}")
                    split_stats['annotation_issues'].append(str(label_file))
            
            results['stats'][split] = split_stats
            
            if verbose:
                print(f"\nStatistics for {split} set:")
                print(f"  Images: {split_stats['total_images']}")
                print(f"  Labels: {split_stats['total_labels']}")
                print(f"  Total annotations: {split_stats['total_annotations']}")
                print(f"  Class distribution:")
                for cls, count in split_stats['class_distribution'].items():
                    print(f"    {cls}: {count}")
        
        results['valid'] = len(results['errors']) == 0
        
        if verbose:
            print(f"\n{'='*60}")
            print("Validation Summary:")
            print(f"{'='*60}")
            print(f"Valid: {results['valid']}")
            print(f"Errors: {len(results['errors'])}")
            print(f"Warnings: {len(results['warnings'])}")
            
            if results['errors']:
                print(f"\nFirst 5 errors:")
                for error in results['errors'][:5]:
                    print(f"  - {error}")
            
            if results['warnings']:
                print(f"\nWarnings:")
                for warning in results['warnings'][:5]:
                    print(f"  - {warning}")
        
        return results


def create_dataset_yaml(dataset_dir, classes_list, output_path='data.yaml'):
    """
    Create YOLOv5 format data.yaml file
    
    Args:
        dataset_dir: Root directory of dataset
        classes_list: List of class names
        output_path: Path to save yaml file
    """
    # Detect which layout the dataset uses and write appropriate relative paths
    dataset_path = Path(dataset_dir)
    if (dataset_path / 'images' / 'train').exists():
        train_path = 'images/train'
        val_path = 'images/val'
        test_path = 'images/test'
    elif (dataset_path / 'train' / 'images').exists():
        train_path = 'train/images'
        val_path = 'val/images'
        test_path = 'test/images'
    else:
        # Default to YOLOv5 style
        train_path = 'images/train'
        val_path = 'images/val'
        test_path = 'images/test'

    yaml_content = (
        f"path: {dataset_dir}\n"
        f"train: {train_path}\n"
        f"val: {val_path}\n"
        f"test: {test_path}\n\n"
        f"nc: {len(classes_list)}\n"
        f"names: {classes_list}\n"
    )
    
    with open(output_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"Created {output_path}")


if __name__ == '__main__':
    print("YOLOv5 data utilities")
