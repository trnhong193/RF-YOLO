import torch
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from torch.utils.data import Dataset
import random
from tqdm import tqdm


class DroneDataset(Dataset):
    """
    YOLOv5 format dataset loader cho drone detection
    
    Cấu trúc dữ liệu YOLOv5:
    dataset/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
    
    Mỗi file label là text file với format:
    <class_id> <x_center> <y_center> <width> <height>
    (tọa độ được normalize từ 0-1)
    """
    
    def __init__(self, config, split='train', transforms=None):
        """
        Args:
            config: Config object
            split: 'train', 'val', hoặc 'test'
            transforms: Data augmentation transforms
        """
        self.config = config
        self.split = split
        self.transforms = transforms
        
        # Dataset paths (support both layouts)
        # 1) YOLOv5 style: <data_dir>/images/{split} and <data_dir>/labels/{split}
        # 2) split-first style: <data_dir>/{split}/images and <data_dir>/{split}/labels
        self.data_dir = Path(config.DATA_DIR)
        images_candidate_1 = self.data_dir / 'images' / split
        labels_candidate_1 = self.data_dir / 'labels' / split
        images_candidate_2 = self.data_dir / split / 'images'
        labels_candidate_2 = self.data_dir / split / 'labels'

        if images_candidate_2.exists() and labels_candidate_2.exists():
            self.images_dir = images_candidate_2
            self.labels_dir = labels_candidate_2
        elif images_candidate_1.exists() and labels_candidate_1.exists():
            self.images_dir = images_candidate_1
            self.labels_dir = labels_candidate_1
        else:
            # Raise helpful error listing checked locations
            raise FileNotFoundError(
                f"Could not find dataset folders for split '{split}'.\n"
                f"Checked: {images_candidate_1}, {labels_candidate_1}, {images_candidate_2}, {labels_candidate_2}"
            )
        
        # Get image files (support .jpg, .jpeg, .png)
        self.image_files = sorted(
            list(self.images_dir.glob('*.jpg')) +
            list(self.images_dir.glob('*.jpeg')) +
            list(self.images_dir.glob('*.png'))
        )
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.images_dir}")
        
        print(f"Loaded {len(self.image_files)} images from {split} set")
        
        # Get supported classes from config
        self.classes = config.CLASSES
        self.num_classes = config.NUM_CLASSES
        self.grid_size = config.GRID_SIZE
        self.num_boxes = config.NUM_BOXES
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        Returns:
            spectrogram: (1, H, W) tensor
            target: (S, S, B*5+C) tensor - YOLO format target
        """
        image_file = self.image_files[idx]
        label_file = self.labels_dir / (image_file.stem + '.txt')
        
        # Load image
        image = self._load_image(image_file)
        
        # Load annotations
        annotations = self._load_annotations(label_file)
        
        # Create YOLO format target
        target = self._create_target(image.shape, annotations)
        
        # Convert to tensor
        spectrogram = torch.FloatTensor(image).unsqueeze(0)  # (1, H, W)
        target_tensor = torch.FloatTensor(target)  # (S, S, B*5+C)
        
        # Apply transforms if available
        if self.transforms:
            spectrogram = self.transforms(spectrogram)
        
        return spectrogram, target_tensor
    
    def _load_image(self, image_path):
        """Load image và normalize"""
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Resize to input size
        image = cv2.resize(image, self.config.INPUT_SIZE, 
                          interpolation=cv2.INTER_LINEAR)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def _load_annotations(self, label_file):
        """
        Load YOLO format annotations
        
        Format: <class_id> <x_center> <y_center> <width> <height>
        (normalized coordinates 0-1)
        
        Returns:
            List of annotations [(class_id, x, y, w, h), ...]
        """
        annotations = []
        
        if not label_file.exists():
            return annotations
        
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Validate values
                    if not (0 <= class_id < self.num_classes):
                        continue
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
                        continue
                    if not (0 < width <= 1 and 0 < height <= 1):
                        continue
                    
                    annotations.append({
                        'class_id': class_id,
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height
                    })
        
        except Exception as e:
            print(f"Error loading annotations from {label_file}: {e}")
        
        return annotations
    
    def _create_target(self, image_shape, annotations):
        """
        Create YOLO format target tensor
        
        Args:
            image_shape: (H, W)
            annotations: List of annotation dicts
            
        Returns:
            target: (S, S, B*5+C) array
            Format: [obj_confidence, x, y, w, h, class_probs...]
        """
        S = self.grid_size
        B = self.num_boxes
        C = self.num_classes
        
        # Initialize target
        target = np.zeros((S, S, B * 5 + C), dtype=np.float32)
        
        for ann in annotations:
            class_id = ann['class_id']
            x_center = ann['x_center']
            y_center = ann['y_center']
            width = ann['width']
            height = ann['height']
            
            # Convert normalized coordinates to grid cell
            x_grid = x_center * S
            y_grid = y_center * S
            
            # Grid cell indices
            i = int(y_grid)
            j = int(x_grid)
            
            # Clamp to grid
            i = np.clip(i, 0, S - 1)
            j = np.clip(j, 0, S - 1)
            
            # Offset trong grid cell (0-1)
            x_offset = x_grid - j
            y_offset = y_grid - i
            
            # Check if cell is empty (no object assigned yet)
            cell = target[i, j, :]
            has_object = cell[0] > 0
            
            if not has_object:
                # Assign to first box
                box_idx = 0
            else:
                # Assign to second box if available
                if B > 1:
                    box_idx = 1
                else:
                    # If cell already has object and only 1 box per cell,
                    # skip this annotation
                    continue
            
            # Write to target
            start_idx = box_idx * 5
            target[i, j, start_idx] = 1.0  # Object confidence
            target[i, j, start_idx + 1] = x_offset  # x_offset in cell
            target[i, j, start_idx + 2] = y_offset  # y_offset in cell
            target[i, j, start_idx + 3] = width   # width (normalized)
            target[i, j, start_idx + 4] = height  # height (normalized)
            
            # Class probabilities
            class_start = B * 5
            target[i, j, class_start + class_id] = 1.0
        
        return target
    
    def get_image_with_annotations(self, idx):
        """
        Get image và annotations cho visualization
        
        Returns:
            image: (H, W) numpy array
            annotations: List of annotation dicts
        """
        image_file = self.image_files[idx]
        label_file = self.labels_dir / (image_file.stem + '.txt')
        
        image = self._load_image(image_file)
        annotations = self._load_annotations(label_file)
        
        return image, annotations
    
    @staticmethod
    def convert_yolo_to_xyxy(annotation, image_size):
        """
        Convert YOLO format to xyxy format
        
        Args:
            annotation: dict with 'x_center', 'y_center', 'width', 'height' (normalized)
            image_size: (H, W)
            
        Returns:
            (x1, y1, x2, y2) in pixel coordinates
        """
        H, W = image_size
        x_center = annotation['x_center'] * W
        y_center = annotation['y_center'] * H
        width = annotation['width'] * W
        height = annotation['height'] * H
        
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)
        
        return (x1, y1, x2, y2)
    
    @staticmethod
    def convert_xyxy_to_yolo(bbox, image_size):
        """
        Convert xyxy format to YOLO format
        
        Args:
            bbox: (x1, y1, x2, y2) in pixel coordinates
            image_size: (H, W)
            
        Returns:
            dict with normalized YOLO format coordinates
        """
        H, W = image_size
        x1, y1, x2, y2 = bbox
        
        x_center = (x1 + x2) / 2 / W
        y_center = (y1 + y2) / 2 / H
        width = (x2 - x1) / W
        height = (y2 - y1) / H
        
        return {
            'x_center': np.clip(x_center, 0, 1),
            'y_center': np.clip(y_center, 0, 1),
            'width': np.clip(width, 0, 1),
            'height': np.clip(height, 0, 1)
        }


class DroneDatasetWithAugmentation(DroneDataset):
    """Dataset với data augmentation"""
    
    def __init__(self, config, split='train', augment=True):
        super().__init__(config, split)
        self.augment = augment and split == 'train'
    
    def __getitem__(self, idx):
        spectrogram, target = super().__getitem__(idx)
        
        if self.augment:
            spectrogram, target = self._augment(spectrogram, target)
        
        return spectrogram, target
    
    def _augment(self, spectrogram, target):
        """Apply augmentation"""
        # Convert to numpy
        spec_np = spectrogram.squeeze().numpy()
        # Note: flips are intentionally avoided for spectrograms (they are not
        # spatial natural images). Only apply photometric/noise augmentations.

        # Random brightness adjustment
        if random.random() < 0.5:
            brightness = random.uniform(0.85, 1.15)
            spec_np = np.clip(spec_np * brightness, 0, 1)

        # Random Gaussian noise
        if random.random() < 0.3:
            noise = np.random.normal(0, 0.01, spec_np.shape)
            spec_np = np.clip(spec_np + noise, 0, 1)
        
        spectrogram = torch.FloatTensor(spec_np).unsqueeze(0)
        
        return spectrogram, target


def get_dataset(config, split='train', augment=True):
    """Factory function để tạo dataset"""
    if augment:
        return DroneDatasetWithAugmentation(config, split=split, augment=True)
    else:
        return DroneDataset(config, split=split)


if __name__ == '__main__':
    from config import Config
    
    config = Config()
    
    # Test dataset
    print("Testing dataset loading...")
    
    for split in ['train', 'val', 'test']:
        try:
            dataset = DroneDataset(config, split=split)
            print(f"\n{split.upper()} SET:")
            print(f"  Samples: {len(dataset)}")
            
            # Load one sample
            spectrogram, target = dataset[0]
            print(f"  Spectrogram shape: {spectrogram.shape}")
            print(f"  Target shape: {target.shape}")
            print(f"  Expected target shape: ({config.GRID_SIZE}, {config.GRID_SIZE}, {config.NUM_BOXES * 5 + config.NUM_CLASSES})")
            
            # Check annotations
            image, annotations = dataset.get_image_with_annotations(0)
            print(f"  Annotations in first image: {len(annotations)}")
            if annotations:
                print(f"  First annotation: {annotations[0]}")
        
        except FileNotFoundError as e:
            print(f"\n{split.upper()} SET: {e}")
        except Exception as e:
            print(f"\n{split.upper()} SET ERROR: {e}")
