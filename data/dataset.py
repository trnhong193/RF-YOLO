import torch
from torch.utils.data import Dataset
import h5py
import json
import numpy as np
from pathlib import Path

class DroneDataset(Dataset):
    """Dataset cho YOLO drone detection"""
    
    def __init__(self, config, split='train', transform=None):
        self.config = config
        self.split = split
        self.transform = transform
        
        data_dir = Path(config.DATA_DIR)
        
        # Load spectrograms
        h5_path = data_dir / f'{split}.h5'
        self.h5_file = h5py.File(h5_path, 'r')
        self.spectrograms = self.h5_file['spectrograms']
        
        # Load annotations
        json_path = data_dir / f'{split}_annotations.json'
        with open(json_path, 'r') as f:
            self.annotations = json.load(f)
        
        print(f"Loaded {len(self.spectrograms)} {split} samples")
    
    def __len__(self):
        return len(self.spectrograms)
    
    def __getitem__(self, idx):
        # Load spectrogram
        spectrogram = self.spectrograms[idx]
        spectrogram = torch.FloatTensor(spectrogram).unsqueeze(0)  # Add channel
        
        # Load annotations
        annotations = self.annotations[idx]
        
        # Convert to YOLO format: grid cell assignments
        target = self.encode_target(annotations)
        
        if self.transform:
            spectrogram = self.transform(spectrogram)
        
        return spectrogram, target
    
    def encode_target(self, annotations):
        """
        Encode annotations thành YOLO target format
        
        Target shape: (S, S, B * 5 + C)
        - S = GRID_SIZE (16)
        - B = NUM_BOXES (2)
        - 5 = [confidence, x, y, w, h]
        - C = NUM_CLASSES (10)
        """
        S = self.config.GRID_SIZE
        B = self.config.NUM_BOXES
        C = self.config.NUM_CLASSES
        
        target = torch.zeros(S, S, B * 5 + C)
        
        for annot in annotations:
            class_idx = annot['class']
            x, y, w, h = annot['bbox']
            
            # Xác định grid cell chứa object
            grid_x = int(x * S)
            grid_y = int(y * S)
            
            # Clip to valid range
            grid_x = min(grid_x, S - 1)
            grid_y = min(grid_y, S - 1)
            
            # Offset trong grid cell (0 to 1)
            x_cell = x * S - grid_x
            y_cell = y * S - grid_y
            
            # Gán vào box đầu tiên của grid cell
            # [conf, x, y, w, h] cho box 1
            target[grid_y, grid_x, 0] = 1.0  # Confidence
            target[grid_y, grid_x, 1] = x_cell
            target[grid_y, grid_x, 2] = y_cell
            target[grid_y, grid_x, 3] = w
            target[grid_y, grid_x, 4] = h
            
            # Class probabilities
            target[grid_y, grid_x, B * 5 + class_idx] = 1.0
        
        return target
    
    def __del__(self):
        if hasattr(self, 'h5_file'):
            self.h5_file.close()