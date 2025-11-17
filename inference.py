import torch
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

from config import Config
from models.yolo_lite import YOLOLite
from data.dataset import DroneDataset
from utils.visualization import Visualizer
from utils.metrics import non_max_suppression, evaluate_detection

class DroneDetector:
    """Inference class cho drone detection"""
    
    def __init__(self, config, checkpoint_path):
        self.config = config
        self.device = config.DEVICE
        
        # Load model
        self.model = YOLOLite(config).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Loaded model from {checkpoint_path}")
        print(f"Device: {self.device}")
        
        self.visualizer = Visualizer(config)
    
    def decode_predictions(self, output):
        """
        Decode YOLO output thành detections
        
        Args:
            output: (batch, S, S, B*5+C) tensor
            
        Returns:
            List of detections per image
        """
        batch_size = output.size(0)
        S = self.config.GRID_SIZE
        B = self.config.NUM_BOXES
        C = self.config.NUM_CLASSES
        
        all_detections = []
        
        for b in range(batch_size):
            detections = []
            
            for i in range(S):
                for j in range(S):
                    # Lấy predictions cho grid cell (i, j)
                    cell_pred = output[b, i, j, :]
                    
                    # Decode từng bounding box
                    for box_idx in range(B):
                        start = box_idx * 5
                        confidence = cell_pred[start].item()
                        
                        # Threshold
                        if confidence < self.config.CONF_THRESHOLD:
                            continue
                        
                        # Bounding box
                        x_cell = cell_pred[start + 1].item()
                        y_cell = cell_pred[start + 2].item()
                        w = cell_pred[start + 3].item()
                        h = cell_pred[start + 4].item()
                        
                        # Convert to absolute coordinates
                        x_center = (j + x_cell) / S
                        y_center = (i + y_cell) / S
                        
                        # Class probabilities
                        class_probs = cell_pred[B * 5:]
                        class_id = torch.argmax(class_probs).item()
                        class_prob = class_probs[class_id].item()
                        
                        # Final confidence
                        final_conf = confidence * class_prob
                        
                        if final_conf > self.config.CONF_THRESHOLD:
                            detections.append({
                                'bbox': [x_center, y_center, w, h],
                                'confidence': final_conf,
                                'class': class_id,
                                'class_name': self.config.CLASSES[class_id]
                            })
            
            # Apply NMS
            detections = non_max_suppression(
                detections,
                iou_threshold=self.config.NMS_THRESHOLD
            )
            
            all_detections.append(detections)
        
        return all_detections
    
    def predict(self, spectrogram):
        """
        Predict trên một spectrogram
        
        Args:
            spectrogram: (H, W) numpy array or (1, H, W) tensor
            
        Returns:
            detections: List of detected objects
        """
        self.model.eval()
        
        with torch.no_grad():
            # Prepare input
            if isinstance(spectrogram, np.ndarray):
                if spectrogram.ndim == 2:
                    spectrogram = torch.FloatTensor(spectrogram).unsqueeze(0).unsqueeze(0)
                else:
                    spectrogram = torch.FloatTensor(spectrogram).unsqueeze(0)
            
            spectrogram = spectrogram.to(self.device)
            
            # Forward
            output = self.model(spectrogram)
            
            # Decode
            detections = self.decode_predictions(output)
        
        return detections[0]
    
    def evaluate(self, dataset_split='test'):
        """
        Đánh giá model trên test set
        """
        print(f"\nEvaluating on {dataset_split} set...")
        
        # Load dataset
        dataset = DroneDataset(self.config, split=dataset_split)
        
        all_predictions = []
        all_ground_truths = []
        
        # Inference
        for idx in tqdm(range(len(dataset))):
            spectrogram, target = dataset[idx]
            
            # Predict
            detections = self.predict(spectrogram)
            all_predictions.append(detections)
            
            # Ground truth
            gt_detections = self.decode_ground_truth(target)
            all_ground_truths.append(gt_detections)
        
        # Compute metrics
        metrics = evaluate_detection(
            all_predictions,
            all_ground_truths,
            iou_threshold=self.config.IOU_THRESHOLD,
            num_classes=self.config.NUM_CLASSES
        )
        
        print("\n" + "="*60)
        print("Evaluation Results:")
        print("="*60)
        print(f"mAP@{self.config.IOU_THRESHOLD}: {metrics['mAP']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        
        print("\nPer-class AP:")
        for class_id, ap in enumerate(metrics['APs']):
            print(f"  {self.config.CLASSES[class_id]}: {ap:.4f}")
        
        return metrics
    
    def decode_ground_truth(self, target):
        """Decode ground truth target thành detections format"""
        S = self.config.GRID_SIZE
        B = self.config.NUM_BOXES
        C = self.config.NUM_CLASSES
        
        detections = []
        
        for i in range(S):
            for j in range(S):
                cell = target[i, j, :]
                
                # Check if object exists
                if cell[0].item() > 0:
                    x_cell = cell[1].item()
                    y_cell = cell[2].item()
                    w = cell[3].item()
                    h = cell[4].item()
                    
                    x_center = (j + x_cell) / S
                    y_center = (i + y_cell) / S
                    
                    class_probs = cell[B * 5:]
                    class_id = torch.argmax(class_probs).item()
                    
                    detections.append({
                        'bbox': [x_center, y_center, w, h],
                        'class': class_id,
                        'confidence': 1.0
                    })
        
        return detections
    
    def visualize_predictions(self, dataset_split='test', num_samples=10,
                            save_dir='results'):
        """
        Visualize predictions
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        dataset = DroneDataset(self.config, split=dataset_split)
        
        indices = np.random.choice(len(dataset), 
                                  size=min(num_samples, len(dataset)),
                                  replace=False)
        
        for idx in indices:
            spectrogram, target = dataset[idx]
            
            # Predict
            detections = self.predict(spectrogram)
            
            # Ground truth
            gt_detections = self.decode_ground_truth(target)
            
            # Visualize
            spec_np = spectrogram.squeeze().numpy()
            
            save_path = save_dir / f'sample_{idx}.png'
            self.visualizer.create_comparison_plot(
                spec_np,
                gt_detections,
                detections,
                save_path=save_path
            )
            
            print(f"Saved visualization to {save_path}")

def main():
    config = Config()
    
    # Load best model
    checkpoint_path = Path(config.CHECKPOINT_DIR) / 'best.pth'
    
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    detector = DroneDetector(config, checkpoint_path)
    
    # Evaluate
    metrics = detector.evaluate(dataset_split='test')
    
    # Visualize
    detector.visualize_predictions(
        dataset_split='test',
        num_samples=20,
        save_dir='results/visualizations'
    )

if __name__ == '__main__':
    main()
