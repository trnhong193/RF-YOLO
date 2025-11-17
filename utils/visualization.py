import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2

class Visualizer:
    """Visualization tools cho detection results"""
    
    def __init__(self, config):
        self.config = config
        self.class_names = config.CLASSES
        self.colors = self._generate_colors(len(self.class_names))
    
    def _generate_colors(self, num_classes):
        """Generate distinct colors cho mỗi class"""
        np.random.seed(42)
        colors = []
        for i in range(num_classes):
            colors.append(tuple(np.random.rand(3)))
        return colors
    
    def draw_boxes_on_spectrogram(self, spectrogram, detections, 
                                   save_path=None, show=True):
        """
        Vẽ bounding boxes lên spectrogram
        
        Args:
            spectrogram: (H, W) numpy array
            detections: List of dicts with keys: 'bbox', 'class', 'confidence'
            save_path: Path to save figure
            show: Whether to show figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Plot spectrogram
        im = ax.imshow(spectrogram, aspect='auto', origin='lower',
                      cmap='viridis', interpolation='nearest')
        
        H, W = spectrogram.shape
        
        # Draw each detection
        for det in detections:
            bbox = det['bbox']  # [x_center, y_center, width, height]
            class_id = det['class']
            confidence = det['confidence']
            
            # Convert from normalized to pixel coordinates
            x_center = bbox[0] * W
            y_center = bbox[1] * H
            width = bbox[2] * W
            height = bbox[3] * H
            
            # Convert to corner coordinates
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            
            # Create rectangle
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2,
                edgecolor=self.colors[class_id],
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            label = f"{self.class_names[class_id]}: {confidence:.2f}"
            ax.text(x1, y1 - 5, label,
                   bbox=dict(boxstyle='round', facecolor=self.colors[class_id], 
                            alpha=0.7),
                   fontsize=10, color='white', weight='bold')
        
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Frequency Bins')
        ax.set_title('Drone Detection Results')
        plt.colorbar(im, ax=ax, label='Power (normalized)')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_training_history(self, history, save_path=None):
        """
        Vẽ training history
        
        Args:
            history: Dict with keys: 'train_loss', 'val_loss', etc.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total loss
        axes[0, 0].plot(history['train_loss'], label='Train')
        if 'val_loss' in history:
            axes[0, 0].plot(history['val_loss'], label='Validation')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Total Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Localization loss
        if 'train_loc_loss' in history:
            axes[0, 1].plot(history['train_loc_loss'], label='Train')
            if 'val_loc_loss' in history:
                axes[0, 1].plot(history['val_loc_loss'], label='Validation')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Localization Loss')
            axes[0, 1].set_title('Localization Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Confidence loss
        if 'train_conf_loss' in history:
            axes[1, 0].plot(history['train_conf_loss'], label='Train')
            if 'val_conf_loss' in history:
                axes[1, 0].plot(history['val_conf_loss'], label='Validation')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Confidence Loss')
            axes[1, 0].set_title('Confidence Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Classification loss
        if 'train_class_loss' in history:
            axes[1, 1].plot(history['train_class_loss'], label='Train')
            if 'val_class_loss' in history:
                axes[1, 1].plot(history['val_class_loss'], label='Validation')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Classification Loss')
            axes[1, 1].set_title('Classification Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved training history to {save_path}")
        
        plt.show()
    
    def create_comparison_plot(self, spectrogram, ground_truth, predictions,
                              save_path=None):
        """
        So sánh ground truth vs predictions
        """
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Ground truth
        axes[0].imshow(spectrogram, aspect='auto', origin='lower',
                      cmap='viridis', interpolation='nearest')
        axes[0].set_title('Ground Truth', fontsize=14, weight='bold')
        self._draw_boxes_on_ax(axes[0], ground_truth, spectrogram.shape)
        
        # Predictions
        axes[1].imshow(spectrogram, aspect='auto', origin='lower',
                      cmap='viridis', interpolation='nearest')
        axes[1].set_title('Predictions', fontsize=14, weight='bold')
        self._draw_boxes_on_ax(axes[1], predictions, spectrogram.shape)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def _draw_boxes_on_ax(self, ax, detections, shape):
        """Helper để vẽ boxes lên axis"""
        H, W = shape
        
        for det in detections:
            bbox = det['bbox']
            class_id = det['class']
            
            x_center = bbox[0] * W
            y_center = bbox[1] * H
            width = bbox[2] * W
            height = bbox[3] * H
            
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2,
                edgecolor=self.colors[class_id],
                facecolor='none'
            )
            ax.add_patch(rect)
            
            label = self.class_names[class_id]
            if 'confidence' in det:
                label += f": {det['confidence']:.2f}"
            
            ax.text(x1, y1 - 5, label,
                   bbox=dict(boxstyle='round', 
                            facecolor=self.colors[class_id], alpha=0.7),
                   fontsize=9, color='white')