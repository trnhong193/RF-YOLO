"""
Demo đơn giản cho inference
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from config import Config
from inference import DroneDetector
from data.generate_data import DroneSignalGenerator
from utils.visualization import Visualizer

def demo_single_signal():
    """Demo với single signal"""
    
    print("="*60)
    print("DEMO: Single Signal Detection")
    print("="*60)
    
    # Load config
    config = Config()
    
    # Load detector
    checkpoint_path = 'checkpoints/best.pth'
    if not Path(checkpoint_path).exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please train the model first!")
        return
    
    detector = DroneDetector(config, checkpoint_path)
    visualizer = Visualizer(config)
    
    # Generate sample signal
    print("\n1. Generating sample drone signal...")
    generator = DroneSignalGenerator(config)
    spectrogram, annotations = generator.generate_sample(num_signals=1)
    
    print(f"   Generated spectrogram shape: {spectrogram.shape}")
    print(f"   Ground truth annotations: {len(annotations)} signals")
    
    for annot in annotations:
        print(f"     - {annot['class_name']}")
    
    # Predict
    print("\n2. Running detection...")
    detections = detector.predict(spectrogram)
    
    print(f"   Detected {len(detections)} signals:")
    for det in detections:
        print(f"     - {det['class_name']}: {det['confidence']:.3f}")
        bbox = det['bbox']
        print(f"       Position: ({bbox[0]:.3f}, {bbox[1]:.3f})")
        print(f"       Size: {bbox[2]:.3f} × {bbox[3]:.3f}")
    
    # Visualize
    print("\n3. Visualizing results...")
    visualizer.draw_boxes_on_spectrogram(
        spectrogram, detections,
        save_path='demo_single_signal.png',
        show=True
    )
    
    print("\n✓ Demo completed!")

def demo_multi_signal():
    """Demo với multiple signals"""
    
    print("\n" + "="*60)
    print("DEMO: Multi-Signal Detection")
    print("="*60)
    
    config = Config()
    detector = DroneDetector(config, 'checkpoints/best.pth')
    visualizer = Visualizer(config)
    generator = DroneSignalGenerator(config)
    
    # Generate với nhiều signals
    num_signals = 4
    print(f"\n1. Generating {num_signals} simultaneous signals...")
    
    spectrogram, annotations = generator.generate_sample(num_signals=num_signals)
    
    print(f"   Ground truth: {len(annotations)} signals")
    for annot in annotations:
        print(f"     - {annot['class_name']}")
    
    # Predict
    print("\n2. Running detection...")
    detections = detector.predict(spectrogram)
    
    print(f"   Detected {len(detections)} signals:")
    for det in detections:
        print(f"     - {det['class_name']}: {det['confidence']:.3f}")
    
    # Visualize comparison
    print("\n3. Visualizing comparison...")
    visualizer.create_comparison_plot(
        spectrogram,
        annotations,
        detections,
        save_path='demo_multi_signal.png'
    )
    
    print("\n✓ Demo completed!")

def demo_different_snr():
    """Demo với different SNR levels"""
    
    print("\n" + "="*60)
    print("DEMO: Detection at Different SNR Levels")
    print("="*60)
    
    config = Config()
    detector = DroneDetector(config, 'checkpoints/best.pth')
    generator = DroneSignalGenerator(config)
    
    snr_levels = [-10, -5, 0, 5, 10]
    
    fig, axes = plt.subplots(2, len(snr_levels), figsize=(20, 8))
    
    for i, snr in enumerate(snr_levels):
        print(f"\nTesting at SNR = {snr} dB...")
        
        # Generate signal
        spectrogram, annotations = generator.generate_sample(
            num_signals=2,
            snr_range=(snr, snr)
        )
        
        # Predict
        detections = detector.predict(spectrogram)
        
        print(f"  Ground truth: {len(annotations)} signals")
        print(f"  Detected: {len(detections)} signals")
        
        # Plot spectrogram
        axes[0, i].imshow(spectrogram, aspect='auto', origin='lower',
                         cmap='viridis', interpolation='nearest')
        axes[0, i].set_title(f'SNR = {snr} dB')
        axes[0, i].set_xlabel('Time')
        axes[0, i].set_ylabel('Frequency')
        
        # Plot with detections
        axes[1, i].imshow(spectrogram, aspect='auto', origin='lower',
                         cmap='viridis', interpolation='nearest')
        
        # Draw boxes
        H, W = spectrogram.shape
        for det in detections:
            bbox = det['bbox']
            x = bbox[0] * W
            y = bbox[1] * H
            w = bbox[2] * W
            h = bbox[3] * H
            
            rect = plt.Rectangle((x - w/2, y - h/2), w, h,
                                linewidth=2, edgecolor='red',
                                facecolor='none')
            axes[1, i].add_patch(rect)
        
        axes[1, i].set_title(f'Detected: {len(detections)}')
        axes[1, i].set_xlabel('Time')
        axes[1, i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('demo_snr_comparison.png', dpi=150)
    print("\n✓ Saved SNR comparison to demo_snr_comparison.png")
    plt.show()

def main():
    """Run all demos"""
    
    try:
        demo_single_signal()
        demo_multi_signal()
        demo_different_snr()
        
        print("\n" + "="*60)
        print("ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()