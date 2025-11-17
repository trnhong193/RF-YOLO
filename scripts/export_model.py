"""
Export model sang các format khác nhau (ONNX, TorchScript)
"""

import torch
import argparse
from pathlib import Path

from config import Config
from models.yolo_lite import YOLOLite

def export_onnx(model, output_path, input_size=(1, 1, 256, 256)):
    """Export model to ONNX format"""
    
    model.eval()
    dummy_input = torch.randn(*input_size).to(next(model.parameters()).device)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✓ Exported ONNX model to {output_path}")
    
    # Verify
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model verification passed")

def export_torchscript(model, output_path, input_size=(1, 1, 256, 256)):
    """Export model to TorchScript format"""
    
    model.eval()
    dummy_input = torch.randn(*input_size).to(next(model.parameters()).device)
    
    # Trace model
    traced_model = torch.jit.trace(model, dummy_input)
    
    # Save
    traced_model.save(output_path)
    
    print(f"✓ Exported TorchScript model to {output_path}")
    
    # Verify
    loaded_model = torch.jit.load(output_path)
    output1 = model(dummy_input)
    output2 = loaded_model(dummy_input)
    
    diff = torch.abs(output1 - output2).max().item()
    print(f"✓ TorchScript verification: max diff = {diff}")

def export_weights_only(model, output_path):
    """Export only model weights"""
    
    torch.save(model.state_dict(), output_path)
    print(f"✓ Exported weights to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint')
    parser.add_argument('--output', type=str, default='exported_model',
                       help='Output path (without extension)')
    parser.add_argument('--format', type=str, default='all',
                       choices=['onnx', 'torchscript', 'weights', 'all'],
                       help='Export format')
    
    args = parser.parse_args()
    
    # Load config and model
    config = Config()
    model = YOLOLite(config).to(config.DEVICE)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded checkpoint from {args.checkpoint}")
    print(f"Export format: {args.format}")
    
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export
    if args.format in ['onnx', 'all']:
        export_onnx(model, f"{args.output}.onnx")
    
    if args.format in ['torchscript', 'all']:
        export_torchscript(model, f"{args.output}.pt")
    
    if args.format in ['weights', 'all']:
        export_weights_only(model, f"{args.output}_weights.pth")
    
    print("\n✓ Export completed!")

if __name__ == '__main__':
    main()