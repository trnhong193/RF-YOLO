import torch
import torch.nn as nn

class YOLOLite(nn.Module):
    """
    YOLO-Lite architecture cho drone detection
    Theo paper architecture (Table II)
    """
    
    def __init__(self, config):
        super(YOLOLite, self).__init__()
        
        self.config = config
        self.grid_size = config.GRID_SIZE
        self.num_boxes = config.NUM_BOXES
        self.num_classes = config.NUM_CLASSES
        
        # Convolutional layers
        self.conv1 = self._conv_block(1, 16, 3, 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = self._conv_block(16, 32, 3, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = self._conv_block(32, 64, 3, 1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = self._conv_block(64, 128, 3, 1)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.conv5 = self._conv_block(128, 128, 3, 1)
        self.pool5 = nn.MaxPool2d(2, 2)
        
        self.conv6 = self._conv_block(128, 256, 3, 1)
        
        # 1x1 convolution
        self.conv7 = nn.Conv2d(256, 125, 1, 1)
        
        # Output = S x S x (B * 5 + C)
        # 16 x 16 x 20 = 5120
        output_size = self.grid_size * self.grid_size * (
            self.num_boxes * 5 + self.num_classes
        )
        
        self.fc = nn.Linear(8 * 8 * 125, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def _conv_block(self, in_channels, out_channels, kernel_size, stride):
        """Convolution block with LeakyReLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, 
                     stride, padding=kernel_size//2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
    
    def forward(self, x):
        # Input: (B, 1, 256, 256)
        
        x = self.conv1(x)   # (B, 16, 256, 256)
        x = self.pool1(x)   # (B, 16, 128, 128)
        
        x = self.conv2(x)   # (B, 32, 128, 128)
        x = self.pool2(x)   # (B, 32, 64, 64)
        
        x = self.conv3(x)   # (B, 64, 64, 64)
        x = self.pool3(x)   # (B, 64, 32, 32)
        
        x = self.conv4(x)   # (B, 128, 32, 32)
        x = self.pool4(x)   # (B, 128, 16, 16)
        
        x = self.conv5(x)   # (B, 128, 16, 16)
        x = self.pool5(x)   # (B, 128, 8, 8)
        
        x = self.conv6(x)   # (B, 256, 8, 8)
        x = self.conv7(x)   # (B, 125, 8, 8)
        
        # Flatten
        x = x.view(x.size(0), -1)  # (B, 8*8*125)
        
        x = self.fc(x)      # (B, S*S*(B*5+C))
        x = self.sigmoid(x)
        
        # Reshape to (B, S, S, B*5+C)
        B = x.size(0)
        S = self.grid_size
        output_depth = self.num_boxes * 5 + self.num_classes
        
        x = x.view(B, S, S, output_depth)
        
        return x

def test_model():
    """Test model architecture"""
    from config import Config
    config = Config()
    
    model = YOLOLite(config)
    print(model)
    
    # Test forward pass
    x = torch.randn(4, 1, 256, 256)
    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: (4, 16, 16, 20)")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

if __name__ == '__main__':
    test_model()