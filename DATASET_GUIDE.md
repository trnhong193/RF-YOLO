# RF-YOLO Dataset Guide

## Cấu trúc Dataset YOLOv5

Dataset của bạn cần tuân theo cấu trúc YOLOv5 tiêu chuẩn:

```
data/
├── images/
│   ├── train/      # Training images
│   ├── val/        # Validation images
│   └── test/       # Test images
└── labels/
    ├── train/      # Training labels (txt files)
    ├── val/        # Validation labels (txt files)
    └── test/       # Test labels (txt files)
```

## Format Annotation

Mỗi file label (`.txt`) tương ứng với một image, chứa các dòng với format:

```
<class_id> <x_center> <y_center> <width> <height>
```

Trong đó:
- `class_id`: ID của class (0 to NUM_CLASSES-1)
- `x_center`: Tọa độ x của center bbox (normalized 0-1)
- `y_center`: Tọa độ y của center bbox (normalized 0-1)
- `width`: Chiều rộng bbox (normalized 0-1)
- `height`: Chiều cao bbox (normalized 0-1)

**Ví dụ:**
```
0 0.5 0.5 0.3 0.4
2 0.8 0.2 0.2 0.3
```

## Cách Sử Dụng

### 1. Tạo Cấu Trúc Dataset

Tạo cấu trúc thư mục YOLOv5:

```bash
python scripts/convert_dataset.py create-structure --dataset-dir ./data
```

### 2. Convert từ PASCAL VOC Format

Nếu bạn có annotations ở format PASCAL VOC (XML):

```bash
python scripts/convert_dataset.py pascal-voc \
    --xml-dir /path/to/xml/files \
    --image-dir /path/to/images \
    --output-dir ./data/labels/train \
    --classes class1 class2 class3
```

### 3. Convert từ COCO Format

Nếu bạn có annotations ở format COCO (JSON):

```bash
python scripts/convert_dataset.py coco \
    --coco-json /path/to/coco.json \
    --image-dir /path/to/images \
    --output-dir ./data/labels/train
```

### 4. Validate Dataset

Kiểm tra dataset của bạn có hợp lệ không:

```bash
python scripts/prepare_dataset.py \
    --dataset-dir ./data \
    --verbose
```

Nếu thành công, sẽ tạo file `data.yaml`:

```yaml
path: /path/to/data
train: images/train
val: images/val
test: images/test

nc: 10
names: ['DJI_Phantom', 'DJI_Mavic', 'DJI_Inspire', 'Parrot', 'Autel', 'Custom_Drone', 'WiFi', 'Bluetooth', 'Noise', 'Background']
```

### 5. Training

Sau khi dataset được validate:

```bash
python train.py
```

### 6. Inference

Đánh giá model trên test set:

```bash
python inference.py
```

## Classes Mặc Định

```python
CLASSES = [
    'DJI_Phantom',    # 0
    'DJI_Mavic',      # 1
    'DJI_Inspire',    # 2
    'Parrot',         # 3
    'Autel',          # 4
    'Custom_Drone',   # 5
    'WiFi',           # 6
    'Bluetooth',      # 7
    'Noise',          # 8
    'Background'      # 9
]
```

Có thể thay đổi trong `config.py`

## Data Augmentation

Data augmentation (lưu ý: ảnh là spectrogram nên không áp dụng flip):
- Random brightness adjustment (approx 0.85x - 1.15x)
- Random Gaussian noise (sigma ~ 0.01)

## Utility Functions

### Dataset Class

```python
from data.dataset import DroneDataset
from config import Config

config = Config()
dataset = DroneDataset(config, split='train')

# Get sample
spectrogram, target = dataset[0]

# Get image with annotations for visualization
image, annotations = dataset.get_image_with_annotations(0)

# Convert coordinates
from data.dataset import DroneDataset
xyxy_bbox = DroneDataset.convert_yolo_to_xyxy(annotation, image_size=(256, 256))
```

### Validation

```python
from data.data_utils import YOLOv5DatasetValidator
from config import Config

config = Config()
validator = YOLOv5DatasetValidator('./data', config.CLASSES)
results = validator.validate(verbose=True)
```

## Lỗi Phổ Biến

### "Images directory not found"
- Đảm bảo folder `images/train`, `images/val`, `images/test` tồn tại

### "Labels directory not found"
- Đảm bảo folder `labels/train`, `labels/val`, `labels/test` tồn tại

### "Invalid class ID"
- Kiểm tra class_id trong label files có nằm trong phạm vi 0 đến NUM_CLASSES-1

### "Coordinates out of range"
- Đảm bảo x_center, y_center, width, height được normalized (0-1)

### "Cannot read image"
- Kiểm tra file image có hợp lệ (.jpg hoặc .png)

## Tips

1. **Chia train/val/test**: Nên chia khoảng 70% train, 15% val, 15% test
2. **Balance dataset**: Cố gắng có tỷ lệ balanced giữa các classes
3. **Kiểm tra quality**: Dùng `prepare_dataset.py --verbose` để kiểm tra chi tiết
4. **Input size**: Default là 256x256, có thể thay đổi trong `config.py`

## Structures

### Spectrogram Input

- Size: 256 x 256 (grayscale)
- Normalized to [0, 1]
- Input channel: 1 (mono)

### YOLO Target Format

- Grid size: 16 x 16
- Boxes per cell: 2
- Classes: 10

Target shape: `(16, 16, 20)` = `(16, 16, 2*5 + 10)`

Mỗi grid cell chứa:
- Box 1: [obj_conf, x_offset, y_offset, width, height]
- Box 2: [obj_conf, x_offset, y_offset, width, height]
- Class probabilities: [10 values]

Tổng: 2*5 + 10 = 20 values per cell

## Contact & Support

Nếu gặp vấn đề, kiểm tra:
1. Logs: `./logs/`
2. Checkpoints: `./checkpoints/`
3. Visualization: `./results/visualizations/`
