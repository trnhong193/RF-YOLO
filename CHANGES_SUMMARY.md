# Tóm Tắt Các Thay Đổi - YOLOv5 Format Support

## 📁 Files Tạo Mới

### 1. **`data/dataset.py`** - Dataset Loader Chính
- `DroneDataset`: Class loader cho YOLOv5 format
  - Hỗ trợ 3 splits: train, val, test
  - Load ảnh từ `images/{split}/`
  - Load labels từ `labels/{split}/` (format YOLO)
  - Tự động convert sang tensor format (1, H, W) cho input
  - Output target shape: (S, S, B*5+C) = (16, 16, 20)
  
- `DroneDatasetWithAugmentation`: Dataset với data augmentation
- `DroneDatasetWithAugmentation`: Dataset với data augmentation
  - Lưu ý: không áp dụng flip (spectrograms không nên flip)
  - Random brightness adjustment
  - Random noise injection
  - Chỉ áp dụng cho training set

- Utility methods:
  - `convert_yolo_to_xyxy()`: Convert normalized YOLO coords → pixel xyxy
  - `convert_xyxy_to_yolo()`: Convert pixel xyxy → normalized YOLO coords

### 2. **`data/data_utils.py`** - Data Utilities
- `YOLOv5Converter`: Convert từ các format khác
  - `pascal_voc_to_yolo()`: Từ PASCAL VOC XML
  - `coco_to_yolo()`: Từ COCO JSON
  - `custom_to_yolo()`: Từ custom format

- `YOLOv5DatasetValidator`: Validate dataset
  - Kiểm tra structure thư mục
  - Validate từng annotation
  - Check invalid values
  - Thống kê chi tiết per split

- `create_dataset_yaml()`: Tạo data.yaml

### 3. **`scripts/prepare_dataset.py`** - Dataset Validation Script
```bash
python scripts/prepare_dataset.py --dataset-dir ./data --verbose
```
- Validate toàn bộ dataset
- Tạo `data.yaml`
- In ra statistics và warnings

### 4. **`scripts/convert_dataset.py`** - Dataset Conversion Script
```bash
# Tạo structure
python scripts/convert_dataset.py create-structure --dataset-dir ./data

# Convert từ PASCAL VOC
python scripts/convert_dataset.py pascal-voc --xml-dir ... --image-dir ... 

# Convert từ COCO
python scripts/convert_dataset.py coco --coco-json ... --image-dir ...
```

### 5. **`scripts/test_dataset.py`** - Dataset Testing Script
```bash
python scripts/test_dataset.py
```
- Test load dataset
- Visualize batch
- Kiểm tra shapes và values
- Tạo `sample_batch.png`

### 6. **`DATASET_GUIDE.md`** - Dataset Documentation
Hướng dẫn chi tiết:
- Cấu trúc dataset YOLOv5
- Format annotation
- Cách setup dataset
- Conversion từ các format khác
- Troubleshooting

## 🔧 Files Chỉnh Sửa

### 1. **`train.py`**
```python
# Thay đổi:
from data.dataset import DroneDataset, DroneDatasetWithAugmentation

def create_dataloader(self, split):
    if split == 'train':
        dataset = DroneDatasetWithAugmentation(...)
    else:
        dataset = DroneDataset(...)
```

### 2. **`config.py`**
```python
# Thay đổi:
DATA_DIR = './data'  # YOLOv5 format structure
DATASET_FORMAT = 'yolov5'
```

### 3. **`scripts/__init__.py`** (tạo file)
Sửa typo từ `__inint__.py` → `__init__.py`

### 4. **`utils/__init__.py`** (tạo file)
Sửa typo từ `__inti__.py` → `__init__.py`

## 📊 Dataset Structure

```
data/
├── images/
│   ├── train/      # Training images (e.g., 1000 ảnh)
│   ├── val/        # Validation images (e.g., 300 ảnh)
│   └── test/       # Test images (e.g., 200 ảnh)
└── labels/
    ├── train/      # Training labels (*.txt files)
    ├── val/        # Validation labels (*.txt files)
    └── test/       # Test labels (*.txt files)
```

## 🏷️ Annotation Format

Mỗi file label `image_name.txt`:
```
<class_id> <x_center> <y_center> <width> <height>
<class_id> <x_center> <y_center> <width> <height>
...
```

Ví dụ:
```
0 0.512 0.514 0.312 0.425
2 0.832 0.201 0.152 0.238
```

Tất cả tọa độ được **normalize** (0-1)

## 🚀 Cách Sử Dụng

### Setup Dataset (1 lần)
```bash
# Tạo structure
python scripts/convert_dataset.py create-structure --dataset-dir ./data

# Copy ảnh vào:
# - data/images/train/
# - data/images/val/
# - data/images/test/

# Convert labels hoặc copy labels vào:
# - data/labels/train/
# - data/labels/val/
# - data/labels/test/

# Validate
python scripts/prepare_dataset.py --dataset-dir ./data --verbose
```

### Training
```bash
python train.py
```
- Tự động load từ `data/images/train` và `data/labels/train`
- Áp dụng augmentation
- Validation trên `data/images/val` và `data/labels/val`

### Inference
```bash
python inference.py
```
- Đánh giá trên `data/images/test` và `data/labels/test`

### Test Dataset Loader
```bash
python scripts/test_dataset.py
```

## 📝 Classes Mặc Định (trong config.py)

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

## 🔄 Conversion từ Các Format Khác

### Từ PASCAL VOC
```bash
python scripts/convert_dataset.py pascal-voc \
    --xml-dir ./annotations/xml \
    --image-dir ./images \
    --output-dir ./data/labels/train
```

### Từ COCO
```bash
python scripts/convert_dataset.py coco \
    --coco-json ./annotations/instances.json \
    --image-dir ./images \
    --output-dir ./data/labels/train
```

## ✅ Validation Checklist

Trước khi train, chạy:
```bash
python scripts/prepare_dataset.py --dataset-dir ./data --verbose
```

Kiểm tra:
- ✓ Thư mục `images/{train,val,test}` tồn tại
- ✓ Thư mục `labels/{train,val,test}` tồn tại
- ✓ Số file ảnh = số file label trong mỗi split
- ✓ Class IDs nằm trong [0, NUM_CLASSES)
- ✓ Coordinates normalized [0, 1]
- ✓ Không có format errors

## 🐛 Troubleshooting

### Error: "Images directory not found"
- Kiểm tra `data/images/train`, `data/images/val`, `data/images/test` tồn tại

### Error: "Invalid class ID"
- Kiểm tra class IDs trong labels < NUM_CLASSES (mặc định 10)
- Hoặc update NUM_CLASSES trong config.py

### Error: "Coordinates out of range"
- Đảm bảo x_center, y_center ∈ [0, 1]
- Đảm bảo width, height ∈ (0, 1]

## 📚 Dependencies

Thêm vào `requirements.txt` (nếu chưa có):
- opencv-python (cv2) - for image I/O
- tqdm - for progress bars
- numpy - for arrays
- torch - for tensors

Đã có trong project hiện tại.

## 🎯 Next Steps

1. Prepare dataset theo cấu trúc YOLOv5
2. Chạy validation: `python scripts/prepare_dataset.py --verbose`
3. Test dataset loader: `python scripts/test_dataset.py`
4. Start training: `python train.py`
5. Evaluate: `python inference.py`
