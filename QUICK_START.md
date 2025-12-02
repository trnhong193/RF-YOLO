# RF-YOLO Quick Start Guide - YOLOv5 Dataset

## 🚀 Quick Start (5 Bước)

### Bước 1: Chuẩn Bị Dataset Structure

```bash
# Tạo thư mục structure
python scripts/convert_dataset.py create-structure --dataset-dir ./data
```

Sau đó tạo thư mục như sau:
```
data/
├── images/
│   ├── train/   ← Copy ảnh training ở đây
│   ├── val/     ← Copy ảnh validation ở đây
│   └── test/    ← Copy ảnh test ở đây
└── labels/
    ├── train/   ← Copy labels training ở đây
    ├── val/     ← Copy labels validation ở đây
    └── test/    ← Copy labels test ở đây
```

### Bước 2: Format Labels (YOLOv5)

Mỗi file label phải có tên giống như ảnh (nhưng `.txt` thay vì `.jpg/.png`)

**Nội dung file label:**
```
<class_id> <x_center> <y_center> <width> <height>
```

**Ví dụ:** `image123.txt`
```
0 0.512 0.514 0.312 0.425
2 0.832 0.201 0.152 0.238
```

- Các tọa độ được **normalize** từ 0 đến 1
- Mỗi hàng là một object
- `class_id` từ 0 đến 9

### Bước 3: Convert Label (Nếu Cần)

Nếu bạn có labels ở format khác:

```bash
# Từ PASCAL VOC XML
python scripts/convert_dataset.py pascal-voc \
    --xml-dir ./annotations/xml \
    --image-dir ./images \
    --output-dir ./data/labels/train

# Từ COCO JSON
python scripts/convert_dataset.py coco \
    --coco-json ./annotations/coco.json \
    --image-dir ./images \
    --output-dir ./data/labels/train
```

### Bước 4: Validate Dataset

```bash
python scripts/prepare_dataset.py --dataset-dir ./data --verbose
```

Output:
```
✓ Dataset validation PASSED
✓ Data yaml created at: data.yaml

TRAINING SET:
  Images: 1000
  Annotations: 2500
  Class distribution:
    - DJI_Phantom: 450
    - DJI_Mavic: 520
    - ...

VALIDATION SET:
  Images: 300
  Annotations: 750

TEST SET:
  Images: 200
  Annotations: 500

TOTAL:
  Images: 1500
  Annotations: 3750
```

### Bước 5: Train Model

```bash
python train.py
```

Hoặc chạy resume từ checkpoint:
```bash
python train.py --resume ./checkpoints/last.pth
```

## 🔍 Troubleshooting

### Error: "Images directory not found"
```bash
# Đảm bảo cấu trúc này tồn tại
ls -la data/images/train/    # Nên có ảnh ở đây
ls -la data/labels/train/    # Nên có .txt files ở đây
```

### Error: "Invalid class ID"
```bash
# Kiểm tra class IDs trong labels
grep -oP '^\K[0-9]' data/labels/train/*.txt | sort | uniq

# Output nên từ 0-9 (tương ứng 10 classes)
```

### Error: "Coordinates out of range"
```bash
# Kiểm tra format labels
head data/labels/train/sample.txt
# Nên có format: <class_id> <x> <y> <w> <h>
# Các giá trị từ 0-1
```

### Tậu các class distributions không balanced
```bash
# Kiểm tra thống kê
python scripts/prepare_dataset.py --dataset-dir ./data --verbose | grep -A 20 "Class distribution"

# Có thể cần data augmentation hoặc resampling
```

## 📊 Monitoring Training

Sau khi training bắt đầu:

```bash
# Xem logs
tail -f logs/history.json

# Xem checkpoints
ls -lh checkpoints/
# best.pth  - Best model (theo validation loss)
# last.pth  - Latest model
```

## 🧪 Test Dataset Loading

```bash
python scripts/test_dataset.py
```

Output:
```
Testing Dataset Loading...
Classes: ['DJI_Phantom', 'DJI_Mavic', ...]

TRAIN SET:
  Samples: 1000
  Spectrogram shape: torch.Size([1, 256, 256])
  Target shape: torch.Size([16, 16, 20])
  Annotations in first image: 2

VAL SET:
  Samples: 300
  ...

TEST SET:
  Samples: 200
  ...
```

## 📚 Ví Dụ Sử Dụng

```bash
python examples/dataset_usage.py
```

Xem chi tiết trong `examples/dataset_usage.py`:
- Load dataset
- Get single sample
- Batch loading
- Coordinate conversion
- Visualization

## 🔧 Config Options (config.py)

```python
# Classes
CLASSES = ['DJI_Phantom', 'DJI_Mavic', ...]
NUM_CLASSES = 10

# Input
INPUT_SIZE = (256, 256)

# YOLO grid
GRID_SIZE = 16
NUM_BOXES = 2

# Training
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 100

# Detection
CONF_THRESHOLD = 0.4
NMS_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

# Paths
DATA_DIR = './data'  # YOLOv5 format
CHECKPOINT_DIR = './checkpoints'
LOG_DIR = './logs'
```

## 📝 Dataset Format Chi Tiết

### Cấu Trúc Thư Mục
```
data/
├── images/
│   ├── train/
│   │   ├── img001.jpg
│   │   ├── img002.png
│   │   └── ...
│   ├── val/
│   │   ├── img501.jpg
│   │   └── ...
│   └── test/
│       ├── img801.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── img001.txt
    │   ├── img002.txt
    │   └── ...
    ├── val/
    │   ├── img501.txt
    │   └── ...
    └── test/
        ├── img801.txt
        └── ...
```

### Label Format Chi Tiết

**File: `data/labels/train/image123.txt`**
```
0 0.412 0.514 0.312 0.425
5 0.812 0.201 0.152 0.238
2 0.150 0.700 0.100 0.150
```

Giải thích:
- Dòng 1: Class 0 (DJI_Phantom), center (0.412, 0.514), size (0.312, 0.425)
- Dòng 2: Class 5 (Custom_Drone), center (0.812, 0.201), size (0.152, 0.238)
- Dòng 3: Class 2 (DJI_Inspire), center (0.150, 0.700), size (0.100, 0.150)

## ✅ Checklist Trước Khi Train

- [ ] Cấu trúc `data/images/{train,val,test}` tồn tại
- [ ] Cấu trúc `data/labels/{train,val,test}` tồn tại
- [ ] Số ảnh = số labels trong mỗi split
- [ ] Chạy `prepare_dataset.py --verbose` không có errors
- [ ] Classes khớp với config.py
- [ ] Labels format là YOLOv5
- [ ] Test dataset loader: `test_dataset.py`

## 🎯 Next Steps

1. **Setup dataset** → Chuẩn bị files theo format YOLOv5
2. **Validate** → Chạy `prepare_dataset.py`
3. **Test** → Chạy `test_dataset.py`
4. **Train** → Chạy `train.py`
5. **Evaluate** → Chạy `inference.py`

## 📞 Support

Nếu gặp vấn đề:
1. Xem `DATASET_GUIDE.md` để hướng dẫn chi tiết
2. Xem `CHANGES_SUMMARY.md` để understand thay đổi
3. Xem `examples/dataset_usage.py` để ví dụ code

---

**Version:** 1.0  
**Date:** December 2024
