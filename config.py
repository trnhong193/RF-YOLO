import torch

class Config:
    # Thiết bị
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dữ liệu
    SAMPLE_RATE = 24576000000  # 245.76 MHz   # tần số lấy mẫu
    FFT_SIZE = 2048
    TIME_STEPS = 256
    INPUT_SIZE = (256, 256)
    
    # Dải tần
    FREQ_BANDS = {
        '900MHz': {'center': 915e6, 'bandwidth': 25e6},
        '2.4GHz': {'center': 2.45e9, 'bandwidth': 100e6},
        '5.8GHz': {'center': 5.8e9, 'bandwidth': 200e6}
    }
    
    # Classes
    CLASSES = [
        'DJI_Phantom', 'DJI_Mavic', 'DJI_Inspire',
        'Parrot', 'Autel', 'Custom_Drone',
        'WiFi', 'Bluetooth', 'Noise', 'Background'
    ]
    NUM_CLASSES = len(CLASSES)
    
    # YOLO parameters
    GRID_SIZE = 16  # 16x16 grid
    NUM_BOXES = 2   # 2 bounding boxes per grid cell
    
    # Training
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    NUM_WORKERS = 4
    
    # Loss weights
    LAMBDA_COORD = 5.0
    LAMBDA_NOOBJ = 0.5
    
    # Detection thresholds
    CONF_THRESHOLD = 0.4
    NMS_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.5
    
    # Paths
    DATA_DIR = '/home/tth193/Documents/Drones_prj/data_388'  # YOLOv5 format: data/images/{train,val,test} and data/labels/{train,val,test}
    CHECKPOINT_DIR = './checkpoints'
    LOG_DIR = './logs'
    
    # Dataset format: 'yolov5' - supports (images/train,val,test) and (labels/train,val,test)
    DATASET_FORMAT = 'yolov5'