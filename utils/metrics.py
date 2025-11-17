import numpy as np
import torch

def compute_iou(box1, box2):
    """
    Tính IoU giữa 2 boxes
    box format: [x_center, y_center, width, height]
    """
    # Convert to corners
    box1_x1 = box1[0] - box1[2] / 2
    box1_y1 = box1[1] - box1[3] / 2
    box1_x2 = box1[0] + box1[2] / 2
    box1_y2 = box1[1] + box1[3] / 2
    
    box2_x1 = box2[0] - box2[2] / 2
    box2_y1 = box2[1] - box2[3] / 2
    box2_x2 = box2[0] + box2[2] / 2
    box2_y2 = box2[1] + box2[3] / 2
    
    # Intersection
    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    # Union
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / (union_area + 1e-6)
    
    return iou

def non_max_suppression(detections, iou_threshold=0.5):
    """
    Non-Maximum Suppression
    
    Args:
        detections: List of dicts with 'bbox', 'confidence', 'class'
        iou_threshold: IoU threshold for suppression
        
    Returns:
        filtered_detections: Detections after NMS
    """
    if len(detections) == 0:
        return []
    
    # Sort by confidence
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    keep = []
    
    while len(detections) > 0:
        # Keep highest confidence detection
        best = detections.pop(0)
        keep.append(best)
        
        # Remove overlapping detections
        filtered = []
        for det in detections:
            # Only suppress same class
            if det['class'] == best['class']:
                iou = compute_iou(best['bbox'], det['bbox'])
                if iou < iou_threshold:
                    filtered.append(det)
            else:
                filtered.append(det)
        
        detections = filtered
    
    return keep

def compute_ap(recalls, precisions):
    """
    Tính Average Precision
    
    Args:
        recalls: List of recall values
        precisions: List of precision values
        
    Returns:
        ap: Average Precision
    """
    # Add sentinel values
    recalls = np.concatenate([[0], recalls, [1]])
    precisions = np.concatenate([[0], precisions, [0]])
    
    # Compute precision envelope
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    # Integrate
    indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
    ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
    
    return ap

def evaluate_detection(predictions, ground_truths, iou_threshold=0.5, 
                       num_classes=10):
    """
    Đánh giá detection performance
    
    Args:
        predictions: List of lists of detections per image
        ground_truths: List of lists of ground truth per image
        iou_threshold: IoU threshold for TP
        num_classes: Number of classes
        
    Returns:
        metrics: Dict with mAP, precision, recall per class
    """
    # Initialize
    all_precisions = []
    all_recalls = []
    aps = []
    
    for class_id in range(num_classes):
        # Collect all predictions and GTs for this class
        class_predictions = []
        class_gts = []
        
        for img_idx, (preds, gts) in enumerate(zip(predictions, ground_truths)):
            # Filter by class
            img_preds = [p for p in preds if p['class'] == class_id]
            img_gts = [g for g in gts if g['class'] == class_id]
            
            for pred in img_preds:
                class_predictions.append({
                    'image_id': img_idx,
                    'confidence': pred['confidence'],
                    'bbox': pred['bbox']
                })
            
            for gt in img_gts:
                class_gts.append({
                    'image_id': img_idx,
                    'bbox': gt['bbox'],
                    'detected': False
                })
        
        if len(class_gts) == 0:
            continue
        
        # Sort predictions by confidence
        class_predictions = sorted(class_predictions, 
                                  key=lambda x: x['confidence'], 
                                  reverse=True)
        
        # Match predictions to ground truths
        tp = np.zeros(len(class_predictions))
        fp = np.zeros(len(class_predictions))
        
        for pred_idx, pred in enumerate(class_predictions):
            # Find GTs in same image
            img_gts = [gt for gt in class_gts 
                      if gt['image_id'] == pred['image_id']]
            
            max_iou = 0
            max_gt_idx = -1
            
            for gt_idx, gt in enumerate(img_gts):
                iou = compute_iou(pred['bbox'], gt['bbox'])
                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = gt_idx
            
            if max_iou >= iou_threshold:
                if not img_gts[max_gt_idx]['detected']:
                    tp[pred_idx] = 1
                    img_gts[max_gt_idx]['detected'] = True
                else:
                    fp[pred_idx] = 1
            else:
                fp[pred_idx] = 1
        
        # Compute precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / len(class_gts)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        
        # Compute AP
        ap = compute_ap(recalls, precisions)
        aps.append(ap)
        
        if len(precisions) > 0:
            all_precisions.append(precisions[-1])
            all_recalls.append(recalls[-1])
    
    # Compute mAP
    mAP = np.mean(aps) if len(aps) > 0 else 0
    mean_precision = np.mean(all_precisions) if len(all_precisions) > 0 else 0
    mean_recall = np.mean(all_recalls) if len(all_recalls) > 0 else 0
    
    return {
        'mAP': mAP,
        'precision': mean_precision,
        'recall': mean_recall,
        'APs': aps
    }