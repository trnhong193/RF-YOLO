import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLOLoss(nn.Module):
    """
    YOLO Loss function
    Kết hợp localization loss, confidence loss và classification loss
    """
    
    def __init__(self, config):
        super(YOLOLoss, self).__init__()
        
        self.config = config
        self.S = config.GRID_SIZE
        self.B = config.NUM_BOXES
        self.C = config.NUM_CLASSES
        
        self.lambda_coord = config.LAMBDA_COORD
        self.lambda_noobj = config.LAMBDA_NOOBJ
    
    def compute_iou(self, box1, box2):
        """
        Tính IoU giữa 2 boxes
        box format: [x_center, y_center, width, height]
        """
        # Convert to corner coordinates
        box1_x1 = box1[..., 0] - box1[..., 2] / 2
        box1_y1 = box1[..., 1] - box1[..., 3] / 2
        box1_x2 = box1[..., 0] + box1[..., 2] / 2
        box1_y2 = box1[..., 1] + box1[..., 3] / 2
        
        box2_x1 = box2[..., 0] - box2[..., 2] / 2
        box2_y1 = box2[..., 1] - box2[..., 3] / 2
        box2_x2 = box2[..., 0] + box2[..., 2] / 2
        box2_y2 = box2[..., 1] + box2[..., 3] / 2
        
        # Intersection
        inter_x1 = torch.max(box1_x1, box2_x1)
        inter_y1 = torch.max(box1_y1, box2_y1)
        inter_x2 = torch.min(box1_x2, box2_x2)
        inter_y2 = torch.min(box1_y2, box2_y2)
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                     torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Union
        box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
        box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
        union_area = box1_area + box2_area - inter_area
        
        iou = inter_area / (union_area + 1e-6)
        
        return iou
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: (batch_size, S, S, B*5+C)
            targets: (batch_size, S, S, B*5+C)
        """
        batch_size = predictions.size(0)
        
        # Split predictions
        pred_boxes = []
        pred_confidence = []
        for b in range(self.B):
            start = b * 5
            pred_confidence.append(predictions[..., start:start+1])
            pred_boxes.append(predictions[..., start+1:start+5])
        
        pred_classes = predictions[..., self.B*5:]
        
        # Split targets
        target_boxes = []
        target_confidence = []
        for b in range(self.B):
            start = b * 5
            target_confidence.append(targets[..., start:start+1])
            target_boxes.append(targets[..., start+1:start+5])
        
        target_classes = targets[..., self.B*5:]
        
        # Mask for cells containing objects
        obj_mask = target_confidence[0] > 0  # (batch, S, S, 1)
        noobj_mask = target_confidence[0] == 0
        
        # Localization loss (chỉ tính cho cells có object)
        loc_loss = 0
        for b in range(self.B):
            # xy loss
            xy_loss = F.mse_loss(
                pred_boxes[b][..., :2] * obj_mask,
                target_boxes[b][..., :2] * obj_mask,
                reduction='sum'
            )
            
            # wh loss (square root)
            wh_pred = torch.sqrt(torch.abs(pred_boxes[b][..., 2:]) + 1e-6)
            wh_target = torch.sqrt(torch.abs(target_boxes[b][..., 2:]) + 1e-6)
            wh_loss = F.mse_loss(
                wh_pred * obj_mask,
                wh_target * obj_mask,
                reduction='sum'
            )
            
            loc_loss += xy_loss + wh_loss
        
        loc_loss = self.lambda_coord * loc_loss / batch_size
        
        # Confidence loss
        conf_loss = 0
        
        # Object confidence loss
        for b in range(self.B):
            obj_conf_loss = F.mse_loss(
                pred_confidence[b] * obj_mask,
                target_confidence[b] * obj_mask,
                reduction='sum'
            )
            conf_loss += obj_conf_loss
        
        # No-object confidence loss
        for b in range(self.B):
            noobj_conf_loss = F.mse_loss(
                pred_confidence[b] * noobj_mask,
                target_confidence[b] * noobj_mask,
                reduction='sum'
            )
            conf_loss += self.lambda_noobj * noobj_conf_loss
        
        conf_loss = conf_loss / batch_size
        
        # Classification loss (chỉ tính cho cells có object)
        class_loss = F.mse_loss(
            pred_classes * obj_mask,
            target_classes * obj_mask,
            reduction='sum'
        ) / batch_size
        
        # Total loss
        total_loss = loc_loss + conf_loss + class_loss
        
        return {
            'total': total_loss,
            'localization': loc_loss,
            'confidence': conf_loss,
            'classification': class_loss
        }