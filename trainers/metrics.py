
from typing import List, Union, Dict, Optional, Tuple, Any

import torch
from torch import nn, Tensor
import torchvision.ops as O

class SIRSTMetrics:
    def __init__(
        self, 
        iou_thresholds: List[float] = [0.0, 0.5, 1.0],
        eps: float = 1e-7
        ):
        self.true_pos = [0] * len(iou_thresholds)
        self.false_pos = [0] * len(iou_thresholds)
        self.n_preds = 0
        self.n_gts = 0
        self.iou_thresholds = iou_thresholds
        self.eps = eps
    
    def compute(self) -> Dict[str, Tensor]:
        true_pos = torch.tensor(self.true_pos, device=self._device)
        false_pos = torch.tensor(self.false_pos, device=self._device)
        detect_rate = true_pos / (self.n_gts + self.eps) # true / n_targets
        false_alarm = false_pos / (self.n_preds + self.eps) # false / n_preds
        return {f"detection_rate_{self.iou_thresholds[i]}": detect_rate[i] for i in range(len(self.iou_thresholds))}, {f"false_alarm_rate_{self.iou_thresholds[i]}": false_alarm[i] for i in range(len(self.iou_thresholds))}
    
    def update(
        self, 
        preds: List[Dict[str, Tensor]], # [x, ...]
        targets: List[Dict[str, Tensor]] # [M, ...]
        ):
        self._device = targets[0]["boxes"].device
        old_n_preds = self.n_preds
        
        max_preds = len(max(preds, key=lambda r: len(r["boxes"]))["boxes"])
        max_targets = len(max(targets, key=lambda r: len(r["boxes"]))["boxes"])
        ious_mat = torch.zeros((len(preds), max_preds, max_targets), device=self._device) # [B, max_preds, max_targets]
        for i in range(len(preds)):
            pred = preds[i] # [N, 4]
            target = targets[i] # [M, 4]
            self.n_preds += pred["boxes"].size(dim=0)
            self.n_gts += target["boxes"].size(dim=0)
            
            iou_mat = O.box_iou(pred["boxes"], target["boxes"]) # [N, M]
            expand_h = max_preds - iou_mat.size(dim=0)
            expand_w = max_targets - iou_mat.size(dim=1)
            if expand_h == 0 and expand_w == 0:
                ious_mat[i, :] = iou_mat
                continue
            else:
                if expand_h == 0:
                    expand_h = 1
                elif expand_w == 0:
                    expand_w = 1
            
            ious_mat[i, :] = torch.vstack((iou_mat, torch.zeros((expand_h, expand_w)))) 
            
        for i, iou_threshold in enumerate(self.iou_thresholds):
            true_pos_inds = torch.where(ious_mat > iou_threshold)
            self.true_pos[i] += len(true_pos_inds[0])
            self.false_pos[i] += (self.n_preds - old_n_preds) - len(true_pos_inds[0])