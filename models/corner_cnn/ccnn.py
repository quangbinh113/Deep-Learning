from typing import List, Union, Dict, Optional, Tuple, Any

import torch
from torch import nn, Tensor

from .feature_extractor import *
from .roi_proposal import AnchorGenerator, CornerProposal
from ..losses import DIoULoss

class CCNN(nn.Module):
    def __init__(
        self,
        predictor: nn.Module,
        roi_transform: Any = None,
        
        anchor_size: Optional[Tuple[int, int]] = (31, 31),
        max_corners: int = 600, 
        quality_level: int = 0.002, 
        min_distance: int = 11,
        
        pos_thresh: float = 0.7,
        neg_thresh: float = 0.2,
        
        score_thresh: float = 0.5,
        loss_reg_cls_ratio: float = 0.80,
        nms_thresh: float = 0.6
        ):
        super().__init__()
        self.score_thresh = score_thresh
        self.reg_ratio = loss_reg_cls_ratio
        self.nms_thresh = nms_thresh
        
        self.anchor_generator = AnchorGenerator(anchor_size, max_corners, quality_level, min_distance)
        self.corner_proposal = CornerProposal(max_corners, min_distance, pos_thresh, neg_thresh)
        self.roi_transform = roi_transform
        self.predictor = predictor
    
    def forward(
        self, 
        images: Optional[List[Tensor]], 
        targets: Optional[List[Dict[str, Tensor]]] = None
        ) -> Union[Tensor, List[Dict[str, Tensor]]]:
        torch._assert(images is not None, "[ERROR] Images cannot be missing")
        if self.training:
            torch._assert(targets is not None, "[ERROR] Targets should not be none during training")
            targets = self._stack_targets(targets) # [B, N, 4], [B, N]
        
        anc_bases = self.anchor_generator(images)
        if images is not Tensor:
            images = torch.stack(images)
        h, w = images.size(dim=-2), images.size(dim=-1)
        rois, roi_cls, roi_gts, roi_uplefts = self.corner_proposal(images, anc_bases, targets)
        if self.roi_transform:
            rois = self.roi_transform(rois)
        
        if self.training:
            locs, scores = self.predictor(rois) # [B, out_channels, 1, 1]
            locs = self._decode_loc(locs, roi_uplefts) # [~2xB, 4]
            scores = torch.flatten(scores, start_dim=0) # [~2xB]
            loss = self.compute_loss(locs, scores, roi_gts, roi_cls)
            return loss
        
        detections = []
        for b, ins_rois in enumerate(rois):
            locs, scores = self.predictor(ins_rois)
            locs = self._decode_loc(locs, roi_uplefts[b]) # [~2xB, 4]
            scores = torch.flatten(scores, start_dim=0) # [~2xB]
            locs, scores, labels = self.postprocess_detections(locs, scores, (h, w))
            detections.append(self._one_detection(locs, scores, labels))
        return detections
            
    def compute_loss(self, locs, scores, gts, labels) -> Tensor:
        loc_loss_fn = DIoULoss(reduction="mean", weights=labels)
        cls_loss_fn = nn.BCELoss()
        cls_loss = cls_loss_fn(scores, labels.float())
        loc_loss = loc_loss_fn(locs, gts)
        return (1 - self.reg_ratio) * cls_loss + self.reg_ratio * loc_loss
    
    def _stack_targets(
        self, 
        targets: List[Dict[str, Tensor]]
        ) -> Dict[str, Tensor]:
        target_stack = {}
        for k in targets[0].keys():
            tensors = (targets[0][k],)
            for i in range(1, len(targets)):
                tensors = tensors + (targets[i][k],)
            target_stack[k] = torch.stack(tensors)
        return target_stack
        
    def _one_detection(self, locs, scores, labels) -> Dict[str, Tensor]:
        return {
            "boxes": locs, # [~2xB, 4]
            "scores": scores, # [~2xB]
            "labels": labels # [~2xB]
        }
    
    def _decode_loc(self, locs, roi_uplefts):
        locs = locs * (self.corner_proposal.min_distance - 1) + torch.cat((roi_uplefts, roi_uplefts), dim=1) # [~2xB, 4]
        locs[:, 2:] = torch.ceil(locs[:, 2:])
        locs[:, :2] = torch.floor(locs[:, :2])
        return locs # [~2xB, 4]
    
    def postprocess_detections(self, locs, scores, image_shape):
        locs = O.clip_boxes_to_image(locs, image_shape)
        labels = torch.where(scores.double() > self.score_thresh, 1, 0) # [~2xB]

        # remove low scoring boxes
        keep = torch.where(scores > self.score_thresh)
        locs, labels, scores = locs[keep], labels[keep], scores[keep]

        # remove empty boxes
        keep = O.remove_small_boxes(locs, min_size=1e-3)
        locs, scores, labels = locs[keep], scores[keep], labels[keep]

        # non-maximum suppression
        keep = O.nms(locs, scores, self.nms_thresh)
        locs, scores, labels = locs[keep], scores[keep], labels[keep]

        return locs, scores, labels