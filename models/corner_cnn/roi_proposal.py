import time
import gc
import cv2
import numpy as np
from typing import List, Union, Dict, Optional, Tuple, Any

import torch
from torch import nn, Tensor
import torchvision.ops as O
import torchvision


class AnchorGenerator(nn.Module):
    def __init__(
        self, 
        anc_size: Optional[Tuple[int, int]] = (31, 31), #  h, w
        max_corners: int = 600, 
        quality_level: int = 0.002, 
        min_distance: int = 31,
        ):
        super().__init__()
        self.anc_size = anc_size
        
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
    
    def forward(
        self, 
        images: List[Tensor]
        ) -> Tensor:
        anc_bases = torch.neg(torch.ones(len(images), self.max_corners, 4)) * self.min_distance # [B, n_ancs, 4]
        for b, image in enumerate(images):
            image = image.squeeze().cpu().detach().numpy()
            corners = cv2.goodFeaturesToTrack(image, self.max_corners, self.quality_level, self.min_distance)
            corners = np.int0(corners)
            anc_centers = torch.from_numpy(corners).squeeze()
            for anc_id, (x, y) in enumerate(anc_centers):
                xmin = x - self.anc_size[1] // 2
                ymin = y - self.anc_size[0] // 2
                xmax = x + self.anc_size[1] // 2
                ymax = y + self.anc_size[0] // 2
                anc_boxes = torch.Tensor([xmin, ymin, xmax, ymax])
                anc_bases[b, anc_id, :] = O.clip_boxes_to_image(anc_boxes, size=(image.shape[0], image.shape[1]))
            del image
            del corners
            gc.collect()
        anc_bases = anc_bases.cuda() if torch.cuda.is_available() else anc_bases
        return anc_bases

# build corner proposal module
class CornerProposal(nn.Module):
    def __init__(
        self, 
        max_corners: int = 600, 
        min_distance: int = 31,
        pos_thresh: float = 0.8,
        neg_thresh: float = 0.2,
        ):
        super().__init__()
        self.max_corners = max_corners
        self.min_distance = min_distance

        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh
    
    def forward(
        self, 
        images: Tensor, # [B, c, h, w]
        anc_bases: Tensor, # [B, max_corners, 4]
        targets: Optional[Dict[str, Tensor]] = None
        ) -> Tuple[Tensor, Union[Tensor, None], Union[Tensor, None], Tensor]:
        B, c0, h, w = len(images), images[0].size(dim=0), images[0].size(dim=1), images[0].size(dim=2)
        
        if self.training:
            N = targets["labels"].size(dim=1) # max number of objects per image across batch
            gts, cls = targets["boxes"], targets["labels"] # [B, N, 4], [B, N]
            # compute an "ioa" matrix per image
            ioas_mat = torch.zeros((B, N, self.max_corners), device=gts.device) # [B, N, max_corners]
            for i in range(B):
                gt_boxes = gts[i] # [N, 4]
                anc_boxes = anc_bases[i] # [max_corners, 4]
                ioas_mat[i, :] = self._box_ioa(gt_boxes, anc_boxes) # [N, max_corners]
            ioas_mat = torch.transpose(ioas_mat, 1, 2) # [B, max_corners, N]
            gt_most_overlap_inds = torch.argmax(ioas_mat, dim=2) # [B, max_corners]
            max_iou_per_gt_box, _ = ioas_mat.max(dim=1, keepdim=True) # [B, 1, N]

            # get positive anchor boxes
            positive_anc_mask = torch.logical_and(ioas_mat == max_iou_per_gt_box, max_iou_per_gt_box > 0)
            positive_anc_mask = torch.logical_or(positive_anc_mask, ioas_mat > self.pos_thresh) # [B, max_corners, 1]
            pos_inds = positive_anc_mask.nonzero(as_tuple=True)[:-1] # 2 tensors of indices
            # get negative anchor boxes
            negative_anc_mask = ioas_mat < self.neg_thresh # [B, max_corners, 1]
            neg_inds = negative_anc_mask.nonzero(as_tuple=True)[:-1] # 2 tensors of indices

            if pos_inds[0].numel() != 0 or neg_inds[0].numel() != 0:
                # map gts to corr anchors
                gts_expand = gts.view(B, 1, N, 4).expand(B, self.max_corners, N, 4)
                ancs_to_gts = torch.gather(gts_expand, -2, gt_most_overlap_inds.reshape(B, self.max_corners, 1, 1).repeat(1, 1, 1, 4)) # [B, max_corners, 1, 4]
                ancs_to_gts = ancs_to_gts.flatten(start_dim=2) # [B, max_corners, 4]
                ancs_to_gts = torch.where(positive_anc_mask, ancs_to_gts, torch.tensor(-self.min_distance, device=gts.device).float())

                # map cls to corr anchors
                cls_expand = cls.view(B, 1, N).expand(B, self.max_corners, N)
                ancs_to_cls = torch.gather(cls_expand, -1, gt_most_overlap_inds.unsqueeze(-1)).squeeze(-1) # [B, max_corners]
                ancs_to_cls = torch.where(positive_anc_mask.flatten(start_dim=1), ancs_to_cls, 0)

                # extract a sastified anchor's indices
                pos_ind_selected = torch.stack(pos_inds)[:, :B] # [2, ~B]
                neg_ind_selected = torch.stack(neg_inds)[:, :pos_ind_selected.size(dim=1)] # [2, ~B]
                roi_inds = tuple(torch.cat((pos_ind_selected, neg_ind_selected), dim=1)) # [2, ~2xB]

                # extract ROIs
                roi_bases = anc_bases[roi_inds] # [~2xB, 4] -> xmin, ymin, xmax, ymax
                roi_centers = (roi_bases[:, :2] + roi_bases[:, 2:]) // 2 # [~2xB, 2]
                images = images[roi_inds[:-1]] # [~2xB, c, h, w]
                rois = self._extract_glimpse(
                    images, 
                    size=(self.min_distance, self.min_distance), 
                    offsets=roi_centers
                    ) # [~2xB, c, min_distance, min_distance]
                roi_cls = ancs_to_cls[roi_inds] # [~2xB]
                #roi_gts = (ancs_to_gts[roi_inds] - torch.cat((roi_bases[roi_inds[:-1]][:, :2], roi_bases[roi_inds[:-1]][:, :2]), dim=1)) / (self.min_distance - 1) # [~2xB, 4]
                roi_gts = ancs_to_gts[roi_inds]

                return rois, roi_cls, roi_gts, roi_bases[roi_inds[:-1]][:, :2] # [~2xB, 2]
        
        anc_centers = (anc_bases[:, :, :2] + anc_bases[:, :, :2]) // 2 # [B, max_corners, 2]
        rois = self._extract_glimpses(
            images, 
            size=(self.min_distance, self.min_distance), 
            offsets=anc_centers
            ) # [B, max_corners, c, min_distance, min_distance]
        return rois, None, None, anc_bases[:, :, :2] # [B, max_corners, 2]

    def _box_ioa(
        self, 
        gt_boxes: Tensor, 
        anc_boxes: Tensor
        ) -> Tensor:
        ioa_mat = torch.zeros((gt_boxes.size(dim=0), anc_boxes.size(dim=0)))
        for i in range(len(gt_boxes)):
            xmin = torch.max(gt_boxes[i, 0], anc_boxes[:, 0])
            ymin = torch.max(gt_boxes[i, 1], anc_boxes[:, 1])
            xmax = torch.min(gt_boxes[i, 2], anc_boxes[:, 2])
            ymax = torch.min(gt_boxes[i, 3], anc_boxes[:, 3])

            w = (xmax - xmin + 1).double()
            h = (ymax - ymin + 1).double()
            intersection = torch.where((w > 0) & (h > 0), w * h, 0.)
            gt_area = (gt_boxes[i, 2] - gt_boxes[i, 0] + 1) * (gt_boxes[i, 3] - gt_boxes[i, 1] + 1)
            ioa_mat[i, :] = intersection / gt_area
        return ioa_mat

    def _extract_glimpse(
        self,
        input: Tensor, # [B, C, H, W]
        size: Tuple[int, int],
        offsets: Tensor, # [B, 2]
        centered=False, 
        normalized=False, 
        mode='bilinear', 
        padding_mode='zeros'
        ) -> Tensor:
        W, H = input.size(-1), input.size(-2)

        if normalized and centered:
            offsets = (offsets + 1) * offsets.new_tensor([W/2, H/2])
        elif normalized:
            offsets = offsets * offsets.new_tensor([W, H])
        elif centered:
            raise ValueError(
                f'Invalid parameter that offsets centered but not normlized')

        h, w = size
        xs = torch.arange(0, w, dtype=input.dtype,
                        device=input.device) - (w - 1) / 2.0
        ys = torch.arange(0, h, dtype=input.dtype,
                        device=input.device) - (h - 1) / 2.0

        vy, vx = torch.meshgrid(ys, xs)
        grid = torch.stack([vx, vy], dim=-1)  # h, w, 2

        offsets_grid = offsets[:, None, None, :] + grid[None, ...]

        # normalised grid to [-1, 1]
        offsets_grid = (
            offsets_grid - offsets_grid.new_tensor([W/2, H/2])) / offsets_grid.new_tensor([W/2, H/2])

        return torch.nn.functional.grid_sample(
            input, offsets_grid, mode=mode, align_corners=False, padding_mode=padding_mode)

    def _extract_glimpses(
        self,
        input: Tensor, # [B, C, H, W]
        size: Tuple[int, int],
        offsets: Tensor, # [B, max_corners, 2]
        centered=False, 
        normalized=False, 
        mode='bilinear', 
        padding_mode='zeros'
        ) -> Tensor:
        patches = [] # [max_corners, B, c, size, size]
        for i in range(offsets.size(-2)):
            patch = self._extract_glimpse(input, size, offsets[:, i, :], centered, normalized, mode, padding_mode=padding_mode) # [B, c, size, size]
            patches.append(patch)
        return torch.stack(patches, dim=1) # [B, max_corners, c, size, size]
