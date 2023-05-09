import torch
from torch import nn, Tensor
from typing import List, Union, Dict, Optional, Tuple, Any

class DIoULoss(nn.Module):
    """
    Distance Intersection over Union Loss (Zhaohui Zheng et. al)
    https://arxiv.org/abs/1911.08287
    Args:
        input, target (Tensor): box locations in XYXY format, shape (N, 4) or (4,).
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        eps (float): small number to prevent division by zero
    """
    __constant__ = ["none", "sum", "mean"]
    
    def __init__(
        self,
        eps: float = 1e-7,
        reduction: Optional[str] = None,
        weights: Optional[Tensor] = None
        ):
        super(DIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.weights = weights
    
    def forward(
        self,
        input: Tensor,
        target: Tensor
        ) -> Tensor:
        intsct, union = self._loss_inter_union(input, target)
        iou = intsct / (union + self.eps)
        
        # smallest enclosing box
        x1, y1, x2, y2 = input.unbind(dim=-1)
        x1g, y1g, x2g, y2g = target.unbind(dim=-1)
        xc1 = torch.min(x1, x1g)
        yc1 = torch.min(y1, y1g)
        xc2 = torch.max(x2, x2g)
        yc2 = torch.max(y2, y2g)
        
        # the diagonal distance of the smallest enclosing box squared
        diagonal_distance_squared = ((xc2 - xc1) ** 2) + ((yc2 - yc1) ** 2) + self.eps
        
        # centers of boxes
        x_p = (x2 + x1) / 2
        y_p = (y2 + y1) / 2
        x_g = (x1g + x2g) / 2
        y_g = (y1g + y2g) / 2
        
        # the distance between boxes' centers squared.
        centers_distance_squared = ((x_p - x_g) ** 2) + ((y_p - y_g) ** 2)
        
        # distance between boxes' centers squared.
        loss = 1 - iou + (centers_distance_squared / diagonal_distance_squared)
        
        # eqn. (7)
        loss = 1 - iou + (centers_distance_squared / diagonal_distance_squared)
        if self.weights is not None:
            loss = loss * self.weights
        loss = loss[torch.nonzero(loss, as_tuple=True)]
        
        if self.reduction == "mean":
            loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss
    
    def _loss_inter_union(
        self,
        boxes1: torch.Tensor,
        boxes2: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:

        x1, y1, x2, y2 = boxes1.unbind(dim=-1)
        x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

        # Intersection keypoints
        xkis1 = torch.max(x1, x1g)
        ykis1 = torch.max(y1, y1g)
        xkis2 = torch.min(x2, x2g)
        ykis2 = torch.min(y2, y2g)

        intsctk = torch.zeros_like(x1)
        mask = (ykis2 > ykis1) & (xkis2 > xkis1)
        intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
        unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk

        return intsctk, unionk