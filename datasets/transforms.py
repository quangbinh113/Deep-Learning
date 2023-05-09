import numpy as np
from typing import Union, Dict, List

import torch
import torchvision.transforms as T

class TransformComposer:
    def __init__(
        self, 
        training: bool = True,
        transforms: Union[Dict[str, List], None] = {"default": [T.ToTensor(), T.ConvertImageDtype(torch.float32)], "train": [], "eval": []}
        ):
        self.training = training
        self.transforms = transforms if transforms is not None else {"default": [], "train": [], "eval": []}
    
    def __call__(self, sample):
        if self.training:
            transform = T.Compose(self.transforms.get("default", []) + self.transforms.get("train", []))
            return transform(sample)
        else:
            transform = T.Compose(self.transforms.get("default", []) + self.transforms.get("eval", []))
            return transform(sample)

class XywhToXyxy:
    def __call__(self, xywh_boxes):
        xyxy_boxes = []
        for i in range(len(xywh_boxes)):
            xmin = xywh_boxes[i][0]
            ymin = xywh_boxes[i][1]
            xmax = xywh_boxes[i][2] + xmin
            ymax = xywh_boxes[i][3] + ymin
            xyxy_boxes.append([xmin, ymin, xmax, ymax])
        return np.array(xyxy_boxes)