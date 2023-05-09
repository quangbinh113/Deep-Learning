import os
import pandas as pd
import numpy as np
from PIL import Image
from typing import Dict, Tuple, Union, Optional

import torch
from torch.utils.data import Dataset
from torch import Tensor

from .transforms import TransformComposer

class SIRSTDataset(Dataset):
    def __init__(
        self, 
        annotations_file: str, 
        img_dir: str, 
        transform: Optional[TransformComposer] = None, 
        target_transform: Optional[TransformComposer] = None
        ):
        self.img_labels = pd.read_csv(annotations_file).to_numpy()
        self.img_dir = img_dir
        self.transform = transform if transform else TransformComposer()
        self.target_transform = target_transform if target_transform else TransformComposer(transforms=None)

class NUDTSIRSTDataset(SIRSTDataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        super().__init__(annotations_file, img_dir, transform, target_transform)
        self.training = True

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(
        self, ix: int
        ) -> Tuple[Tensor, Dict[str, Tensor]]:
        img_path = os.path.join(self.img_dir, 
                                os.path.join(self.img_dir, self.img_labels[ix, 0]))
        img = Image.open(img_path.replace('\\', os.sep)).convert('L')
        boxes = self.img_labels[ix:ix+1, 1:-1] # exclude contrast
        
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            boxes = self.target_transform(boxes)
        
        target = {}
        target['boxes'] = torch.from_numpy(boxes.astype('float32'))
        target['labels'] = torch.from_numpy(np.ones(len(boxes))).type(torch.int64)

        return img, target
    
    def eval(self):
        self.training = False