import copy
from typing import Union, Tuple, Any, List, Dict
import time
import gc

import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from ..datasets.datasets import SIRSTDataset
from . import callbacks as cb
from . import metrics


class Trainer:
    MAX_LOAD_SIZE = 8 # 15GB VRAM

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        dataset: SIRSTDataset,
        batch_size: int = 32,
        valid_ratio: int = 0.05,
        valid_freq: int  = 1,
        tensorboard: SummaryWriter = None,
        output: Union[str, None] = None,
        input: Union[str, None] = None,
        log_freq_gradient: int = 10,
        metric: Any = metrics.SIRSTMetrics(),
        lr_scheduler: Any = None
        ):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.valid_ratio = valid_ratio
        self.valid_freq = valid_freq
        self.data_loader, self.data_loader_valid = self._train_test_split()

        self.input = input
        self.output = output
        
        self.tensorboard = tensorboard
        self.save_model = cb.BestCheckpoint(output=output)
        self.save_validated_model = cb.BestCheckpoint(output="valid_" + output)
        self.display = cb.Displayer()
        self.metric = metric

        self.model.to(self.device)

        self.log_freq_gradient = log_freq_gradient
    
    def fit(
        self,
        start_epoch: int = 0, # inclusive
        end_epoch: int = 20, # exclusive
        ):
        warm_up = True
        for epoch in range(start_epoch, end_epoch):
            if not warm_up:
                print() # vanish \r
            else:
                warm_up = False
            print(f"Epoch {epoch+1}/{end_epoch}")
            self.train_one_epoch(epoch)
    
    def train_one_epoch(self, epoch):
        for iter, (images, targets) in enumerate(self.data_loader):
            self.train_one_step(iter, images, targets, epoch)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
    
    def train_one_step(self, iter, images, targets, epoch):
        ts = time.time()
        self.model.train()
        
        self.optimizer.zero_grad()
        chunks = [(images[i:i+self.MAX_LOAD_SIZE], targets[i:i+self.MAX_LOAD_SIZE]) for i in range(0, len(images), self.MAX_LOAD_SIZE)]
        for imgs, tars in chunks:
            imgs = list(image.to(self.device) for image in imgs)
            tars = [{k: v.to(self.device) for k, v in t.items()} for t in tars]
            
            losses = self.model(imgs, tars)
            losses = losses / len(chunks) # normalize across chunks

            del imgs
            del tars
            gc.collect()
            torch.cuda.empty_cache()

            losses.backward()
        
        if self.log_freq_gradient != 0 and iter % self.log_freq_gradient == 0:
            for name, weight in self.model.named_parameters():
                if weight.requires_grad:
                    self.tensorboard.add_histogram(name + ".grad", weight.grad, epoch * len(self.data_loader) + iter)
        self.optimizer.step()

        te = time.time()
        valid_losses = None
        if iter % int(self.valid_freq * len(self.data_loader)) == 0 and iter != 0:
            valid_losses = self.validate()
            for i, valid_loss in enumerate(valid_losses):
                self.tensorboard.add_scalar(f"valid/metric_{i}", valid_loss, epoch * len(self.data_loader) + iter)
                self.save_validated_model(sum(valid_loss.values()) / len(valid_loss),  epoch, self.model, self.optimizer, output=f"metric_{i}_" + self.output)
        
        valid_loss = None
        if valid_losses is not None:
            valid_loss = valid_losses[0]
            valid_loss = (sum(valid_loss.values()) / len(valid_loss)).item()
        
        self.display(len(self.data_loader), iter+1, te-ts, losses.item(), valid_loss=valid_loss)
        self.tensorboard.add_scalar("train/loss", losses.item(), epoch * len(self.data_loader) + iter)
        self.save_model(losses.item(), epoch, self.model, self.optimizer)
    
    def validate(self):
        self.model.eval()
        with torch.no_grad():
            for images, targets in self.data_loader_valid:
                chunks = [(images[i:i+self.MAX_LOAD_SIZE], targets[i:i+self.MAX_LOAD_SIZE]) for i in range(0, len(images), self.MAX_LOAD_SIZE)]
                for imgs, tars in chunks:
                    imgs = list(image.to(self.device) for image in imgs)
                    tars = [{k: v.to(self.device) for k, v in t.items()} for t in tars]
                    detections = self.model(imgs, tars)
                    
                    self.metric.update(detections, tars)
                    del imgs
                    del tars
                    gc.collect()
                    torch.cuda.empty_cache()
        return self.metric.compute()
    
    @torch.inference_mode()
    def evaluate(self, data_loader_test: DataLoader) -> Dict[str, Tensor]:
        self.model.eval()
        with torch.no_grad():
            for batch, (images, targets) in enumerate(data_loader_test):
                chunks = [(images[i:i+self.MAX_LOAD_SIZE], targets[i:i+self.MAX_LOAD_SIZE]) for i in range(0, len(images), self.MAX_LOAD_SIZE)]
                for imgs, tars in chunks:
                    imgs = list(image.to(self.device) for image in imgs)
                    tars = [{k: v.to(self.device) for k, v in t.items()} for t in tars]
                    detections = self.model(imgs, tars)
                    
                    self.metric.update(detections, tars)
                    del imgs
                    del tars
                    gc.collect()
                    torch.cuda.empty_cache()
                print(f"[{batch}]", "true_positive:", self.metric.true_pos, "false_positive:", self.metric.false_pos, "n_grounds:", self.metric.n_gts, "n_predictions:", self.metric.n_preds)
        return self.metric.compute()
    
    def _train_test_split(self) -> Tuple[DataLoader]:
        dataset_valid = copy.deepcopy(self.dataset)
        dataset_valid.transform.training = False
        dataset_valid.target_transform.training = False
        
        torch.manual_seed(1)
        split_idx = int(self.valid_ratio * len(self.dataset))
        indices = torch.randperm(len(self.dataset)).tolist()
        dataset = Subset(self.dataset, indices[:-split_idx])
        dataset_valid = Subset(dataset_valid, indices[-split_idx:])

        data_loader = DataLoader(
            dataset, batch_size=self.batch_size, 
            shuffle=True, collate_fn=lambda batch: tuple(zip(*batch)))
        valid_data_loader = DataLoader(
            dataset_valid, batch_size=self.batch_size,
            shuffle=False, collate_fn=lambda batch: tuple(zip(*batch)))
        
        return data_loader, valid_data_loader
    
    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name == "input":
            if __value is not None:
                try:
                    checkpoint = torch.load(__value, map_location="cpu")
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.model.to(self.device)
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    cb.optimizer_to(self.optimizer, self.device)
                except:
                    if self.model is None or self.optimizer is None:
                        print("[ERROR] - Must have the desired model/optimizer")
                    print("[ERROR] - The path would be wrong.")

        super(Trainer, self).__setattr__(__name, __value)
        
        if __name in ("dataset", "batch_size", "valid_ratio"):
            if __name == "batch_size":
                assert self.batch_size % self.MAX_LOAD_SIZE == 0
            try:
                self.data_loader, self.data_loader_valid = self._train_test_split()
            except AttributeError:
                pass
    