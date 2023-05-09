import os
from typing import Union
from pathlib import Path
PRJ_ROOT = Path(__file__).parent.parent.resolve()

import torch

class BestCheckpoint:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_loss = float('inf'), quality_loss = float('inf'),
        output=os.path.join(PRJ_ROOT, "logs", "checkpoints")
    ):
        self.best_loss = best_loss
        self.output = PRJ_ROOT if output is None else output
            
    def __call__(
        self, loss, 
        epoch, model, optimizer, output=None
        ):
        if loss <= self.best_loss:
            self.best_loss = loss
            if output is None:
                output = self.output
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, output)

class Displayer:
    def __init__(self):
        self.prev_print_len = 1

    def __call__(
        self,
        n_iters: int,
        iter: int,
        eta,
        train_loss: float,
        valid_loss: Union[float, None] = None
        ) -> None:
        valid_slot = f"valid_metrics: {valid_loss:.4f}" if valid_loss is not None else "valid_loss: nan"
        cur_print = " - ".join([" "*(len(str(n_iters)) - len(str(iter))) + f"{iter}/{n_iters}" + " " + f"[{iter*100//n_iters:>3d}%]",
                                f"eta: {eta:.2f}s",
                                f"train_loss: {train_loss:.4f}",
                                valid_slot])

        print(" " * self.prev_print_len, end="\r")
        self.prev_print_len = len(cur_print)
        print(cur_print, end="\r")

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)