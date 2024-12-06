import random
import torch
from transformers import Trainer, TrainerCallback, EvalPrediction

def get_mask(rank: int, k: int, device: torch.device='cuda'):
    mask = torch.zeros((rank, rank), device=device, requires_grad=False)
    mask[:k, :k] = 1
    return mask

class MaskingCallback(TrainerCallback):
    def __init__(self, k_min=10, k_max=25, rank=25):
        self.k_min = k_min
        self.k_max = k_max
        self.rank = rank
        self.cur_k = k_max
        self.cur_mask = None
        self.params_copy = {}

    def on_step_begin(self, args, state, control, model, **kwargs):
        self.params_copy.clear()

        self.cur_k = random.randint(self.k_min, self.k_max)
        device = model.device if hasattr(model, 'device') else torch.device('cuda')  # Ensure model is on correct device

        self.cur_mask = get_mask(self.rank, self.cur_k, device)

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and param.shape == (self.rank, self.rank):
                    self.params_copy[name] = param.data[self.cur_k:, self.cur_k:].clone()
                    param.data = param.data * self.cur_mask

        return control

    # def on_step_end(self, args, state, control, model, **kwargs):
    #     with torch.no_grad():
    #         for name, param in model.named_parameters():
    #             if param.requires_grad and param.shape == (self.rank, self.rank):
    #                 param.data[self.cur_k:, self.cur_k:] = self.params_copy[name]

    #     return control
    