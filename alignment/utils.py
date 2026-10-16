import math
import random

import torch


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def trim_padding(batch):
    width = int(batch["attention_mask"].sum(dim=1).max())
    return {key: value[:, :width] for key, value in batch.items()}


def cosine_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
