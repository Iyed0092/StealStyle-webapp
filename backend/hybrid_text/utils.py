import os
import random
import numpy as np
import torch
from datetime import datetime

def set_seed(seed: int = 42, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: str, epoch: int, meta: dict = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "meta": meta or {}
    }
    torch.save(state, path)
    print(f"[INFO] Saved checkpoint: {path}")

def load_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    print(f"[INFO] Loaded checkpoint from {path}")
    return ckpt
