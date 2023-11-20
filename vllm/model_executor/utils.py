"""Utils for model executor."""
import random

import numpy as np
import torch


def set_random_seed(seed: int, device: str = 'cuda') -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    elif device == 'xpu':
        try:
            import intel_extension_for_pytorch
        except ImportError:
            print("Can't find intel extension for pytorch in your environment")
        if torch.xpu.is_available():
            torch.xpu.manual_seed_all(seed)
