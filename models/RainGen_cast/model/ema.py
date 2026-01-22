# file: utils/ema.py

import torch
import torch.nn as nn
from copy import deepcopy
from collections import OrderedDict

class EMA:
    """
    高质量的指数移动平均（EMA）实现，专为PyTorch和分布式训练设计。
    """
    def __init__(self, model, decay, device=None):
        self.ema_model = deepcopy(model)
        self.ema_model.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.ema_model.to(self.device)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.model_params = list(model.parameters())
        self.ema_params = list(self.ema_model.parameters())

    def update(self, model=None):
        if model is None:
            source_params = self.model_params
        else:
            if isinstance(model, nn.parallel.DistributedDataParallel):
                source_params = list(model.module.parameters())
            else:
                source_params = list(model.parameters())
        with torch.no_grad():
            for ema_p, model_p in zip(self.ema_params, source_params):
                if ema_p.device != model_p.device:
                    model_p = model_p.to(ema_p.device)
                ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)

    def state_dict(self):
        state_dict = self.ema_model.state_dict()
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        return new_state_dict

    def load_state_dict(self, state_dict):
        ema_keys = self.ema_model.state_dict().keys()
        ema_has_module_prefix = any(k.startswith('module.') for k in ema_keys)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        if ema_has_module_prefix:
            self.ema_model.module.load_state_dict(new_state_dict)
        else:
            self.ema_model.load_state_dict(new_state_dict)