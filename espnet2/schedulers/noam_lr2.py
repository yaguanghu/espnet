from distutils.version import LooseVersion
from typing import Union
import warnings
import logging

import torch
from typeguard import check_argument_types

from espnet2.schedulers.abs_scheduler import AbsBatchStepScheduler


class NoamLR2(AbsBatchStepScheduler):
    """The LR scheduler proposed by Noam

    Ref:
        "Attention Is All You Need", https://arxiv.org/pdf/1706.03762.pdf

    FIXME(kamo): PyTorch doesn't provide _LRScheduler as public class,
     thus the behaviour isn't guaranteed at forward PyTorch version.

    NOTE(kamo): The "model_size" in original implementation is derived from
     the model, but in this implementation, this parameter is a constant value.
     You need to change it if the model is changed.

    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model_size: Union[int, float] = 320,
        warmup_steps: Union[int, float] = 25000,
        factor: float = 1.0,
    ):
        if LooseVersion(torch.__version__) < LooseVersion("1.1.0"):
            raise NotImplementedError(
                f"Require PyTorch>=1.1.0: {torch.__version__}")
        assert check_argument_types()
        self.optimizer = optimizer
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        self._step = 0
        self.factor = factor
        self._rate = 0

    @property
    def param_groups(self):
        """Return param_groups."""
        return self.optimizer.param_groups

    def step(self):
        """Update parameters and rate."""
        self._step += 1
        logging.info(f"current step: {self._step}")
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """Implement `lrate` above."""
        if step is None:
            step = self._step
        return (self.factor * self.model_size**(-0.5) *
                min(step**(-0.5), step * self.warmup_steps**(-1.5)))

    def zero_grad(self):
        """Reset gradient."""
        self.optimizer.zero_grad()

    def state_dict(self):
        """Return state_dict."""
        return {
            "_step": self._step,
            "warmup": self.warmup_steps,
            "factor": self.factor,
            "model_size": self.model_size,
            "_rate": self._rate,
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        """Load state_dict."""
        for key, value in state_dict.items():
            if key == "optimizer":
                self.optimizer.load_state_dict(state_dict["optimizer"])
            else:
                setattr(self, key, value)

    def __repr__(self):
        return (f"{self.__class__.__name__}(model_size={self.model_size}, "
                f"warmup_steps={self.warmup_steps})")
