import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional
from .config import KLConfig

class KLDivergence:
    """
    KL divergence computations.
    Currently implemented: KL(p || uniform) on last-token softmax.
    """

    def __init__(self, config: Optional[KLConfig] = None):
        self.config = config or KLConfig()

    @staticmethod
    def multinomial(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-8) -> float:
        p = np.maximum(p, epsilon); p /= p.sum()
        q = np.maximum(q, epsilon); q /= q.sum()
        return float(np.sum(p * np.log(p / q)))

    @torch.no_grad()
    def vs_uniform(self, logits: torch.Tensor) -> float:
        vocab_size = logits.shape[-1]
        log_uniform = -torch.log(torch.tensor(vocab_size, device=logits.device, dtype=logits.dtype))
        log_p = F.log_softmax(logits, dim=-1)
        kl = F.kl_div(
            input=log_uniform.expand_as(log_p),
            target=log_p,
            log_target=True,
            reduction="batchmean"
        )
        return float(kl)
