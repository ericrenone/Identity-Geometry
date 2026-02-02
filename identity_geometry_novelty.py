import torch
from typing import Optional
from .fisher import FisherInfo
from .kl import KLDivergence
from .config import NoveltyConfig

class NoveltyFunctional:
    """
    Minimal novelty functional for LLM outputs:
    Novelty = KL(p || uniform) / attention_proxy
    """

    def __init__(self, config: Optional[NoveltyConfig] = None):
        self.config = config or NoveltyConfig()
        self.fisher = FisherInfo(self.config.fisher)
        self.kl_div = KLDivergence(self.config.kl)

    @torch.no_grad()
    def compute(self, text: str, model: torch.nn.Module, tokenizer) -> float:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(model.device)

        logits = model(**inputs).logits[:, -1, :]
        kl = self.kl_div.vs_uniform(logits)

        attention_proxy = (inputs.input_ids.shape[1] / self.config.attention_normalizer) + 0.1

        return kl / attention_proxy
