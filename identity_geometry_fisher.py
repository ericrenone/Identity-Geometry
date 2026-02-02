import torch
from typing import List, Optional
from .config import FisherConfig, FisherMethod

class FisherInfo:
    """
    Minimal diagonal Fisher trace estimator.
    Computes E[(d log p / d θ)^2] for loss w.r.t model parameters.
    """

    def __init__(self, config: Optional[FisherConfig] = None):
        self.config = config or FisherConfig()
        self.device = torch.device(
            self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

    @torch.no_grad()
    def trace(self, model: torch.nn.Module, texts: List[str], tokenizer) -> float:
        if self.config.method != FisherMethod.DIAGONAL:
            raise NotImplementedError("Only diagonal Fisher trace implemented.")

        model.eval().to(self.device)
        total_trace = 0.0
        n = len(texts)

        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                return_attention_mask=False
            ).to(self.device)

            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

            model.zero_grad(set_to_none=True)
            loss.backward()

            trace = sum(p.grad.pow(2).sum().item() for p in model.parameters() if p.grad is not None)
            total_trace += trace

        return total_trace / n if n > 0 else 0.0
