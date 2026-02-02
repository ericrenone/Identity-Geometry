from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class FisherMethod(Enum):
    DIAGONAL = "diagonal"

@dataclass
class FisherConfig:
    method: FisherMethod = FisherMethod.DIAGONAL
    num_samples: int = 512
    batch_size: int = 32
    device: Optional[str] = None  # auto-detect if None

@dataclass
class KLConfig:
    epsilon: float = 1e-8
    prior_temperature: float = 1.0  # placeholder for empirical/unigram prior

@dataclass
class NoveltyConfig:
    fisher: FisherConfig = field(default_factory=FisherConfig)
    kl: KLConfig = field(default_factory=KLConfig)
    attention_normalizer: float = 512.0
