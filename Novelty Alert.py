#!/usr/bin/env python3
import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import warnings
import csv
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, logging

# 1. Silencing library noise
warnings.filterwarnings("ignore", message=".*loss_type.*")
logging.set_verbosity_error()

@dataclass
class NoveltyConfig:
    attention_normalizer: float = 512.0
    eps: float = 1e-6
    target_layers: List[str] = field(default_factory=lambda: ["lm_head"])
    novelty_threshold: float = 0.5  # The red line level

class NoveltyEngine:
    def __init__(self, config: NoveltyConfig):
        self.config = config

    def compute(self, text: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Dict[str, float]:
        device = next(model.parameters()).device
        inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits[:, -1, :]
            log_p = F.log_softmax(logits, dim=-1)
            log_uniform = -torch.log(torch.tensor(logits.shape[-1], device=device))
            kl = F.kl_div(log_uniform.expand_as(log_p), log_p, log_target=True, reduction="batchmean")

        model.eval()
        with torch.enable_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            model.zero_grad(set_to_none=True)
            outputs.loss.backward()
            fisher_trace = sum(p.grad.pow(2).sum().item() for n, p in model.named_parameters() 
                               if any(t in n for t in self.config.target_layers) and p.grad is not None)
            model.zero_grad(set_to_none=True)

        token_count = inputs.input_ids.shape[1]
        novelty = (float(kl) * fisher_trace) / ((token_count / self.config.attention_normalizer) + self.config.eps)

        return {"novelty": novelty, "kl": float(kl), "fisher": fisher_trace, "tokens": token_count}

def run_simulation_2d(texts: List[str], model_name="gpt2", export_file="novelty_results.csv"):
    print(f"--- Starting Analysis with {model_name} ---")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = NoveltyConfig()
    engine = NoveltyEngine(config)

    data_log = []
    metrics = {"novelty": [], "kl": [], "fisher": []}
    labels = [f"T{i+1}" for i in range(len(texts))]

    plt.ion()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    plt.subplots_adjust(hspace=0.4)

    # Raw strings r"" used to prevent SyntaxWarning for LaTeX
    titles = [r"Novelty Functional $\Phi$", "KL Divergence", "Fisher Trace"]
    colors = ["teal", "orange", "purple"]

    for i, text in enumerate(texts):
        res = engine.compute(text, model, tokenizer)
        is_outlier = res["novelty"] > config.novelty_threshold
        
        # Log data
        metrics["novelty"].append(res["novelty"])
        metrics["kl"].append(res["kl"])
        metrics["fisher"].append(res["fisher"])
        data_log.append({"text": text, "is_alert": is_outlier, **res})

        # Console Alert Logic
        status = " [!] NOVELTY ALERT" if is_outlier else " [.] Normal"
        print(f"Step {i+1}: {text[:30]}... | Score: {res['novelty']:.4f}{status}")

        # Plotting Logic
        for ax, key, title, color in zip([ax1, ax2, ax3], ["novelty", "kl", "fisher"], titles, colors):
            ax.cla()
            ax.plot(labels[:i+1], metrics[key], marker='o', color=color, linewidth=2)
            ax.set_title(title, fontweight='bold')
            ax.grid(alpha=0.3)
            
            if key == "novelty":
                ax.axhline(y=config.novelty_threshold, color='red', linestyle='--', alpha=0.6, label='Threshold')
                ax.legend(loc='upper left')

        plt.xticks(rotation=45)
        plt.draw()
        plt.pause(0.1)

    # Export to CSV with Alert column
    with open(export_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["text", "novelty", "kl", "fisher", "tokens", "is_alert"])
        writer.writeheader()
        writer.writerows(data_log)

    plt.ioff()
    print(f"\nSimulation Finished. Results saved to {export_file}")
    plt.show()

if __name__ == "__main__":
    dataset = [
        "The quick brown fox jumps over the lazy dog.",
        "A sequence of random numbers: 4, 8, 15, 16, 23, 42.",
        "I like apples.",
        "The Schrödinger equation describes the wave function of a quantum system.",
        "Sphinx of black quartz, judge my vow."
    ]
    run_simulation_2d(dataset)
