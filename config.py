from dataclasses import dataclass
import torch


@dataclass
class Config:
    # Tokenizer — fixed at 256 for byte-level encoding
    vocab_size: int = 256

    # Architecture
    d_model: int = 256
    n_heads: int = 8        # head_dim = 32
    n_layers: int = 8
    dropout: float = 0.1

    # Sequence length — attention window and training chunk size
    context_length: int = 128

    # Training
    lr: float = 1.0           # Prodigy scale factor — keep at 1.0
    max_iters: int = 500_000
    eval_interval: int = 250
    eval_iters: int = 50      # number of context-length chunks evaluated per val check
    grad_clip: float = 1.0

    # Dataset — "wikitext103", "fineweb_edu", or "oasst2"
    dataset: str = "fineweb_edu"

    # Checkpointing
    checkpoint_interval: int = 5000
    checkpoint_dir: str = "checkpoints"

    # Inference
    max_new_tokens: int = 200
    temperature: float = 0.8

    # Batched training
    batch_size: int = 256

    # Refiner architecture
    lora_rank: int = 8
    refiner_d: int = 128      # bottleneck dimension
    refiner_layers: int = 2
    refiner_heads: int = 4
    noise_std: float = 0.1    # std of noise injected at bottleneck

    # Intervention
    n_samples: int = 1000                  # LoRA candidates per intervention
    top_k_frac: float = 0.1               # top/bottom 10% → winners/losers
    n_refiner_train: int = 32             # fresh samples for refiner gradient step
    intervention_interval: int = 500      # normal training steps between interventions
    refiner_start_iter: int = 5_000       # normal steps before first intervention
    disc_warmup_interventions: int = 5    # interventions before refiner starts training

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
