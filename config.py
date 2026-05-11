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

    # Discriminator / meta-network architecture (shared encoder)
    refiner_d: int = 128      # bottleneck dimension
    refiner_layers: int = 3
    refiner_heads: int = 4
    encoder_n_out: int = 4    # tokens produced per weight matrix by cross-attention

    # Intervention
    intervention_interval: int = 1000      # normal training steps between interventions
    intervention_commit_margin: float = 0.02  # commit if cand_loss <= baseline_loss + margin
    disc_ascent_steps: int = 50            # gradient ascent steps through discriminator per intervention
    disc_ascent_lr: float = 1e-3           # step size (applied to unit-normalised gradient)

    # Discriminator training on GD transitions
    disc_start_iter: int = 20_000          # when discriminator training begins
    disc_snap_min: int = 16               # minimum snapshots before discriminator training begins
    disc_n_pairs: int = 64               # random pairs sampled per discriminator update
    disc_train_interval: int = 182         # steps between discriminator training steps
    disc_bad_sample_start_iter: int = 30_000  # when bad-sample generation via disc descent begins
    disc_bad_sample_interval: int = 440       # steps between bad sample generation runs
    disc_bad_sample_margin: float = 1.02      # bad sample must be this much worse than snapshot loss
    bad_sample_buffer_size: int = 256         # explicit (bad, good) pairs stored for disc training
    disc_bad_n_pairs: int = 1               # bad sample pairs included per disc training step
    # Full model snapshots — used for discriminator training and intervention anchors
    model_snap_buffer_size: int = 1024
    model_snap_interval: int = 100
    refiner_start_iter: int = 35_000      # when interventions (Phase 3) begin

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
