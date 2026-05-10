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
    refiner_layers: int = 2
    refiner_heads: int = 4
    encoder_n_out: int = 4    # tokens produced per weight matrix by cross-attention

    # Intervention
    intervention_interval: int = 1000      # normal training steps between interventions
    intervention_commit_margin: float = 0.1  # commit if cand_loss <= baseline_loss + margin
    disc_ascent_steps: int = 10           # gradient ascent steps through discriminator per intervention
    disc_ascent_lr: float = 1e-3          # step size (applied to unit-normalised gradient)

    # Discriminator training on GD transitions
    disc_start_iter: int = 20_000          # when discriminator training begins
    disc_buffer_size: int = 2048            # rolling window of snapshots per per-layer buffer
    disc_buffer_start: int = 16            # minimum entries before discriminator training begins
    disc_n_pairs: int = 128                # random pairs sampled per discriminator update
    disc_snapshot_interval: int = 10       # steps between weight snapshots per layer cycle
    disc_train_interval: int = 50          # steps between discriminator training steps
    disc_bad_sample_start_iter: int = 30_000  # when bad-sample generation via disc descent begins
    # Full model snapshots — used as stable "before" starting points for disc descent bad samples
    model_snap_buffer_size: int = 10
    model_snap_interval: int = 50
    refiner_start_iter: int = 45_000       # when interventions (Phase 3) begin

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
