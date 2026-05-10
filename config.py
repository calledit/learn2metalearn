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
    refiner_d: int = 128      # bottleneck dimension
    refiner_layers: int = 2
    refiner_heads: int = 4
    noise_std: float = 0.1    # std of noise injected at bottleneck
    encoder_n_out: int = 4    # tokens produced per weight matrix by cross-attention

    # Intervention
    n_samples: int = 500                  # refiner candidates per intervention
    refiner_warmup_batch: int = 128        # real snapshots sampled from refiner_buf per warmup step
    intervention_interval: int = 1000      # normal training steps between interventions
    intervention_commit_margin: float = 0.1  # commit if cand_loss <= baseline_loss + margin

    # Discriminator training on GD transitions
    disc_start_iter: int = 20_000          # when discriminator training begins
    disc_buffer_size: int = 2048            # rolling window of snapshots per per-layer buffer
    disc_buffer_start: int = 16           # minimum entries before discriminator training begins
    disc_n_pairs: int = 128               # random pairs sampled per discriminator update
    disc_snapshot_interval: int = 10      # steps between weight snapshots per layer cycle
    disc_train_interval: int = 50        # steps between discriminator training steps
    # Refiner buffer — real weight snapshots offset from disc snapshots by interval//2
    refiner_buffer_size: int = 512
    refiner_buffer_start: int = 8
    # Full model snapshots for ref_gen training
    model_snap_buffer_size: int = 10
    model_snap_interval: int = 50

    # Refiner warmup (Phase 2.5) — refiner trains against discriminator, no weight commits
    refiner_warmup_start_iter: int = 30_000
    # ref_gen_loss anchors the refiner toward proposals that actually reduce generator loss,
    # preventing the refiner from constantly wandering from bad to bad guided by negative feadback from the discriminator.
    # The risk of too much ref_gen signal is that the refiner learns to replicate gradient
    # descent on the snapshot batch — effectively running GD many times on the same data,
    # which drives weights into extreme local optima. Keep scale small. May be usefull to have it higher initaully durring warmup
    ref_gen_loss_scale: float = 0.01        # loss multiplier — reduces gradient magnitude
    ref_gen_loss_end_iter: int = 60_000   # stop gen-loss refiner training at this step (0 = never)
    refiner_start_iter: int = 45_000      # when interventions (Phase 3) begin

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
