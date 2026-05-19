# learn2metalearn

An experimental meta-learning system that trains a **Discriminator** to recognize good weight transitions in a transformer language model, then uses it to guide optimization via gradient ascent interventions.

The core question: can a learned critic find improvement directions that standard gradient descent misses?

## Origin

This project grew out of [Stateweaver](./Stateweaver/), where the same idea was first explored using a 3-network setup (Generator, Refiner, Discriminator). Here the architecture is simplified to 2 networks: the Discriminator alone guides updates by serving as a differentiable objective for gradient ascent through the Generator's weights. No separate refiner network is needed.

## Concept

Training proceeds in three phases:

1. **Baseline (0–20K steps)** — Standard supervised next-token prediction on the Generator.
2. **Discriminator training (20K–35K steps)** — The Discriminator learns to classify weight-space transitions as "real" (loss went down) or "fake" (loss went up). Deliberately bad weight configurations are synthesized via gradient descent through the Discriminator to sharpen its signal.
3. **Intervention phase (35K+ steps)** — Every 1000 steps, gradient ascent through the Discriminator nudges the Generator's weights toward configurations it scores as improving. The intervention is committed only if validation loss actually decreases by a margin of ≥ 0.02.

## Architecture

**Generator** (`model.py`)
- Byte-level transformer (vocab size 256, no tokenizer needed)
- 8 layers, 8 attention heads, d_model=256, context length 128
- GPT-2-style initialization, weight tying, Flash Attention via PyTorch SDPA

**Discriminator** (`model.py`)
- `LayerEncoder` — cross-attention that compresses each of the 4 weight matrices per layer (qkv, out_proj, ff_up, ff_down) into learned token representations
- Small transformer encoder over before/after weight token sequences
- `ScoreFunnel` — per-token MLP + learned aggregation → scalar score
- Spectral normalization for adversarial training stability


## Usage

**Train**
```bash
python train.py
```
Automatically resumes from the latest checkpoint in `checkpoints/` if one exists.

**Generate text**
```bash
python inference.py --prompt "The history of" --tokens 500 --temperature 0.8

# From a specific checkpoint
python inference.py -c base_checkpoints/checkpoint_latest.pt -p "The history of"
```

**Plot training curves**
```bash
python tools/plot_loss.py
```

## Status & Known Problem

The approach is theoretically sound but has stalled on a fundamental issue: **the Discriminator's task is too hard for it to converge reliably.** Classifying weight-space transitions — deciding from raw parameter deltas whether a model improved — is an extremely high-dimensional and under-constrained problem. Without a well-converged discriminator, gradient ascent through it produces noisy directions that do not consistently reduce validation loss.

A possible solution to the issue may be some type of modularization of the network that would allow the problem to be broken down in to smaller chunks.
