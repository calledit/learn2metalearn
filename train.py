import csv
import glob
import itertools
import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from prodigyopt import Prodigy

from config import Config
from model import Generator
from data import build_dataset
from inference import generate


# ──────────────────────────────────────────────────────────────── checkpointing

def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    import re
    files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pt"))
    if not files:
        return None
    def _step(f):
        m = re.search(r"checkpoint_(\d+)\.pt", os.path.basename(f))
        return int(m.group(1)) if m else -1
    return max(files, key=_step)


def save_checkpoint(model, optimizer, scheduler, grad_updates, cfg):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    data = {
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "grad_updates":    grad_updates,
        "cfg":             cfg,
    }
    path = os.path.join(cfg.checkpoint_dir, f"checkpoint_{grad_updates:07d}.pt")
    torch.save(data, path)
    print(f"  [ckpt] step {grad_updates} → {path}")


# ─────────────────────────────────────────────────────────────────── evaluation

@torch.no_grad()
def evaluate(model: Generator, val_data: torch.Tensor, cfg: Config) -> float:
    model.eval()
    device = next(model.parameters()).device
    total  = 0.0
    n      = 0
    for i in range(0, len(val_data) - cfg.context_length - 1, cfg.context_length):
        x = val_data[i     : i + cfg.context_length    ].unsqueeze(0).to(device)
        y = val_data[i + 1 : i + cfg.context_length + 1].unsqueeze(0).to(device)
        logits = model(x)
        total += F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1)).item()
        n += 1
        if n >= cfg.eval_iters:
            break
    model.train()
    return total / max(n, 1)


# ──────────────────────────────────────────────────────────────────── training

def train():
    cfg    = Config()
    device = torch.device(cfg.device)
    print(f"Device: {device}")

    train_dataset, val_data, tokenizer = build_dataset(cfg)
    print(f"Vocab: {cfg.vocab_size} | context_length: {cfg.context_length}")

    loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=False)
    print(f"batch_size={cfg.batch_size}")

    model = Generator(cfg).to(device)

    n_param = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_param:,}")

    optimizer = Prodigy(
        model.parameters(), lr=cfg.lr, weight_decay=0.1,
        safeguard_warmup=True, use_bias_correction=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_iters)

    # ── resume from checkpoint ────────────────────────────────────────────────
    grad_updates = 0
    ckpt_path = find_latest_checkpoint(cfg.checkpoint_dir)
    if ckpt_path:
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt["model_state"]
        if any(k.startswith("_orig_mod.") for k in state_dict):
            state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        grad_updates = ckpt["grad_updates"]
        print(f"  resumed at step {grad_updates}")
    else:
        print("No checkpoint found — starting from scratch")

    val_data = val_data.to(device)

    # ── logging setup ─────────────────────────────────────────────────────────
    log_path   = os.path.join(cfg.checkpoint_dir, "training_log.csv")
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    write_header = not os.path.exists(log_path)
    log_file   = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if write_header:
        log_writer.writerow(["step", "train_loss", "val_loss", "lr", "elapsed_s", "tok_per_s"])

    last_checkpoint_saved = grad_updates // cfg.checkpoint_interval
    t0              = time.time()
    t_last_log      = t0
    train_loss_sum  = 0.0
    train_loss_count = 0
    tokens_since_log = 0

    for epoch in itertools.count(1):
        for batch in loader:
            if grad_updates >= cfg.max_iters:
                break

            # batch: [B, context_length + 1]
            batch = batch.to(device)
            x = batch[:, :-1]   # [B, T]
            y = batch[:, 1:]    # [B, T]

            logits = model(x)   # [B, T, vocab_size]
            loss   = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))

            if torch.isnan(loss):
                print(f"WARNING: NaN loss at step {grad_updates + 1}, skipping batch")
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            train_loss_sum   += loss.item()
            train_loss_count += 1
            tokens_since_log += batch.size(0) * cfg.context_length
            grad_updates     += 1

            if grad_updates % cfg.eval_interval == 0:
                avg_train_loss  = train_loss_sum / train_loss_count
                train_loss_sum  = 0.0
                train_loss_count = 0

                val_loss  = evaluate(model, val_data, cfg)
                elapsed   = time.time() - t0
                lr        = scheduler.get_last_lr()[0]
                tok_per_s = tokens_since_log / max(time.time() - t_last_log, 1e-9)
                t_last_log      = time.time()
                tokens_since_log = 0

                model.eval()
                sample = generate(model, tokenizer, cfg, prompt="The history of", max_new_tokens=20)
                model.train()
                sample_line = " ".join(sample.split())

                print(
                    f"step {grad_updates:7d}/{cfg.max_iters} | "
                    f"t_loss {avg_train_loss:.4f} | "
                    f"v_loss {val_loss:.4f} | "
                    f"lr {lr:.2e} | "
                    f"{tok_per_s:,.0f} tok/s | "
                    f"time: {elapsed:.0f}s | "
                    f"samp: {sample_line}"
                )
                log_writer.writerow([
                    grad_updates,
                    f"{avg_train_loss:.6f}",
                    f"{val_loss:.6f}",
                    f"{lr:.6e}",
                    f"{elapsed:.1f}",
                    f"{tok_per_s:.0f}",
                ])
                log_file.flush()

            current_interval = grad_updates // cfg.checkpoint_interval
            if current_interval > last_checkpoint_saved:
                save_checkpoint(model, optimizer, scheduler, grad_updates, cfg)
                last_checkpoint_saved = current_interval

        if grad_updates >= cfg.max_iters:
            break

    save_checkpoint(model, optimizer, scheduler, grad_updates, cfg)
    log_file.close()
    print(f"\nTraining done in {time.time() - t0:.0f}s")
    return model, tokenizer, cfg


if __name__ == "__main__":
    train()
