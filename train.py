import csv
import glob
import importlib
import itertools
import os
import time
from collections import deque
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from prodigyopt import Prodigy

import config as _config_module
from config import Config
from model import Generator, Discriminator, get_block_linears, disc_hinge_loss
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


def save_checkpoint(model, optimizer, scheduler,
                    discriminator, disc_opt,
                    grad_updates, intervention_count, cfg):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    data = {
        "model_state":      model.state_dict(),
        "optimizer_state":  optimizer.state_dict(),
        "scheduler_state":  scheduler.state_dict(),
        "disc_state":       discriminator.state_dict(),
        "disc_opt":         disc_opt.state_dict(),
        "grad_updates":     grad_updates,
        "intervention_count": intervention_count,
        "cfg":              cfg,
    }
    path = os.path.join(cfg.checkpoint_dir, f"checkpoint_{grad_updates:07d}.pt")
    torch.save(data, path)
    print(f"  [ckpt] step {grad_updates} → {path}")


# ─────────────────────────────────────────────────────────────────── evaluation

@torch.no_grad()
def evaluate(model: Generator, val_data: torch.Tensor, cfg: Config) -> float:
    model.eval()
    device = next(model.parameters()).device
    total, n = 0.0, 0
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


# ─────────────────────────────────────── discriminator helpers

def _all_layer_weights(snap_state: dict, n_layers: int, device) -> list:
    """Extract all transformer block weights from a state dict as list[list[Tensor]]."""
    result = []
    for li in range(n_layers):
        names = [
            f"blocks.{li}.attn.qkv.weight",
            f"blocks.{li}.attn.out_proj.weight",
            f"blocks.{li}.ff.net.0.weight",
            f"blocks.{li}.ff.net.2.weight",
        ]
        result.append([snap_state[n].unsqueeze(0).to(device) for n in names])
    return result


def warmup_discriminator(discriminator, disc_opt, before_weights, after_weights):
    """
    before_weights / after_weights: list of n_layers entries, each a list of
    4 [B, out_i, in_i] tensors. before has higher loss, after has lower loss.
    Real: [before → after].  Fake: [after → before].
    """
    real_scores = discriminator(before_weights, after_weights)
    fake_scores = discriminator(after_weights, before_weights)
    d_loss = disc_hinge_loss(real_scores, fake_scores)
    disc_opt.zero_grad()
    d_loss.backward()
    disc_opt.step()
    return d_loss.item()


def _train_disc_from_snaps(snap_buf, bad_sample_buf, discriminator, disc_opt, device, cfg):
    """Sample pairs from full-model snapshots and explicit bad sample pairs."""
    buf = list(snap_buf)
    if len(buf) < 2:
        return None
    n = min(cfg.disc_n_pairs, len(buf) * (len(buf) - 1) // 2)
    idx_a = torch.randint(len(buf), (n,)).tolist()
    idx_b = torch.randint(len(buf), (n,)).tolist()
    before_snaps, after_snaps = [], []
    for a, b in zip(idx_a, idx_b):
        lo, hi = (a, b) if buf[a][1] <= buf[b][1] else (b, a)
        after_snaps.append(buf[lo][0])   # lower loss → after
        before_snaps.append(buf[hi][0])  # higher loss → before

    # Append explicit (bad, good) pairs from bad sample buffer
    if bad_sample_buf:
        bad_buf = list(bad_sample_buf)
        n_bad = min(cfg.disc_bad_n_pairs, len(bad_buf))
        bad_idx = torch.randint(len(bad_buf), (n_bad,)).tolist()
        for i in bad_idx:
            bad_state, good_state = bad_buf[i]
            before_snaps.append(bad_state)   # bad → before
            after_snaps.append(good_state)   # good → after

    def _batch(snap_list):
        batched = []
        for li in range(cfg.n_layers):
            names = [
                f"blocks.{li}.attn.qkv.weight",
                f"blocks.{li}.attn.out_proj.weight",
                f"blocks.{li}.ff.net.0.weight",
                f"blocks.{li}.ff.net.2.weight",
            ]
            batched.append([
                torch.stack([s[name] for s in snap_list]).to(device) for name in names
            ])
        return batched

    return warmup_discriminator(discriminator, disc_opt, _batch(before_snaps), _batch(after_snaps))


# ────────────────────────────────────────────────────────────────── intervention

def run_intervention_disc(model, discriminator, val_data, cfg, model_snap_buf=None):
    """
    Gradient ascent through the discriminator across all transformer layers.
    Uses a past snapshot as 'before' and current weights (all layers) as 'after',
    pushing 'after' further in the direction the discriminator scores as improving.
    Commits if loss improves.
    """
    device = next(model.parameters()).device
    model.eval()

    start  = torch.randint(0, len(val_data) - cfg.context_length - 1, (1,)).item()
    rank_x = val_data[start     : start + cfg.context_length    ].unsqueeze(0)
    rank_y = val_data[start + 1 : start + cfg.context_length + 1].unsqueeze(0)

    all_block_linears = [get_block_linears(model.blocks[li]) for li in range(cfg.n_layers)]

    # after = current weights across all layers (the good side to push further)
    # before = past snapshot as historical anchor; fall back to current if buffer empty
    after = [
        [lin.weight.detach().unsqueeze(0).clone().requires_grad_(True) for lin in lins]
        for lins in all_block_linears
    ]
    if model_snap_buf and len(model_snap_buf) > 0:
        snap, _ = list(model_snap_buf)[torch.randint(len(model_snap_buf), (1,)).item()]
        before = _all_layer_weights(snap, cfg.n_layers, device)
    else:
        before = [[w.detach().clone() for w in layer] for layer in after]

    for p in discriminator.parameters():
        p.requires_grad_(False)

    flat_after = [w for layer in after for w in layer]
    for _ in range(cfg.disc_ascent_steps):
        after_nested = [flat_after[li * 4:(li + 1) * 4] for li in range(cfg.n_layers)]
        score = discriminator(before, after_nested)
        grads = torch.autograd.grad(score.mean(), flat_after)
        total_norm = sum(g.norm() ** 2 for g in grads) ** 0.5 + 1e-8
        with torch.no_grad():
            flat_after = [a + cfg.disc_ascent_lr * g / total_norm
                          for a, g in zip(flat_after, grads)]
        flat_after = [a.requires_grad_(True) for a in flat_after]
    after = [flat_after[li * 4:(li + 1) * 4] for li in range(cfg.n_layers)]

    for p in discriminator.parameters():
        p.requires_grad_(True)

    with torch.inference_mode():
        baseline_loss = F.cross_entropy(
            model(rank_x).reshape(-1, cfg.vocab_size), rank_y.reshape(-1)
        ).item()

    original = [[lin.weight.data.clone() for lin in lins] for lins in all_block_linears]
    with torch.inference_mode():
        for li, lins in enumerate(all_block_linears):
            for lin, a in zip(lins, after[li]):
                lin.weight.data.copy_(a.detach().squeeze(0))
        cand_loss = F.cross_entropy(
            model(rank_x).reshape(-1, cfg.vocab_size), rank_y.reshape(-1)
        ).item()
        for li, lins in enumerate(all_block_linears):
            for lin, orig in zip(lins, original[li]):
                lin.weight.data.copy_(orig)

    with torch.no_grad():
        r_score = discriminator(before, after).mean().item()

    committed = False
    if cand_loss <= baseline_loss + cfg.intervention_commit_margin:
        with torch.no_grad():
            for li, lins in enumerate(all_block_linears):
                for lin, a in zip(lins, after[li]):
                    lin.weight.data.copy_(a.detach().squeeze(0))
        committed = True

    model.train()
    return cand_loss, baseline_loss, r_score, committed


# ──────────────────────────────────────────────────────────────────── training

def train():
    cfg    = Config()
    device = torch.device(cfg.device)
    _config_mtime = os.path.getmtime(_config_module.__file__)
    print(f"Device: {device}")

    train_dataset, val_data, tokenizer = build_dataset(cfg)
    print(f"Vocab: {cfg.vocab_size} | context_length: {cfg.context_length}")

    loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=False)
    print(f"batch_size={cfg.batch_size}")

    model         = Generator(cfg).to(device)
    discriminator = Discriminator(cfg).to(device)

    n_gen   = sum(p.numel() for p in model.parameters())
    n_disc  = sum(p.numel() for p in discriminator.parameters())
    print(f"Generator: {n_gen:,} params | Discriminator: {n_disc:,}")

    optimizer = Prodigy(model.parameters(), lr=cfg.lr, weight_decay=0.1,
                        safeguard_warmup=True, use_bias_correction=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_iters)

    disc_opt    = Prodigy(discriminator.parameters(), lr=cfg.lr, weight_decay=0.01,
                          safeguard_warmup=True, use_bias_correction=True)

    # ── Resume from checkpoint ────────────────────────────────────────────────
    grad_updates        = 0
    intervention_count  = 0
    model_snap_buf    = deque(maxlen=cfg.model_snap_buffer_size)  # (state_dict, loss)
    bad_sample_buf    = deque(maxlen=cfg.bad_sample_buffer_size)  # (bad_state_dict, good_state_dict)
    warmup_disc_sum   = 0.0
    warmup_disc_count = 0
    bad_sample_sum    = 0.0
    bad_sample_count  = 0

    ckpt_path = find_latest_checkpoint(cfg.checkpoint_dir)
    if ckpt_path:
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        def _load(model, key):
            sd = ckpt[key]
            if any(k.startswith("_orig_mod.") for k in sd):
                sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
            model.load_state_dict(sd)

        _load(model, "model_state")
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        if "disc_state" in ckpt:
            try:
                _load(discriminator, "disc_state")
                disc_opt.load_state_dict(ckpt["disc_opt"])
            except RuntimeError as e:
                print(f"  [warn] discriminator checkpoint incompatible, starting fresh: {e}")
        grad_updates       = ckpt["grad_updates"]
        intervention_count = ckpt.get("intervention_count", 0)
        print(f"  resumed at step {grad_updates}, interventions {intervention_count}")
    else:
        print("No checkpoint found — starting from scratch")

    val_data = val_data.to(device)

    # ── Logging ───────────────────────────────────────────────────────────────
    log_path   = os.path.join(cfg.checkpoint_dir, "training_log.csv")
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    write_header = not os.path.exists(log_path)
    log_file   = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if write_header:
        log_writer.writerow([
            "step", "train_loss", "val_loss", "lr", "elapsed_s", "tok_per_s",
            "disc_loss", "interventions", "best_inter_loss", "bad_sample_loss",
        ])

    last_checkpoint_saved = grad_updates // cfg.checkpoint_interval
    t0               = time.time()
    t_last_log       = t0
    train_loss_sum   = 0.0
    train_loss_count = 0
    tokens_since_log = 0
    # Intervention stats (averaged over the eval interval)
    inter_best_sum   = 0.0
    inter_r_sum      = 0.0
    inter_count_log  = 0

    for epoch in itertools.count(1):
        for batch in loader:
            if grad_updates >= cfg.max_iters:
                break

            batch = batch.to(device)
            x = batch[:, :-1]
            y = batch[:, 1:]

            logits = model(x)
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

            # ── Discriminator training on GD transitions ─────────────────────
            if grad_updates >= cfg.disc_start_iter:
                if grad_updates % cfg.model_snap_interval == 0:
                    model_snap_buf.append((
                        {n: p.detach().cpu().clone() for n, p in model.named_parameters()},
                        loss.item(),
                    ))

                if grad_updates % cfg.disc_train_interval == 0 and len(model_snap_buf) >= cfg.disc_snap_min:
                    result = _train_disc_from_snaps(model_snap_buf, bad_sample_buf, discriminator, disc_opt, device, cfg)
                    if result is not None:
                        warmup_disc_sum   += result
                        warmup_disc_count += 1

                if (grad_updates >= cfg.disc_bad_sample_start_iter
                        and grad_updates % cfg.disc_bad_sample_interval == 0
                        and len(model_snap_buf) >= cfg.disc_snap_min):
                        _snap_state, _snap_loss = list(model_snap_buf)[torch.randint(len(model_snap_buf), (1,)).item()]
                        _before_bad = _all_layer_weights(_snap_state, cfg.n_layers, device)
                        _flat_bad = [w.clone().requires_grad_(True)
                                     for layer in _before_bad for w in layer]
                        for p in discriminator.parameters():
                            p.requires_grad_(False)
                        for _ in range(cfg.disc_ascent_steps):
                            _bad_nested = [_flat_bad[li * 4:(li + 1) * 4] for li in range(cfg.n_layers)]
                            _score = discriminator(_before_bad, _bad_nested)
                            _grads = torch.autograd.grad(_score.mean(), _flat_bad)
                            _norm  = sum(g.norm() ** 2 for g in _grads) ** 0.5 + 1e-8
                            with torch.no_grad():
                                _flat_bad = [a - cfg.disc_ascent_lr * g / _norm
                                             for a, g in zip(_flat_bad, _grads)]
                            _flat_bad = [a.requires_grad_(True) for a in _flat_bad]
                        for p in discriminator.parameters():
                            p.requires_grad_(True)

                        _all_lins = [get_block_linears(model.blocks[li]) for li in range(cfg.n_layers)]
                        _orig_all = [[lin.weight.data.clone() for lin in lins] for lins in _all_lins]
                        with torch.no_grad():
                            for li, lins in enumerate(_all_lins):
                                for lin, a in zip(lins, _flat_bad[li * 4:(li + 1) * 4]):
                                    lin.weight.data.copy_(a.detach().squeeze(0))
                            model.eval()
                            _cand_loss = F.cross_entropy(
                                model(x).reshape(-1, cfg.vocab_size), y.reshape(-1)
                            ).item()
                            model.train()
                            for li, lins in enumerate(_all_lins):
                                for lin, orig in zip(lins, _orig_all[li]):
                                    lin.weight.data.copy_(orig)

                        if _cand_loss > _snap_loss * cfg.disc_bad_sample_margin:
                            bad_state = dict(_snap_state)
                            for li in range(cfg.n_layers):
                                _names = [
                                    f"blocks.{li}.attn.qkv.weight",
                                    f"blocks.{li}.attn.out_proj.weight",
                                    f"blocks.{li}.ff.net.0.weight",
                                    f"blocks.{li}.ff.net.2.weight",
                                ]
                                for name, a in zip(_names, _flat_bad[li * 4:(li + 1) * 4]):
                                    bad_state[name] = a.detach().squeeze(0).cpu().clone()
                            bad_sample_buf.append((bad_state, _snap_state))
                            bad_sample_sum   += _cand_loss
                            bad_sample_count += 1

            # ── Phase 3: intervention ─────────────────────────────────────────
            if (grad_updates >= cfg.refiner_start_iter
                    and grad_updates % cfg.intervention_interval == 0):
                best_loss, baseline_loss, r_score, committed = run_intervention_disc(
                    model, discriminator, val_data, cfg, model_snap_buf,
                )
                intervention_count += 1
                inter_best_sum  += best_loss
                inter_r_sum     += r_score
                inter_count_log += 1
                print(
                    f"  [intervention {intervention_count}] "
                    f"cand={best_loss:.4f} baseline={baseline_loss:.4f} | "
                    f"r_score={r_score:.4f} | {'COMMITTED' if committed else 'skipped'}"
                )

            # ── Eval + log ────────────────────────────────────────────────────
            if grad_updates % cfg.eval_interval == 0:
                avg_train  = train_loss_sum / train_loss_count
                train_loss_sum, train_loss_count = 0.0, 0

                val_loss  = evaluate(model, val_data, cfg)
                elapsed   = time.time() - t0
                lr        = scheduler.get_last_lr()[0]
                tok_per_s = tokens_since_log / max(time.time() - t_last_log, 1e-9)
                t_last_log      = time.time()
                tokens_since_log = 0

                model.eval()
                sample = generate(model, tokenizer, cfg, prompt="The history of", max_new_tokens=20)
                model.train()

                avg_best     = inter_best_sum  / inter_count_log  if inter_count_log  else 0.0
                avg_disc     = warmup_disc_sum / warmup_disc_count if warmup_disc_count else 0.0
                avg_bad      = bad_sample_sum  / bad_sample_count  if bad_sample_count  else 0.0
                inter_best_sum  = 0.0
                inter_r_sum     = 0.0
                inter_count_log = 0
                warmup_disc_sum   = 0.0
                warmup_disc_count = 0
                bad_sample_sum   = 0.0
                bad_sample_count = 0

                print(
                    f"step {grad_updates:7d}/{cfg.max_iters} | "
                    f"t_loss {avg_train:.4f} | v_loss {val_loss:.4f} | "
                    f"lr {lr:.2e} | {tok_per_s:,.0f} tok/s | "
                    f"time: {elapsed:.0f}s | "
                    + (f"disc_loss {avg_disc:.4f} | " if avg_disc else "")
                    + (f"bad_sample {avg_bad:.4f} | " if avg_bad else "")
                    + f"samp: {' '.join(sample.split())}"
                )
                log_writer.writerow([
                    grad_updates,
                    f"{avg_train:.6f}",
                    f"{val_loss:.6f}",
                    f"{lr:.6e}",
                    f"{elapsed:.1f}",
                    f"{tok_per_s:.0f}",
                    f"{avg_disc:.6f}" if avg_disc else "",
                    intervention_count,
                    f"{avg_best:.6f}" if avg_best else "",
                    f"{avg_bad:.6f}"  if avg_bad  else "",
                ])
                log_file.flush()

            current_interval = grad_updates // cfg.checkpoint_interval
            if current_interval > last_checkpoint_saved:
                save_checkpoint(model, optimizer, scheduler,
                                discriminator, disc_opt,
                                grad_updates, intervention_count, cfg)
                last_checkpoint_saved = current_interval

                new_mtime = os.path.getmtime(_config_module.__file__)
                if new_mtime != _config_mtime:
                    _config_mtime = new_mtime
                    answer = input("  [config] config.py changed — reload? [y/N] ").strip().lower()
                    if answer == "y":
                        importlib.reload(_config_module)
                        cfg = _config_module.Config()
                        print("  [config] reloaded")

        if grad_updates >= cfg.max_iters:
            break

    save_checkpoint(model, optimizer, scheduler,
                    discriminator, disc_opt,
                    grad_updates, intervention_count, cfg)
    log_file.close()
    print(f"\nTraining done in {time.time() - t0:.0f}s")
    return model, tokenizer, cfg


if __name__ == "__main__":
    train()
