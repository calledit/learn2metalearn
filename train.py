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
from model import Generator, Refiner, Discriminator, get_block_linears, disc_hinge_loss
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
                    refiner, refiner_opt,
                    discriminator, disc_opt,
                    grad_updates, intervention_count, cfg):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    data = {
        "model_state":      model.state_dict(),
        "optimizer_state":  optimizer.state_dict(),
        "scheduler_state":  scheduler.state_dict(),
        "refiner_state":    refiner.state_dict(),
        "refiner_opt":      refiner_opt.state_dict(),
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


# ────────────────────────────────────────────────────────────────── intervention

def run_intervention(model, refiner, discriminator, refiner_opt, disc_opt,
                     val_data, layer_idx, intervention_count, cfg):
    device = next(model.parameters()).device
    model.eval()

    # One random sequence from val_data as ranking signal
    start   = torch.randint(0, len(val_data) - cfg.context_length - 1, (1,)).item()
    rank_x  = val_data[start     : start + cfg.context_length    ].unsqueeze(0)
    rank_y  = val_data[start + 1 : start + cfg.context_length + 1].unsqueeze(0)

    target_block  = model.blocks[layer_idx]
    block_linears = get_block_linears(target_block)  # [qkv, out_proj, ff_up, ff_down]
    layer_weights = [lin.weight for lin in block_linears]

    # ── Encode (once) ─────────────────────────────────────────────────────────
    with torch.no_grad():
        bw      = [w.detach().unsqueeze(0) for w in layer_weights]
        latent  = refiner.encode(bw).squeeze(0)   # [4, refiner_d]

    # ── Sample n_samples LoRA pairs, compute all deltas ───────────────────────
    with torch.no_grad():
        lora_pairs = refiner.sample(latent)
        # lora_pairs: 4 × (A[n,out,r], B[n,r,in])
        deltas = [torch.bmm(A, B) for A, B in lora_pairs]
        # deltas[i]: [n_samples, out_i, in_i]

    # ── Rank by loss (loop; each step only modifies one layer) ────────────────
    losses = torch.zeros(cfg.n_samples, device=device)
    with torch.inference_mode():
        for s in range(cfg.n_samples):
            for lin, delta in zip(block_linears, deltas):
                lin.weight.data.add_(delta[s])
            logits   = model(rank_x)
            losses[s] = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), rank_y.reshape(-1))
            for lin, delta in zip(block_linears, deltas):
                lin.weight.data.sub_(delta[s])

    k          = max(1, int(cfg.n_samples * cfg.top_k_frac))
    sorted_idx = losses.argsort()
    winner_idx = sorted_idx[:k]
    loser_idx  = sorted_idx[-k:]
    best_idx   = sorted_idx[0].item()

    # ── Build batched weight configs for discriminator ─────────────────────────
    def make_batch(indices):
        batch = []
        for lin, delta in zip(block_linears, deltas):
            W = lin.weight.detach()
            batch.append(W.unsqueeze(0) + delta[indices])  # [k, out_i, in_i]
        return batch

    winner_weights = make_batch(winner_idx)
    loser_weights  = make_batch(loser_idx)

    # ── Train discriminator ────────────────────────────────────────────────────
    real_scores = discriminator(winner_weights)
    fake_scores = discriminator(loser_weights)
    d_loss      = disc_hinge_loss(real_scores, fake_scores)
    disc_opt.zero_grad()
    d_loss.backward()
    disc_opt.step()

    # ── Train refiner (after discriminator warmup) ─────────────────────────────
    r_loss_val = 0.0
    if intervention_count >= cfg.disc_warmup_interventions:
        for p in discriminator.parameters():
            p.requires_grad_(False)

        bw_g    = [w.detach().unsqueeze(0) for w in layer_weights]
        latent_g = refiner.encode(bw_g).squeeze(0)               # [4, refiner_d]

        n_rt     = cfg.n_refiner_train
        latent_n = latent_g.unsqueeze(0).expand(n_rt, -1, -1).contiguous()
        latent_n = latent_n + torch.randn_like(latent_n) * cfg.noise_std
        fresh    = refiner.decode(latent_n)

        fresh_weights = []
        for i, (A, B) in enumerate(fresh):
            W = block_linears[i].weight.detach().unsqueeze(0)  # [1, out_i, in_i]
            fresh_weights.append(W + torch.bmm(A, B))          # [n_rt, out_i, in_i]

        r_scores   = discriminator(fresh_weights)
        r_loss     = -r_scores.mean()
        refiner_opt.zero_grad()
        r_loss.backward()
        refiner_opt.step()
        r_loss_val = r_loss.item()

        for p in discriminator.parameters():
            p.requires_grad_(True)

    # ── Commit best winner ─────────────────────────────────────────────────────
    with torch.no_grad():
        for lin, delta in zip(block_linears, deltas):
            lin.weight.data.add_(delta[best_idx])

    model.train()
    return losses[best_idx].item(), d_loss.item(), r_loss_val


# ──────────────────────────────────────────────────────────────────── training

def train():
    cfg    = Config()
    device = torch.device(cfg.device)
    print(f"Device: {device}")

    train_dataset, val_data, tokenizer = build_dataset(cfg)
    print(f"Vocab: {cfg.vocab_size} | context_length: {cfg.context_length}")

    loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=False)
    print(f"batch_size={cfg.batch_size}")

    model         = Generator(cfg).to(device)
    refiner       = Refiner(cfg).to(device)
    discriminator = Discriminator(cfg).to(device)

    n_gen   = sum(p.numel() for p in model.parameters())
    n_ref   = sum(p.numel() for p in refiner.parameters())
    n_disc  = sum(p.numel() for p in discriminator.parameters())
    print(f"Generator: {n_gen:,} params | Refiner: {n_ref:,} | Discriminator: {n_disc:,}")

    optimizer = Prodigy(model.parameters(), lr=cfg.lr, weight_decay=0.1,
                        safeguard_warmup=True, use_bias_correction=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_iters)

    refiner_opt = Prodigy(refiner.parameters(), lr=cfg.lr, weight_decay=0.01,
                          safeguard_warmup=True, use_bias_correction=True)
    disc_opt    = Prodigy(discriminator.parameters(), lr=cfg.lr, weight_decay=0.01,
                          safeguard_warmup=True, use_bias_correction=True)

    # ── Resume from checkpoint ────────────────────────────────────────────────
    grad_updates       = 0
    intervention_count = 0

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
        if "refiner_state" in ckpt:
            _load(refiner, "refiner_state")
            refiner_opt.load_state_dict(ckpt["refiner_opt"])
        if "disc_state" in ckpt:
            _load(discriminator, "disc_state")
            disc_opt.load_state_dict(ckpt["disc_opt"])
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
            "interventions", "best_inter_loss", "disc_loss", "refiner_loss",
        ])

    last_checkpoint_saved = grad_updates // cfg.checkpoint_interval
    t0               = time.time()
    t_last_log       = t0
    train_loss_sum   = 0.0
    train_loss_count = 0
    tokens_since_log = 0
    # Intervention stats (averaged over the eval interval)
    inter_best_sum   = 0.0
    inter_disc_sum   = 0.0
    inter_ref_sum    = 0.0
    inter_count_log  = 0

    layer_cycle = 0   # which layer the next intervention targets

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

            # ── Intervention ──────────────────────────────────────────────────
            if (grad_updates >= cfg.refiner_start_iter
                    and grad_updates % cfg.intervention_interval == 0):
                layer_idx = layer_cycle % cfg.n_layers
                layer_cycle += 1
                best_loss, d_loss, r_loss = run_intervention(
                    model, refiner, discriminator, refiner_opt, disc_opt,
                    val_data, layer_idx, intervention_count, cfg,
                )
                intervention_count += 1
                inter_best_sum  += best_loss
                inter_disc_sum  += d_loss
                inter_ref_sum   += r_loss
                inter_count_log += 1
                print(
                    f"  [intervention {intervention_count}] "
                    f"layer={layer_idx} | best_loss={best_loss:.4f} | "
                    f"d_loss={d_loss:.4f} | r_loss={r_loss:.4f}"
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

                avg_best = inter_best_sum / inter_count_log if inter_count_log else 0.0
                avg_disc = inter_disc_sum / inter_count_log if inter_count_log else 0.0
                avg_ref  = inter_ref_sum  / inter_count_log if inter_count_log else 0.0
                inter_best_sum = inter_disc_sum = inter_ref_sum = 0.0
                inter_count_log = 0

                print(
                    f"step {grad_updates:7d}/{cfg.max_iters} | "
                    f"t_loss {avg_train:.4f} | v_loss {val_loss:.4f} | "
                    f"lr {lr:.2e} | {tok_per_s:,.0f} tok/s | "
                    f"time: {elapsed:.0f}s | "
                    f"samp: {' '.join(sample.split())}"
                )
                log_writer.writerow([
                    grad_updates,
                    f"{avg_train:.6f}",
                    f"{val_loss:.6f}",
                    f"{lr:.6e}",
                    f"{elapsed:.1f}",
                    f"{tok_per_s:.0f}",
                    intervention_count,
                    f"{avg_best:.6f}" if avg_best else "",
                    f"{avg_disc:.6f}"  if avg_disc else "",
                    f"{avg_ref:.6f}"   if avg_ref  else "",
                ])
                log_file.flush()

            current_interval = grad_updates // cfg.checkpoint_interval
            if current_interval > last_checkpoint_saved:
                save_checkpoint(model, optimizer, scheduler,
                                refiner, refiner_opt,
                                discriminator, disc_opt,
                                grad_updates, intervention_count, cfg)
                last_checkpoint_saved = current_interval

        if grad_updates >= cfg.max_iters:
            break

    save_checkpoint(model, optimizer, scheduler,
                    refiner, refiner_opt,
                    discriminator, disc_opt,
                    grad_updates, intervention_count, cfg)
    log_file.close()
    print(f"\nTraining done in {time.time() - t0:.0f}s")
    return model, tokenizer, cfg


if __name__ == "__main__":
    train()
