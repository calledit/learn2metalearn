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


# ─────────────────────────────────────── discriminator warmup (Phase 2)

def warmup_discriminator(discriminator, disc_opt, before_weights, after_weights, layer_idx):
    """
    before_weights / after_weights: list of 4 tensors each [B, out_i, in_i].
    before has higher loss, after has lower loss within the same layer.
    Real: [before → after].  Fake: [after → before].
    """
    real_scores = discriminator(before_weights, after_weights, layer_idx)
    fake_scores = discriminator(after_weights, before_weights, layer_idx)
    d_loss = disc_hinge_loss(real_scores, fake_scores)
    disc_opt.zero_grad()
    d_loss.backward()
    disc_opt.step()
    return d_loss.item()


# ──────────────────────────────────────── disc buffer training helper

def _train_disc_from_buffer(buffer, layer_idx, discriminator, disc_opt, device, cfg):
    """Sample random pairs from the buffer. Lower-loss entry is 'after', higher is 'before'."""
    if len(buffer) < 2:
        return None
    buf = list(buffer)
    max_unique_pairs = len(buf) * (len(buf) - 1) // 2
    n = min(cfg.disc_n_pairs, max_unique_pairs // 4)
    # Sample with replacement so small buffers still produce full batches
    idx_a = torch.randint(len(buf), (n,)).tolist()
    idx_b = torch.randint(len(buf), (n,)).tolist()
    before_list, after_list = [], []
    for a, b in zip(idx_a, idx_b):
        lo, hi = (a, b) if buf[a][1] <= buf[b][1] else (b, a)
        after_list.append(buf[lo][0])   # lower loss → after
        before_list.append(buf[hi][0])  # higher loss → before
    before_batch = [torch.stack([e[j] for e in before_list]).to(device) for j in range(4)]
    after_batch  = [torch.stack([e[j] for e in after_list ]).to(device) for j in range(4)]
    return warmup_discriminator(discriminator, disc_opt, before_batch, after_batch, layer_idx)


# ──────────────────────────────────────── refiner generator-loss training

def train_refiner_on_gen_loss(refiner, model, refiner_opt, x, y, layer_idx, cfg, model_snap_buf):
    """
    Train refiner so its proposed deltas reduce generator cross-entropy.
    Samples a full model snapshot for a coherent forward pass — the 4 target-layer
    weights become snapshot_weights + refiner_delta, everything else comes from the
    snapshot unchanged. Gradients flow: CE loss → deltas → refiner decoder/encoder.
    """
    if not model_snap_buf:
        return None

    device = next(model.parameters()).device
    buf  = list(model_snap_buf)
    snap = buf[torch.randint(len(buf), (1,)).item()]

    param_names = [
        f"blocks.{layer_idx}.attn.qkv.weight",
        f"blocks.{layer_idx}.attn.out_proj.weight",
        f"blocks.{layer_idx}.ff.net.0.weight",
        f"blocks.{layer_idx}.ff.net.2.weight",
    ]
    bw = [snap[name].unsqueeze(0).to(device) for name in param_names]

    latent = refiner.encode(bw, layer_idx)
    deltas = refiner.decode(latent)  # list of 4 [1, out_i, in_i]

    param_name_set = set(param_names)
    modified = {name: p.to(device) for name, p in snap.items() if name not in param_name_set}
    for name, delta in zip(param_names, deltas):
        modified[name] = snap[name].to(device) + delta.squeeze(0)

    model.eval()
    logits = torch.func.functional_call(model, modified, x, strict=False)
    model.train()

    gen_loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
    refiner_opt.zero_grad()
    (gen_loss * cfg.ref_gen_loss_scale).backward()
    refiner_opt.step()

    return gen_loss.item()


# ─────────────────────────────────────── refiner warmup (Phase 2.5)

def warmup_refiner(refiner, discriminator, refiner_opt, refiner_buf, layer_idx, cfg, device):
    """
    Train refiner on real weight snapshots from refiner_buf.
    Encodes each snapshot, decodes to get deltas, scores the transition with
    the discriminator. No noise — diversity comes from the real samples.
    """
    if len(refiner_buf) < cfg.refiner_buffer_start:
        return None

    buf = list(refiner_buf)
    n   = min(cfg.refiner_warmup_batch, len(buf))
    idx = torch.randint(len(buf), (n,)).tolist()
    sampled = [buf[i][0] for i in idx]   # each: list of 4 cpu tensors

    base_batch = [torch.stack([s[j] for s in sampled]).to(device) for j in range(4)]

    latents      = refiner.encode(base_batch, layer_idx)   # [n, 4*n_out, d]
    fresh        = refiner.decode(latents)                  # list of 4 [n, out_i, in_i]
    fresh_weights = [base + delta for base, delta in zip(base_batch, fresh)]

    for p in discriminator.parameters():
        p.requires_grad_(False)

    r_scores = discriminator(base_batch, fresh_weights, layer_idx)
    r_loss   = -r_scores.mean()
    refiner_opt.zero_grad()
    r_loss.backward()
    refiner_opt.step()

    for p in discriminator.parameters():
        p.requires_grad_(True)

    return r_loss.item()


# ────────────────────────────────────────────────────────────────── intervention

def run_intervention(model, refiner, discriminator, val_data, layer_idx, cfg):
    device = next(model.parameters()).device
    model.eval()

    start  = torch.randint(0, len(val_data) - cfg.context_length - 1, (1,)).item()
    rank_x = val_data[start     : start + cfg.context_length    ].unsqueeze(0)
    rank_y = val_data[start + 1 : start + cfg.context_length + 1].unsqueeze(0)

    block_linears = get_block_linears(model.blocks[layer_idx])
    layer_weights = [lin.weight for lin in block_linears]

    # ── Single deterministic refinement ───────────────────────────────────────
    with torch.no_grad():
        bw     = [w.detach().unsqueeze(0) for w in layer_weights]
        latent = refiner.encode(bw, layer_idx)   # [1, 4*n_out, d]
        deltas = refiner.decode(latent)           # list of 4 [1, out_i, in_i]

    # ── Baseline loss ─────────────────────────────────────────────────────────
    with torch.inference_mode():
        baseline_loss = F.cross_entropy(
            model(rank_x).reshape(-1, cfg.vocab_size), rank_y.reshape(-1)
        ).item()

    # ── Evaluate candidate ────────────────────────────────────────────────────
    with torch.inference_mode():
        for lin, delta in zip(block_linears, deltas):
            lin.weight.data.add_(delta.squeeze(0))
        cand_loss = F.cross_entropy(
            model(rank_x).reshape(-1, cfg.vocab_size), rank_y.reshape(-1)
        ).item()
        for lin, delta in zip(block_linears, deltas):
            lin.weight.data.sub_(delta.squeeze(0))

    # ── Discriminator score on proposed transition (informational) ───────────
    with torch.no_grad():
        r_score = discriminator(bw, [d + base for d, base in zip(deltas, bw)], layer_idx).mean().item()

    # ── Commit if better than baseline ────────────────────────────────────────
    committed = False
    if cand_loss <= baseline_loss + cfg.intervention_commit_margin:
        with torch.no_grad():
            for lin, delta in zip(block_linears, deltas):
                lin.weight.data.add_(delta.squeeze(0))
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
    grad_updates        = 0
    intervention_count  = 0
    # Per-layer discriminator buffers — each entry: (list of 4 cpu tensors, float loss)
    disc_buf          = [deque(maxlen=cfg.disc_buffer_size) for _ in range(cfg.n_layers)]
    disc_layer_cycle  = 0
    # Per-layer refiner buffers — snapshotted at disc_snapshot_interval//2 offset
    refiner_buf       = [deque(maxlen=cfg.refiner_buffer_size) for _ in range(cfg.n_layers)]
    refiner_layer_cycle = 0
    # Full model snapshots for ref_gen training — coherent across all layers
    model_snap_buf    = deque(maxlen=cfg.model_snap_buffer_size)
    warmup_disc_sum   = 0.0
    warmup_disc_count = 0
    warmup_ref_sum    = 0.0
    warmup_ref_count  = 0
    bad_ref_sum       = 0.0
    bad_ref_count     = 0
    ref_gen_sum       = 0.0
    ref_gen_count     = 0

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
            try:
                _load(refiner, "refiner_state")
                refiner_opt.load_state_dict(ckpt["refiner_opt"])
            except RuntimeError as e:
                print(f"  [warn] refiner checkpoint incompatible, starting fresh: {e}")
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
            "disc_loss", "warmup_ref_loss",
            "interventions", "best_inter_loss", "bad_ref_loss", "ref_gen_loss",
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

            # ── Discriminator + refiner training on GD transitions ───────────
            if grad_updates >= cfg.disc_start_iter:
                def _snap(layer_idx):
                    lins = get_block_linears(model.blocks[layer_idx])
                    return [l.weight.detach().cpu().clone() for l in lins]

                if grad_updates % cfg.disc_snapshot_interval == 0:
                    _li = disc_layer_cycle % cfg.n_layers
                    disc_buf[_li].append((_snap(_li), loss.item()))
                    disc_layer_cycle += 1

                _refiner_offset = cfg.disc_snapshot_interval // 2
                if grad_updates % cfg.disc_snapshot_interval == _refiner_offset:
                    _li = refiner_layer_cycle % cfg.n_layers
                    refiner_buf[_li].append((_snap(_li), loss.item()))
                    refiner_layer_cycle += 1

                _wr_fired = False
                _ref_gen_active = not cfg.ref_gen_loss_end_iter or grad_updates < cfg.ref_gen_loss_end_iter
                if _ref_gen_active and grad_updates % cfg.model_snap_interval == 0:
                    model_snap_buf.append(
                        {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
                    )
                    if (len(model_snap_buf) == cfg.model_snap_buffer_size
                            and grad_updates >= cfg.refiner_warmup_start_iter):
                        _rg_li = (grad_updates // cfg.model_snap_interval) % cfg.n_layers
                        wr = warmup_refiner(refiner, discriminator, refiner_opt,
                                            refiner_buf[_rg_li], _rg_li, cfg, device)
                        if wr is not None:
                            warmup_ref_sum   += wr
                            warmup_ref_count += 1
                        rg = train_refiner_on_gen_loss(
                            refiner, model, refiner_opt, x, y, _rg_li, cfg, model_snap_buf
                        )
                        if rg is not None:
                            ref_gen_sum   += rg
                            ref_gen_count += 1
                        model_snap_buf.clear()
                        _wr_fired = True

                if grad_updates % cfg.disc_train_interval == 0:
                    any_disc_trained = False
                    for _li in range(cfg.n_layers):
                        if len(disc_buf[_li]) >= cfg.disc_buffer_start:
                            result = _train_disc_from_buffer(disc_buf[_li], _li, discriminator, disc_opt, device, cfg)
                            if result is not None:
                                warmup_disc_sum   += result
                                warmup_disc_count += 1
                                any_disc_trained   = True
                    if any_disc_trained and grad_updates >= cfg.refiner_warmup_start_iter and not _wr_fired:
                        _li = (grad_updates // cfg.disc_train_interval) % cfg.n_layers
                        wr = warmup_refiner(refiner, discriminator, refiner_opt,
                                            refiner_buf[_li], _li, cfg, device)
                        if wr is not None:
                            warmup_ref_sum   += wr
                            warmup_ref_count += 1

                        # Add one bad refiner sample to disc_buf so the discriminator
                        # learns to detect degrading proposals. We only add it when it
                        # increases loss (vs. current batch), and we take the first sample
                        # so we only run one forward pass.
                        with torch.no_grad():
                            _block_lins = get_block_linears(model.blocks[_li])
                            _lw = [lin.weight for lin in _block_lins]
                            _bw = [w.detach().unsqueeze(0) for w in _lw]
                            _lat = refiner.encode(_bw, _li)
                            _lat = _lat + torch.randn_like(_lat) * cfg.noise_std
                            _deltas = refiner.decode(_lat)   # list of 4 [1, out_i, in_i]
                            for lin, d in zip(_block_lins, _deltas):
                                lin.weight.data.add_(d.squeeze(0))
                            model.eval()
                            _cand_loss = F.cross_entropy(
                                model(x).reshape(-1, cfg.vocab_size), y.reshape(-1)
                            ).item()
                            model.train()
                            for lin, d in zip(_block_lins, _deltas):
                                lin.weight.data.sub_(d.squeeze(0))
                            if _cand_loss > loss.item():
                                _snap_ref = [
                                    (w.detach() + d.squeeze(0).detach()).cpu().clone()
                                    for w, d in zip(_lw, _deltas)
                                ]
                                disc_buf[_li].append((_snap_ref, _cand_loss))
                                bad_ref_sum   += _cand_loss
                                bad_ref_count += 1

            # ── Phase 3: intervention ─────────────────────────────────────────
            if (grad_updates >= cfg.refiner_start_iter
                    and grad_updates % cfg.intervention_interval == 0):
                layer_idx = layer_cycle % cfg.n_layers
                layer_cycle += 1
                best_loss, baseline_loss, r_score, committed = run_intervention(
                    model, refiner, discriminator, val_data, layer_idx, cfg,
                )
                intervention_count += 1
                inter_best_sum  += best_loss
                inter_r_sum     += r_score
                inter_count_log += 1
                print(
                    f"  [intervention {intervention_count}] "
                    f"layer={layer_idx} | cand={best_loss:.4f} baseline={baseline_loss:.4f} | "
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

                avg_best  = inter_best_sum / inter_count_log if inter_count_log else 0.0
                avg_rscore = inter_r_sum   / inter_count_log if inter_count_log else 0.0
                avg_disc     = warmup_disc_sum / warmup_disc_count if warmup_disc_count else 0.0
                avg_wref     = warmup_ref_sum  / warmup_ref_count  if warmup_ref_count  else 0.0
                avg_bad_ref  = bad_ref_sum     / bad_ref_count     if bad_ref_count     else 0.0
                avg_ref_gen  = ref_gen_sum     / ref_gen_count     if ref_gen_count     else 0.0
                inter_best_sum  = 0.0
                inter_r_sum     = 0.0
                inter_count_log = 0
                warmup_disc_sum   = 0.0
                warmup_disc_count = 0
                warmup_ref_sum    = 0.0
                warmup_ref_count  = 0
                bad_ref_sum   = 0.0
                bad_ref_count = 0
                ref_gen_sum   = 0.0
                ref_gen_count = 0

                print(
                    f"step {grad_updates:7d}/{cfg.max_iters} | "
                    f"t_loss {avg_train:.4f} | v_loss {val_loss:.4f} | "
                    f"lr {lr:.2e} | {tok_per_s:,.0f} tok/s | "
                    f"time: {elapsed:.0f}s | "
                    + (f"disc_loss {avg_disc:.4f} | " if avg_disc else "")
                    + (f"wr_loss {avg_wref:.4f} | "  if avg_wref  else "")
                    + (f"bad_ref {avg_bad_ref:.4f} | " if avg_bad_ref else "")
                    + (f"ref_gen {avg_ref_gen:.4f} | " if avg_ref_gen else "")
                    + f"samp: {' '.join(sample.split())}"
                )
                log_writer.writerow([
                    grad_updates,
                    f"{avg_train:.6f}",
                    f"{val_loss:.6f}",
                    f"{lr:.6e}",
                    f"{elapsed:.1f}",
                    f"{tok_per_s:.0f}",
                    f"{avg_disc:.6f}"   if avg_disc   else "",
                    f"{avg_wref:.6f}"   if avg_wref   else "",
                    intervention_count,
                    f"{avg_best:.6f}"   if avg_best   else "",
                    f"{avg_bad_ref:.6f}" if avg_bad_ref else "",
                    f"{avg_ref_gen:.6f}" if avg_ref_gen else "",
                ])
                log_file.flush()

            current_interval = grad_updates // cfg.checkpoint_interval
            if current_interval > last_checkpoint_saved:
                save_checkpoint(model, optimizer, scheduler,
                                refiner, refiner_opt,
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
                    refiner, refiner_opt,
                    discriminator, disc_opt,
                    grad_updates, intervention_count, cfg)
    log_file.close()
    print(f"\nTraining done in {time.time() - t0:.0f}s")
    return model, tokenizer, cfg


if __name__ == "__main__":
    train()
