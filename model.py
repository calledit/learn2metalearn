import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as SN

from config import Config


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # Uses Flash Attention when available via PyTorch's SDPA kernel
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class Generator(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.Sequential(*[
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        self.apply(self._init_weights)
        # Scale residual projections by 1/sqrt(2 * n_layers) — GPT-2 recipe
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("ff.net.2.weight"):
                nn.init.normal_(p, std=0.02 / (2 * cfg.n_layers) ** 0.5)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T] token ids → logits: [B, T, vocab_size]"""
        B, T = x.shape
        assert T <= self.cfg.context_length, f"Input length {T} exceeds context_length {self.cfg.context_length}"
        pos = torch.arange(T, device=x.device)
        h = self.drop(self.tok_emb(x) + self.pos_emb(pos))
        h = self.blocks(h)
        h = self.norm(h)
        return self.lm_head(h)


# ── Weight utilities ───────────────────────────────────────────────────────────

def get_block_linears(block: TransformerBlock) -> list:
    """The 4 linear modules in canonical order: qkv, out_proj, ff_up, ff_down."""
    return [block.attn.qkv, block.attn.out_proj, block.ff.net[0], block.ff.net[2]]


# ── Shared small transformer builder ──────────────────────────────────────────

def _make_mini_transformer(d: int, n_heads: int, n_layers: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=d, nhead=n_heads, dim_feedforward=d * 4,
        dropout=0.0, batch_first=True, norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=n_layers,
                                 norm=nn.LayerNorm(d), enable_nested_tensor=False)


# ── Layer encoder (shared by Refiner and Discriminator) ───────────────────────

class LayerEncoder(nn.Module):
    """
    Encodes a TransformerBlock's 4 linear weight matrices into 4 latent tokens.
    Input weights are batched: list of 4 tensors each [B, out_i, in_i].
    Returns [B, 4, refiner_d].
    """
    def __init__(self, cfg: Config):
        super().__init__()
        d, dr = cfg.d_model, cfg.refiner_d
        # in_dim per matrix: qkv=d, out_proj=d, ff_up=d, ff_down=4d
        self.row_projs = nn.ModuleList([
            nn.Linear(d,   dr, bias=False),
            nn.Linear(d,   dr, bias=False),
            nn.Linear(d,   dr, bias=False),
            nn.Linear(4*d, dr, bias=False),
        ])
        self.type_emb = nn.Embedding(4, dr)

    def forward(self, weights: list) -> torch.Tensor:
        """weights: list of 4 [B, out_i, in_i] → [B, 4, refiner_d]"""
        tokens = []
        for i, (w, proj) in enumerate(zip(weights, self.row_projs)):
            tok = proj(w).mean(dim=1) + self.type_emb.weight[i]  # [B, refiner_d]
            tokens.append(tok)
        return torch.stack(tokens, dim=1)  # [B, 4, refiner_d]


# ── Refiner ───────────────────────────────────────────────────────────────────

class Refiner(nn.Module):
    """
    Input funnel → bottleneck transformer → noise injection → output funnel.
    Produces LoRA pairs (A, B) where ΔW = A @ B.

    Encode runs once per intervention; sample/decode runs n_samples times cheaply
    since it is just linear layers after the shared bottleneck.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        d, dr, r = cfg.d_model, cfg.refiner_d, cfg.lora_rank

        self.encoder     = LayerEncoder(cfg)
        self.transformer = _make_mini_transformer(dr, cfg.refiner_heads, cfg.refiner_layers)

        # out/in dims for each of the 4 matrix types
        self.out_dims = [3*d, d, 4*d, d]
        self.in_dims  = [d,   d, d,   4*d]
        self.dec_A = nn.ModuleList([nn.Linear(dr, od * r, bias=False) for od in self.out_dims])
        self.dec_B = nn.ModuleList([nn.Linear(dr, r * id, bias=False) for id in self.in_dims])

        for m in [*self.dec_A, *self.dec_B]:
            nn.init.normal_(m.weight, std=0.01)

    def encode(self, weights: list) -> torch.Tensor:
        """weights: list of 4 [B, out_i, in_i]. Returns [B, 4, refiner_d]. Run once."""
        return self.transformer(self.encoder(weights))

    def decode(self, latent: torch.Tensor) -> list:
        """
        latent: [B, 4, refiner_d] with noise already added.
        Returns list of 4 (A, B): A[B, out_i, r], B[B, r, in_i].
        """
        r, pairs = self.cfg.lora_rank, []
        for i in range(4):
            A = self.dec_A[i](latent[:, i]).reshape(-1, self.out_dims[i], r)
            B = self.dec_B[i](latent[:, i]).reshape(-1, r, self.in_dims[i])
            pairs.append((A, B))
        return pairs

    def sample(self, latent_1: torch.Tensor) -> list:
        """
        latent_1: [4, refiner_d] — single squeezed encode output.
        Injects noise and decodes cfg.n_samples times in one batched pass.
        Returns list of 4 (A, B): A[n_samples, out_i, r], B[n_samples, r, in_i].
        """
        n = self.cfg.n_samples
        latent_n = latent_1.unsqueeze(0).expand(n, -1, -1).contiguous()
        latent_n = latent_n + torch.randn_like(latent_n) * self.cfg.noise_std
        return self.decode(latent_n)


# ── Discriminator ──────────────────────────────────────────────────────────────

class ScoreFunnel(nn.Module):
    """Per-token MLP → learned aggregator → scalar. Mirrors Stateweaver design."""
    def __init__(self, d_model: int, n_tokens: int):
        super().__init__()
        self.token_mlp = nn.Sequential(
            SN(nn.Linear(d_model, d_model // 2)),
            nn.GELU(),
            SN(nn.Linear(d_model // 2, 1, bias=False)),
        )
        self.aggregator = SN(nn.Linear(n_tokens, 1, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        per_token = self.token_mlp(x).squeeze(-1)     # [B, n_tokens]
        return self.aggregator(per_token).squeeze(-1)  # [B]


class Discriminator(nn.Module):
    """
    Reads a layer's full weight configuration and outputs a real/fake score.
    Input: list of 4 tensors [B, out_i, in_i]. Output: scores [B].
    Spectral norm applied throughout for training stability.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        dr = cfg.refiner_d
        self.encoder     = LayerEncoder(cfg)
        self.transformer = _make_mini_transformer(dr, cfg.refiner_heads, cfg.refiner_layers)
        self.funnel      = ScoreFunnel(dr, 4)
        self._apply_spectral_norm()

    def _apply_spectral_norm(self):
        mha_ids = {
            id(sub)
            for m in self.transformer.modules()
            if isinstance(m, nn.MultiheadAttention)
            for sub in m.modules()
        }
        for mod in self.transformer.modules():
            if id(mod) not in mha_ids and isinstance(mod, nn.Linear):
                SN(mod)

    def forward(self, weights: list) -> torch.Tensor:
        """weights: list of 4 [B, out_i, in_i] → scores [B]"""
        x = self.transformer(self.encoder(weights))
        return self.funnel(x)


def disc_hinge_loss(real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
    """Push real > +1, fake < −1."""
    return F.relu(1.0 - real_scores).mean() + F.relu(1.0 + fake_scores).mean()
