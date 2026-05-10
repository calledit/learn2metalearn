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

class MatrixEncoder(nn.Module):
    """
    Compresses one weight matrix [B, n_rows, in_dim] → [B, n_out, d] using
    learned cross-attention queries over the rows. Each query can specialise
    on different subspaces of the weight matrix rather than averaging them away.
    """
    def __init__(self, in_dim: int, d: int, n_out: int, n_heads: int):
        super().__init__()
        self.row_proj   = nn.Linear(in_dim, d, bias=False)
        self.queries    = nn.Parameter(torch.randn(n_out, d) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d, n_heads, batch_first=True, bias=False)
        self.norm       = nn.LayerNorm(d)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """w: [B, n_rows, in_dim] → [B, n_out, d]"""
        x = self.row_proj(w)                                      # [B, n_rows, d]
        q = self.queries.unsqueeze(0).expand(w.size(0), -1, -1)  # [B, n_out, d]
        out, _ = self.cross_attn(q, x, x)                        # [B, n_out, d]
        return self.norm(out)


class LayerEncoder(nn.Module):
    """
    Encodes a TransformerBlock's 4 weight matrices into 4*n_out tokens.
    Each matrix gets its own MatrixEncoder that reads rows via cross-attention.
    Input: list of 4 [B, out_i, in_i].  Returns [B, 4*n_out, refiner_d].
    """
    def __init__(self, cfg: Config):
        super().__init__()
        d, n_out, heads = cfg.refiner_d, cfg.encoder_n_out, cfg.refiner_heads
        in_dims = [cfg.d_model, cfg.d_model, cfg.d_model, 4 * cfg.d_model]
        self.encoders = nn.ModuleList([
            MatrixEncoder(in_dim, d, n_out, heads) for in_dim in in_dims
        ])
        self.type_emb = nn.Embedding(4, d)

    def forward(self, weights: list) -> torch.Tensor:
        """weights: list of 4 [B, out_i, in_i] → [B, 4*n_out, refiner_d]"""
        tokens = []
        for i, (w, enc) in enumerate(zip(weights, self.encoders)):
            t = enc(w) + self.type_emb.weight[i]  # [B, n_out, d]
            tokens.append(t)
        return torch.cat(tokens, dim=1)            # [B, 4*n_out, d]


# ── Refiner ───────────────────────────────────────────────────────────────────

class MatrixDecoder(nn.Module):
    """
    Reverse of MatrixEncoder: [B, n_out, d] → [B, out_dim, in_dim].
    Learned queries (one per output row) attend to the n_out encoded tokens,
    then project to in_dim. Mirrors the encoder cross-attention symmetrically.
    """
    def __init__(self, out_dim: int, in_dim: int, d: int, n_heads: int):
        super().__init__()
        self.queries    = nn.Parameter(torch.randn(out_dim, d) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d, n_heads, batch_first=True, bias=False)
        self.out_proj   = nn.Linear(d, in_dim, bias=False)
        self.norm       = nn.LayerNorm(d)
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, n_out, d] → [B, out_dim, in_dim]"""
        B = z.size(0)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        out, _ = self.cross_attn(q, z, z)
        return self.out_proj(self.norm(out))


class LayerDecoder(nn.Module):
    """
    Reverse of LayerEncoder: [B, 4*n_out, d] → list of 4 [B, out_i, in_i] residuals.
    One MatrixDecoder per weight matrix type.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        d, n_out, heads = cfg.refiner_d, cfg.encoder_n_out, cfg.refiner_heads
        shapes = [
            (3*cfg.d_model, cfg.d_model),  # qkv
            (cfg.d_model,   cfg.d_model),  # out_proj
            (4*cfg.d_model, cfg.d_model),  # ff_up
            (cfg.d_model, 4*cfg.d_model),  # ff_down
        ]
        self.decoders = nn.ModuleList([
            MatrixDecoder(od, id_, d, heads) for od, id_ in shapes
        ])
        self.n_out = n_out

    def forward(self, latent: torch.Tensor) -> list:
        """latent: [B, 4*n_out, d] → list of 4 [B, out_i, in_i]"""
        return [dec(latent[:, i*self.n_out:(i+1)*self.n_out])
                for i, dec in enumerate(self.decoders)]


class Refiner(nn.Module):
    """
    Symmetric encoder-decoder over weight matrices.
    Encode: weights → transformer bottleneck (with layer identity).
    Decode: latent + noise → raw ΔW residuals (same shape as inputs).
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        dr = cfg.refiner_d

        self.encoder     = LayerEncoder(cfg)
        self.decoder     = LayerDecoder(cfg)
        self.transformer = _make_mini_transformer(dr, cfg.refiner_heads, cfg.refiner_layers)
        self.layer_emb   = nn.Embedding(cfg.n_layers, dr)
        nn.init.normal_(self.layer_emb.weight, std=0.02)

    def encode(self, weights: list, layer_idx: int) -> torch.Tensor:
        """weights: list of 4 [B, out_i, in_i]. Returns [B, 4*n_out, refiner_d]."""
        tokens = self.encoder(weights)
        layer_token = self.layer_emb(torch.tensor(layer_idx, dtype=torch.long, device=tokens.device))
        return self.transformer(tokens + layer_token)

    def decode(self, latent: torch.Tensor) -> list:
        """latent: [B, 4*n_out, refiner_d] → list of 4 [B, out_i, in_i] deltas."""
        return self.decoder(latent)

    def sample(self, latent_1: torch.Tensor) -> list:
        """
        latent_1: [4*n_out, refiner_d] — single squeezed encode output.
        Injects noise and decodes cfg.n_samples times in one batched pass.
        Returns list of 4 [n_samples, out_i, in_i] deltas.
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
    Transition detector: takes two weight configurations (before, after) and
    scores whether the transition represents genuine improvement.

    Real:  [worse, better] — the ordering the refiner should produce.
    Fake:  [better, worse] — degradation, which the refiner should avoid.

    Encodes each set of 4 matrices to 4*n_out tokens, concatenates to 8*n_out,
    adds a layer embedding so the discriminator knows which layer it's judging,
    then scores the full transition with a learned funnel.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        dr = cfg.refiner_d
        self.encoder     = LayerEncoder(cfg)
        self.transformer = _make_mini_transformer(dr, cfg.refiner_heads, cfg.refiner_layers)
        self.funnel      = ScoreFunnel(dr, 8 * cfg.encoder_n_out)
        self.layer_emb   = nn.Embedding(cfg.n_layers, dr)
        nn.init.normal_(self.layer_emb.weight, std=0.02)
        self._apply_spectral_norm()

    def _apply_spectral_norm(self):
        mha_ids = set()
        for m in self.modules():
            if isinstance(m, nn.MultiheadAttention):
                mha_ids.update(id(sub) for sub in m.modules())
        for part in [self.transformer, self.encoder]:
            for mod in part.modules():
                if id(mod) not in mha_ids and isinstance(mod, nn.Linear):
                    SN(mod)

    def forward(self, weights_before: list, weights_after: list,
                layer_idx: "int | torch.Tensor") -> torch.Tensor:
        """
        weights_before, weights_after: each a list of 4 [B, out_i, in_i].
        layer_idx: int (broadcast) or [B] LongTensor (per-sample).
        Returns scores [B].
        """
        tokens = torch.cat([
            self.encoder(weights_before),   # [B, 4*n_out, dr]
            self.encoder(weights_after),    # [B, 4*n_out, dr]
        ], dim=1)                           # [B, 8*n_out, dr]
        B = tokens.size(0)
        if isinstance(layer_idx, int):
            idx = torch.full((B,), layer_idx, dtype=torch.long, device=tokens.device)
        else:
            idx = layer_idx.to(tokens.device)
        tokens = tokens + self.layer_emb(idx).unsqueeze(1)  # [B, 8*n_out, dr]
        return self.funnel(self.transformer(tokens))


def disc_hinge_loss(real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
    """Push real > +1, fake < −1."""
    return F.relu(1.0 - real_scores).mean() + F.relu(1.0 + fake_scores).mean()
