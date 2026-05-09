import argparse
import glob
import os
import re
import torch
import torch.nn.functional as F

from config import Config
from model import Generator
from data import ByteTokenizer


def find_latest_checkpoint(checkpoint_dir: str = "checkpoints") -> str | None:
    files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pt"))
    if not files:
        return None
    def _step(f):
        m = re.search(r"checkpoint_(\d+)\.pt", os.path.basename(f))
        return int(m.group(1)) if m else -1
    return max(files, key=_step)


def load_checkpoint(path: str = None, device: str = None):
    if path is None:
        path = find_latest_checkpoint()
    if path is None:
        raise FileNotFoundError("No checkpoint found. Train the model first.")

    print(f"Loading {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg  = ckpt["cfg"]

    if device:
        cfg.device = device

    tokenizer = ByteTokenizer()
    model = Generator(cfg)
    state_dict = ckpt["model_state"]
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(cfg.device)
    model.eval()

    step = ckpt.get("grad_updates", "?")
    print(f"Loaded checkpoint from step {step}")
    return model, tokenizer, cfg


@torch.no_grad()
def generate(
    model: Generator,
    tokenizer: ByteTokenizer,
    cfg: Config,
    prompt: str = "\n",
    max_new_tokens: int = None,
    temperature: float = None,
    deterministic: bool = False,
) -> str:
    device         = next(model.parameters()).device
    max_new_tokens = max_new_tokens or cfg.max_new_tokens
    temperature    = temperature    or cfg.temperature

    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        raise ValueError("Prompt must contain at least one character.")

    # Keep a rolling buffer of the last context_length tokens
    context = prompt_ids[:]

    for _ in range(max_new_tokens):
        if deterministic:
            torch.manual_seed(len(context))

        # Trim to context window
        ctx = context[-cfg.context_length:]
        x   = torch.tensor(ctx, dtype=torch.long, device=device).unsqueeze(0)

        logits   = model(x)             # [1, T, vocab_size]
        logits   = logits[0, -1, :]    # last position
        logits   = logits / temperature
        probs    = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1).item()

        context.append(next_tok)

    return tokenizer.decode(context)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a Generator checkpoint.")
    parser.add_argument("--checkpoint",   "-c", default=None,          help="Path to checkpoint (default: latest)")
    parser.add_argument("--prompt",       "-p", default="\n",          help="Prompt string")
    parser.add_argument("--tokens",       "-n", type=int, default=500, help="Number of tokens to generate")
    parser.add_argument("--temperature",  "-t", type=float, default=None)
    parser.add_argument("--deterministic","-d", action="store_true",   help="Fully deterministic sampling")
    parser.add_argument("--device",       "-D", default="cpu")
    args = parser.parse_args()

    if args.deterministic:
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model, tokenizer, cfg = load_checkpoint(args.checkpoint, device=args.device)

    print(f"\nPrompt: {repr(args.prompt)}  deterministic={args.deterministic}")
    print("-" * 60)
    text = generate(
        model, tokenizer, cfg,
        prompt=args.prompt, max_new_tokens=args.tokens,
        temperature=args.temperature, deterministic=args.deterministic,
    )
    print(text)
    print("-" * 60)
