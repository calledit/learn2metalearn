import os
import urllib.request
import zipfile
import torch
from torch.utils.data import Dataset, IterableDataset

from config import Config


_URL = "https://huggingface.co/datasets/mattdangerw/wikitext-103-raw/resolve/main/wikitext-103-raw-v1.zip"
_DIR = "data/wikitext-103-raw"


class ByteTokenizer:
    """Encodes text as raw UTF-8 bytes. vocab_size is always 256."""
    vocab_size = 256

    def encode(self, text: str) -> list:
        return list(text.encode("utf-8"))

    def decode(self, ids) -> str:
        return bytes(int(i) for i in ids).decode("utf-8", errors="replace")


class SequenceDataset(IterableDataset):
    """
    Wraps a HuggingFace streaming dataset and yields fixed-size chunks on the fly.
    Documents are concatenated with a newline separator until the buffer reaches
    context_length + 1 tokens, then a chunk is emitted and the remainder is kept.
    """

    def __init__(self, hf_dataset, context_length: int):
        self.hf_dataset = hf_dataset
        self.context_length = context_length

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        dataset = self.hf_dataset
        if worker_info is not None:
            dataset = dataset.shard(
                num_shards=worker_info.num_workers,
                index=worker_info.id,
            )

        buf = bytearray()
        for doc in dataset:
            buf.extend(doc["text"].encode("utf-8"))
            buf.append(10)  # \n separator between documents
            while len(buf) >= self.context_length + 1:
                chunk = torch.tensor(buf[: self.context_length + 1], dtype=torch.long)
                yield chunk
                del buf[: self.context_length]


class FlatSequenceDataset(Dataset):
    """Sequential non-overlapping chunks from a flat token tensor."""

    def __init__(self, data: torch.Tensor, context_length: int):
        self.data = data
        self.context_length = context_length
        self.n_chunks = (len(data) - 1) // context_length

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int) -> torch.Tensor:
        start = idx * self.context_length
        return self.data[start : start + self.context_length + 1].long()


_FINEWEB_VAL_DOCS = 500
_FINEWEB_SHUFFLE_BUFFER = 10_000


def _build_fineweb_dataset(cfg: Config):
    from datasets import load_dataset

    print("Loading FineWeb-Edu (streaming)...")

    def _stream():
        return load_dataset(
            "HuggingFaceFW/fineweb-edu", name="sample-10BT",
            split="train", streaming=True,
        )

    val_bytes = b"".join(
        doc["text"].encode("utf-8")
        for doc in _stream().take(_FINEWEB_VAL_DOCS)
    )
    val_data = torch.tensor(bytearray(val_bytes), dtype=torch.long)

    train_stream = _stream().skip(_FINEWEB_VAL_DOCS).shuffle(buffer_size=_FINEWEB_SHUFFLE_BUFFER)

    print(f"  Val: {len(val_bytes):,} bytes | Train: streaming with buffer-shuffle {_FINEWEB_SHUFFLE_BUFFER:,}")
    return SequenceDataset(train_stream, cfg.context_length), val_data


def _load_wikitext103() -> tuple[bytes, bytes]:
    train_path = os.path.join(_DIR, "wiki.train.raw")
    val_path   = os.path.join(_DIR, "wiki.valid.raw")

    if not os.path.exists(train_path):
        zip_path = "data/wikitext-103-raw-v1.zip"
        os.makedirs("data", exist_ok=True)

        if not os.path.exists(zip_path):
            print("Downloading WikiText-103 (~183 MB)...")
            def _progress(count, block, total):
                print(f"\r  {min(count * block / total * 100, 100):.1f}%", end="", flush=True)
            urllib.request.urlretrieve(_URL, zip_path, reporthook=_progress)
            print()

        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall("data")

    with open(train_path, "rb") as f:
        train_bytes = f.read()
    with open(val_path, "rb") as f:
        val_bytes = f.read()

    return train_bytes, val_bytes


def _load_oasst2() -> tuple[bytes, bytes]:
    from datasets import load_dataset

    print("Loading OASST2...")
    ds = load_dataset("OpenAssistant/oasst2", num_proc=1)

    def extract(split: str) -> str:
        rows = list(ds[split])

        by_id    = {r["message_id"]: r for r in rows}
        children = {}
        for r in rows:
            if r["parent_id"]:
                children.setdefault(r["parent_id"], []).append(r)

        for kids in children.values():
            kids.sort(key=lambda m: m["rank"] if m["rank"] is not None else 999)

        conversations = []
        for r in rows:
            if r["parent_id"] is not None:
                continue
            if not (r.get("lang") or "").startswith("en"):
                continue
            if r.get("deleted"):
                continue

            turns = []
            node  = r
            while node:
                if node.get("deleted"):
                    break
                role = "User" if node["role"] == "prompter" else "Assistant"
                turns.append(f"{role}: {node['text'].strip()}")
                kids = [
                    c for c in children.get(node["message_id"], [])
                    if not c.get("deleted") and (c.get("lang") or "").startswith("en")
                ]
                node = kids[0] if kids else None

            if len(turns) >= 2:
                conversations.append("\n".join(turns))

        return "\n\n".join(conversations)

    train_text = extract("train")
    val_text   = extract("validation")
    print(f"OASST2: {len(train_text):,} train chars | {len(val_text):,} val chars")
    return train_text.encode("utf-8"), val_text.encode("utf-8")


def build_dataset(cfg: Config):
    tokenizer = ByteTokenizer()

    if cfg.dataset == "fineweb_edu":
        train_dataset, val_data = _build_fineweb_dataset(cfg)
        return train_dataset, val_data, tokenizer

    if cfg.dataset == "oasst2":
        train_raw, val_raw = _load_oasst2()
    else:
        print("Loading WikiText-103...")
        train_raw, val_raw = _load_wikitext103()

    print(f"Train: {len(train_raw):,} bytes | Val: {len(val_raw):,} bytes")
    train_data    = torch.tensor(bytearray(train_raw), dtype=torch.uint8)
    val_data      = torch.tensor(bytearray(val_raw),   dtype=torch.long)
    train_dataset = FlatSequenceDataset(train_data, cfg.context_length)
    return train_dataset, val_data, tokenizer
