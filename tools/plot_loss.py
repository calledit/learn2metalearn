"""
plot_loss.py — plot training curves from checkpoints/training_log.csv

Usage:
    python tools/plot_loss.py [--log PATH] [--smooth N] [--start STEP]
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def smooth(values, window):
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log",    default="checkpoints/training_log.csv")
    parser.add_argument("--smooth", type=int, default=20, help="Smoothing window (steps)")
    parser.add_argument("--start",  type=int, default=0,  help="Ignore steps before this value")
    args = parser.parse_args()

    # Read robustly — handles resumed runs where the header may repeat
    chunks = []
    header = None
    with open(args.log) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(",")
            if header is None:
                header = parts
                continue
            if parts[0] == "step":
                continue
            if len(parts) < len(header):
                parts += [""] * (len(header) - len(parts))
            chunks.append(parts)

    df = pd.DataFrame(chunks, columns=header)
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    for col in ["train_loss", "val_loss", "lr", "elapsed_s", "tok_per_s"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop_duplicates(subset="step", keep="last").sort_values("step").reset_index(drop=True)

    if args.start > 0:
        df = df[df["step"] >= args.start].reset_index(drop=True)

    print(f"Loaded {len(df)} rows, steps {df['step'].min():.0f} – {df['step'].max():.0f}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Generator Training", fontsize=14)

    def plot(ax, col, label, color, set_ylabel=True):
        data = df[col].dropna()
        if data.empty:
            return
        steps = df.loc[data.index, "step"]
        ax.plot(steps, data, alpha=0.25, color=color, linewidth=0.8)
        if len(data) >= args.smooth:
            s       = smooth(data.values, args.smooth)
            s_steps = steps.values[args.smooth - 1:]
            ax.plot(s_steps, s, color=color, linewidth=1.8, label=label)
        else:
            ax.plot(steps, data, color=color, linewidth=1.8, label=label)
        if set_ylabel:
            ax.set_ylabel(label)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    plot(axes[0], "train_loss", "train loss", "steelblue")
    plot(axes[0], "val_loss",   "val loss",   "darkorange")
    axes[0].set_title("Cross-Entropy Loss")

    plot(axes[1], "lr", "learning rate (Prodigy)", "seagreen")
    axes[1].set_title("Learning Rate")
    axes[1].set_yscale("log")

    axes[-1].set_xlabel("Step")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
