import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from alignment.utils import read_json, read_jsonl

TRAIN_COLOR = "#2a78d6"
VAL_COLOR = "#eb6834"
BEST_COLOR = "#52514e"
DPO_PANELS = [("loss", "DPO loss"), ("reward_acc", "reward accuracy"), ("reward_margin", "reward margin")]


def smooth(values, weight=0.9):
    smoothed, last = [], values[0]
    for value in values:
        last = weight * last + (1 - weight) * value
        smoothed.append(last)
    return smoothed


def style(ax, title, ylabel):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_series(ax, records, train_key, val_key):
    train = [record for record in records if record.get(train_key) is not None]
    val = [record for record in records if record.get(val_key) is not None]
    if train:
        steps = [record["step"] for record in train]
        values = [record[train_key] for record in train]
        ax.plot(steps, values, color=TRAIN_COLOR, alpha=0.25, linewidth=1)
        ax.plot(steps, smooth(values), color=TRAIN_COLOR, linewidth=2, label="train (EMA 0.9)")
    if val:
        steps = [record["step"] for record in val]
        values = [record[val_key] for record in val]
        ax.plot(steps, values, color=VAL_COLOR, linewidth=2, marker="o", markersize=5, label="validation")


def mark_best(ax, records):
    best = [record["step"] for record in records if record.get("saved_best")]
    if best:
        ax.axvline(best[-1], color=BEST_COLOR, linestyle="--", linewidth=1, label=f"best checkpoint (step {best[-1]})")


def save(fig, out_dir, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, name), dpi=200)
    plt.close(fig)
    print(f"saved {name}")


def plot_sft(records, out_dir):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    plot_series(ax, records, "train_loss", "val_loss")
    mark_best(ax, records)
    style(ax, "SFT: Qwen2.5-0.5B on UltraChat", "token-mean NLL")
    ax.legend(frameon=False)
    save(fig, out_dir, "sft_loss.png")


def plot_dpo(records, title, out_dir, name):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, (key, label) in zip(axes, DPO_PANELS):
        plot_series(ax, records, f"train_{key}", f"val_{key}")
        mark_best(ax, records)
        style(ax, label, label)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=11)
    save(fig, out_dir, name)


def lora_records(log_history):
    records = []
    for entry in log_history:
        record = {"step": entry["step"]}
        if "loss" in entry:
            record["train_loss"] = entry["loss"]
            record["train_reward_acc"] = entry.get("rewards/accuracies")
            record["train_reward_margin"] = entry.get("rewards/margins")
        if "eval_loss" in entry:
            record["val_loss"] = entry["eval_loss"]
            record["val_reward_acc"] = entry.get("eval_rewards/accuracies")
            record["val_reward_margin"] = entry.get("eval_rewards/margins")
        records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_dir", default="outputs")
    parser.add_argument("--out_dir", default="report/figures")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    sft_path = os.path.join(args.outputs_dir, "sft", "metrics.jsonl")
    if os.path.exists(sft_path):
        plot_sft(read_jsonl(sft_path), args.out_dir)
    for run, title in [("dpo_hh", "DPO on HH-RLHF"), ("dpo_ultrafeedback", "DPO on UltraFeedback")]:
        path = os.path.join(args.outputs_dir, run, "metrics.jsonl")
        if os.path.exists(path):
            plot_dpo(read_jsonl(path), title, args.out_dir, f"{run}.png")
    lora_path = os.path.join(args.outputs_dir, "lora_persona", "log_history.json")
    if os.path.exists(lora_path):
        records = lora_records(read_json(lora_path))
        plot_dpo(records, "LoRA-DPO (TRL) on the persona set", args.out_dir, "lora_persona.png")


if __name__ == "__main__":
    main()
