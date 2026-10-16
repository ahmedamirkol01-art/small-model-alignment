import argparse
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignment.data import load_examples
from alignment.dpo import build_dpo_batch, concat_pair, dpo_loss, pair_logprobs
from alignment.jsonio import append_jsonl, write_json
from alignment.utils import set_seed, trim_padding


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["hh", "ultrafeedback"], required=True)
    parser.add_argument("--sft_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_size", type=int, default=15000)
    parser.add_argument("--val_size", type=int, default=1000)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_prompt_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_pair_batch(tokenizer, examples, max_length):
    prompts = [example["prompt"] for example in examples]
    chosen = [example["chosen"] for example in examples]
    rejected = [example["rejected"] for example in examples]
    return trim_padding(concat_pair(build_dpo_batch(tokenizer, prompts, chosen, rejected, max_length)))


def dataset_logprobs(model, tokenizer, examples, args, use_amp):
    chosen, rejected = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
        for start in range(0, len(examples), args.eval_batch_size):
            batch = make_pair_batch(tokenizer, examples[start:start + args.eval_batch_size], args.max_length)
            chosen_logps, rejected_logps = pair_logprobs(model, batch)
            chosen.append(chosen_logps.cpu())
            rejected.append(rejected_logps.cpu())
    return torch.cat(chosen), torch.cat(rejected)


def reward_metrics(loss, chosen_rewards, rejected_rewards, prefix):
    return {
        f"{prefix}_loss": loss.item(),
        f"{prefix}_reward_acc": (chosen_rewards > rejected_rewards).float().mean().item(),
        f"{prefix}_reward_margin": (chosen_rewards - rejected_rewards).mean().item(),
        f"{prefix}_chosen_reward": chosen_rewards.mean().item(),
        f"{prefix}_rejected_reward": rejected_rewards.mean().item(),
    }


def evaluate(policy, tokenizer, val, reference_val, args, use_amp):
    policy.eval()
    policy_chosen, policy_rejected = dataset_logprobs(policy, tokenizer, val, args, use_amp)
    policy.train()
    loss, chosen_rewards, rejected_rewards = dpo_loss(policy_chosen, policy_rejected, *reference_val, args.beta)
    return reward_metrics(loss, chosen_rewards, rejected_rewards, "val")


def save(model, tokenizer, path):
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    write_json(os.path.join(args.output_dir, "config.json"), vars(args))
    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    open(metrics_path, "w").close()

    use_amp = torch.cuda.is_available()
    policy_device = "cuda:0" if use_amp else "cpu"
    reference_device = "cuda:1" if torch.cuda.device_count() > 1 else policy_device
    tokenizer = AutoTokenizer.from_pretrained(args.sft_path)
    policy = AutoModelForCausalLM.from_pretrained(
        args.sft_path, torch_dtype=torch.float32, attn_implementation="sdpa"
    ).to(policy_device)
    policy.train()
    reference = AutoModelForCausalLM.from_pretrained(
        args.sft_path, torch_dtype=torch.float32, attn_implementation="sdpa"
    ).to(reference_device)
    reference.eval()
    reference.requires_grad_(False)

    train, val, summary = load_examples(
        args.dataset, tokenizer, args.train_size, args.val_size, args.max_prompt_length, None, args.seed
    )
    write_json(os.path.join(args.output_dir, "data_summary.json"), summary)
    print(summary)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    total_steps = len(train) // (args.batch_size * args.grad_accum)

    reference_val = dataset_logprobs(reference, tokenizer, val, args, use_amp)
    record = {"step": 0, **evaluate(policy, tokenizer, val, reference_val, args, use_amp)}
    append_jsonl(metrics_path, record)
    print(record)
    best_acc = -1.0

    for step in tqdm(range(1, total_steps + 1)):
        losses, chosen_rewards, rejected_rewards = [], [], []
        for micro in range(args.grad_accum):
            start = ((step - 1) * args.grad_accum + micro) * args.batch_size
            batch = make_pair_batch(tokenizer, train[start:start + args.batch_size], args.max_length)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                policy_chosen, policy_rejected = pair_logprobs(policy, batch)
                with torch.no_grad():
                    reference_chosen, reference_rejected = pair_logprobs(reference, batch)
            loss, chosen, rejected = dpo_loss(
                policy_chosen, policy_rejected, reference_chosen, reference_rejected, args.beta
            )
            scaler.scale(loss / args.grad_accum).backward()
            losses.append(loss.detach())
            chosen_rewards.append(chosen)
            rejected_rewards.append(rejected)

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        loss = torch.stack(losses).mean()
        record = {"step": step, **reward_metrics(loss, torch.cat(chosen_rewards), torch.cat(rejected_rewards), "train")}
        record["grad_norm"] = grad_norm.item()
        if step % args.eval_every == 0 or step == total_steps:
            record.update(evaluate(policy, tokenizer, val, reference_val, args, use_amp))
            print(f"step {step} | val_loss {record['val_loss']:.4f} | val_acc {record['val_reward_acc']:.3f}")
            if record["val_reward_acc"] > best_acc:
                best_acc = record["val_reward_acc"]
                record["saved_best"] = True
                save(policy, tokenizer, os.path.join(args.output_dir, "best"))
        append_jsonl(metrics_path, record)

    save(policy, tokenizer, os.path.join(args.output_dir, "final"))
    print(f"done | best val_reward_acc {best_acc:.3f}")


if __name__ == "__main__":
    main()
