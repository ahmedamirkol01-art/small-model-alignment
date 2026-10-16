import argparse
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignment.data import load_examples
from alignment.sft import build_sft_batch, sft_loss, token_logprobs
from alignment.jsonio import append_jsonl, write_json
from alignment.utils import cosine_with_warmup, set_seed, trim_padding


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--output_dir", default="outputs/sft")
    parser.add_argument("--train_size", type=int, default=30000)
    parser.add_argument("--val_size", type=int, default=1000)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_batch(tokenizer, examples, max_length):
    prompts = [example["prompt"] for example in examples]
    responses = [example["response"] for example in examples]
    return trim_padding(build_sft_batch(tokenizer, prompts, responses, max_length))


def evaluate(model, tokenizer, examples, args, use_amp):
    model.eval()
    total_nll, total_tokens = 0.0, 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
        for start in range(0, len(examples), args.eval_batch_size):
            batch = make_batch(tokenizer, examples[start:start + args.eval_batch_size], args.max_length)
            logprobs, mask = token_logprobs(model, batch)
            total_nll -= logprobs.sum().item()
            total_tokens += mask.sum().item()
    model.train()
    return total_nll / total_tokens


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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32, attn_implementation="sdpa"
    ).to(device)
    model.train()

    train, val, summary = load_examples(
        "ultrachat", tokenizer, args.train_size, args.val_size, args.max_length, args.max_length, args.seed
    )
    write_json(os.path.join(args.output_dir, "data_summary.json"), summary)
    print(summary)

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
    )
    examples_per_step = args.batch_size * args.grad_accum
    total_steps = len(train) // examples_per_step
    scheduler = cosine_with_warmup(optimizer, round(args.warmup_ratio * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    val_loss = evaluate(model, tokenizer, val, args, use_amp)
    append_jsonl(metrics_path, {"step": 0, "val_loss": val_loss})
    print(f"step 0 | val_loss {val_loss:.4f}")
    best_val_loss = float("inf")

    for step in tqdm(range(1, total_steps + 1)):
        step_loss = 0.0
        for micro in range(args.grad_accum):
            start = ((step - 1) * args.grad_accum + micro) * args.batch_size
            batch = make_batch(tokenizer, train[start:start + args.batch_size], args.max_length)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = sft_loss(model, batch) / args.grad_accum
            scaler.scale(loss).backward()
            step_loss += loss.item()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        lr = optimizer.param_groups[0]["lr"]
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        record = {"step": step, "train_loss": step_loss, "lr": lr, "grad_norm": grad_norm.item()}
        if step % args.eval_every == 0 or step == total_steps:
            record["val_loss"] = evaluate(model, tokenizer, val, args, use_amp)
            print(f"step {step} | train_loss {step_loss:.4f} | val_loss {record['val_loss']:.4f}")
            if record["val_loss"] < best_val_loss:
                best_val_loss = record["val_loss"]
                record["saved_best"] = True
                save(model, tokenizer, os.path.join(args.output_dir, "best"))
        append_jsonl(metrics_path, record)

    save(model, tokenizer, os.path.join(args.output_dir, "final"))
    print(f"done | best val_loss {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
