import argparse
import os
import random

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from alignment.utils import read_json, set_seed, write_json

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument("--data_path", default="data/persona_lora/ramsay_style_preferences.json")
    parser.add_argument("--output_dir", default="outputs/lora_persona")
    parser.add_argument("--val_size", type=int, default=20)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def to_conversation(row):
    return {
        "prompt": [{"role": "user", "content": row["prompt"]}],
        "chosen": [{"role": "assistant", "content": row["chosen"]}],
        "rejected": [{"role": "assistant", "content": row["rejected"]}],
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    write_json(os.path.join(args.output_dir, "config.json"), vars(args))

    rows = read_json(args.data_path)
    random.Random(args.seed).shuffle(rows)
    val_rows, train_rows = rows[:args.val_size], rows[args.val_size:]
    write_json(os.path.join(args.output_dir, "val_rows.json"), val_rows)
    train = Dataset.from_list([to_conversation(row) for row in train_rows])
    val = Dataset.from_list([to_conversation(row) for row in val_rows])
    print(f"{len(train)} train pairs, {len(val)} validation pairs")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32)
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=TARGET_MODULES,
        task_type="CAUSAL_LM",
    )
    training_args = DPOConfig(
        output_dir=args.output_dir,
        beta=args.beta,
        max_length=args.max_length,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=1.0,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=5,
        save_strategy="no",
        fp16=torch.cuda.is_available(),
        bf16=False,
        report_to="none",
        seed=args.seed,
    )
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "adapter"))
    write_json(os.path.join(args.output_dir, "log_history.json"), trainer.state.log_history)
    print(trainer.evaluate())


if __name__ == "__main__":
    main()
