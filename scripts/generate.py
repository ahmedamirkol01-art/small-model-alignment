import argparse
import os

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignment.data import format_prompt
from alignment.jsonio import read_json, write_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--eval_set", required=True)
    parser.add_argument("--generator", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def build_prompt(tokenizer, instruction, use_chat_template):
    if use_chat_template:
        messages = [{"role": "user", "content": instruction}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return format_prompt(instruction)


def main():
    args = parse_args()
    if os.path.exists(args.output_path):
        print(f"skipping, {args.output_path} already exists")
        return
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype).to(device)
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    stop_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")]

    records = read_json(args.eval_set)[:args.limit]
    outputs = []
    for start in tqdm(range(0, len(records), args.batch_size)):
        chunk = records[start:start + args.batch_size]
        prompts = [build_prompt(tokenizer, record["instruction"], args.use_chat_template) for record in chunk]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                repetition_penalty=args.repetition_penalty,
                eos_token_id=stop_ids,
                pad_token_id=tokenizer.pad_token_id,
            )
        texts = tokenizer.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        for record, text in zip(chunk, texts):
            output = {
                "id": record["id"],
                "instruction": record["instruction"],
                "output": text.strip(),
                "generator": args.generator,
                "dataset": record["dataset"],
            }
            if "category" in record:
                output["category"] = record["category"]
            outputs.append(output)

    write_json(args.output_path, outputs)
    print(f"saved {len(outputs)} generations to {args.output_path}")
    for output in outputs[:3]:
        print("=" * 80)
        print(output["instruction"][:300])
        print("-" * 80)
        print(output["output"][:800])


if __name__ == "__main__":
    main()
