import argparse
import random

from huggingface_hub import hf_hub_download

from alignment.utils import read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", default="eval_sets/alpaca_eval_300_seed42.json")
    args = parser.parse_args()

    path = hf_hub_download("tatsu-lab/alpaca_eval", "alpaca_eval.json", repo_type="dataset")
    rows = random.Random(args.seed).sample(read_json(path), args.size)
    records = [
        {
            "id": f"alpaca-eval-{args.seed}-{index:04d}",
            "instruction": row["instruction"],
            "dataset": "alpaca_eval",
            "source": row["dataset"],
        }
        for index, row in enumerate(rows)
    ]
    write_json(args.output_path, records)
    print(f"saved {len(records)} prompts to {args.output_path}")


if __name__ == "__main__":
    main()
