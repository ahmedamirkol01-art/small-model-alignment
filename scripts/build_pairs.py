import argparse
import json
import os
import random

from alignment.utils import read_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_path", required=True)
    parser.add_argument("--baseline_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    candidates = {record["id"]: record for record in read_json(args.candidate_path)}
    baselines = read_json(args.baseline_path)
    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    with open(args.output_path, "w", encoding="utf-8") as f:
        for baseline in baselines:
            candidate = candidates[baseline["id"]]
            first, second = (candidate, baseline) if rng.random() < 0.5 else (baseline, candidate)
            pair = {
                "pair_id": baseline["id"],
                "instruction": baseline["instruction"],
                "category": baseline.get("category"),
                "response_a": first["output"],
                "response_b": second["output"],
                "hidden": {
                    "A": first["generator"],
                    "B": second["generator"],
                    "candidate": candidate["generator"],
                    "baseline": baseline["generator"],
                },
            }
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"saved {len(baselines)} pairs to {args.output_path}")


if __name__ == "__main__":
    main()
