import argparse
import glob
import os
import re

from transformers import AutoTokenizer

from alignment.jsonio import read_json, read_jsonl, write_json

REFUSAL = re.compile(
    r"i'?m sorry|i am sorry|i can'?t|i cannot|i won'?t|i will not|as an ai|i'?m not able|i am not able|not appropriate"
)
TOKENIZERS = {"qwen": "Qwen/Qwen2.5-0.5B", "smollm2": "HuggingFaceTB/SmolLM2-360M-Instruct"}


def generation_stats(path, tokenizers, cap):
    records = read_json(path)
    tokenizer = tokenizers["smollm2" if "smollm2" in records[0]["generator"] else "qwen"]
    lengths = [len(tokenizer(record["output"], add_special_tokens=False).input_ids) for record in records]
    refusals = [bool(REFUSAL.search(record["output"].lower())) for record in records]
    by_category = {}
    for record, refused in zip(records, refusals):
        counts = by_category.setdefault(record.get("category", "all"), [0, 0])
        counts[0] += refused
        counts[1] += 1
    return {
        "generator": records[0]["generator"],
        "n": len(records),
        "mean_tokens": sum(lengths) / len(lengths),
        "hit_cap": sum(length >= cap for length in lengths) / len(lengths),
        "refusal_rate": sum(refusals) / len(refusals),
        "refusal_by_category": {key: refused / total for key, (refused, total) in by_category.items()},
    }


def judge_stats(path):
    summary = read_json(path.replace(".jsonl", "_summary.json"))
    pairs = {pair["pair_id"]: pair for pair in read_jsonl(summary["pairs_path"])}
    judged = read_jsonl(path)
    decided = [record for record in judged if record["winner_position"] in ("A", "B")]
    longer_wins = 0
    for record in decided:
        pair = pairs[record["pair_id"]]
        winner, loser = ("response_a", "response_b") if record["winner_position"] == "A" else ("response_b", "response_a")
        longer_wins += len(pair[winner].split()) > len(pair[loser].split())
    return {
        "judged": len(judged),
        "retried": sum(record["attempts"] > 1 for record in judged),
        "position_a_rate": sum(record["winner_position"] == "A" for record in decided) / len(decided),
        "longer_wins_rate": longer_wins / len(decided),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_dir", default="outputs")
    parser.add_argument("--cap", type=int, default=510)
    args = parser.parse_args()

    tokenizers = {name: AutoTokenizer.from_pretrained(path) for name, path in TOKENIZERS.items()}
    generations = {
        os.path.basename(path): generation_stats(path, tokenizers, args.cap)
        for path in sorted(glob.glob(os.path.join(args.outputs_dir, "generations", "*.json")))
    }
    judgments = {
        os.path.basename(path): judge_stats(path)
        for path in sorted(glob.glob(os.path.join(args.outputs_dir, "eval", "*_judged.jsonl")))
    }
    for name, stats in generations.items():
        print(f"{name:34s} tokens {stats['mean_tokens']:6.1f}  hit_cap {stats['hit_cap']:.2f}  refusal {stats['refusal_rate']:.2f}")
    for name, stats in judgments.items():
        print(f"{name:48s} {stats}")
    write_json(os.path.join(args.outputs_dir, "eval", "analysis.json"), {"generations": generations, "judgments": judgments})


if __name__ == "__main__":
    main()
