# Small-Model Alignment Starter

This branch is the student starter package for the SFT + DPO alignment assignment. It is intentionally minimal: it provides the tested environment metadata, assignment PDF, fixed evaluation/data files, and no completed training implementation.

## What Is Included

```text
requirements.txt
pyproject.toml
submission_adapter.py
tests/test_public_adapter_smoke.py
tests/toy_lm.py
docs/assignment_manual.pdf
data/persona_lora/ramsay_style_preferences.json
data/persona_lora/README.md
eval_sets/hh_rlhf_eval_300_seed42.json
```

## Recommended Kaggle Setup

Use a Kaggle notebook with the `GPU T4 x2` accelerator. After cloning this repository/branch, run:

```bash
python -m pip uninstall -y -q torchao || true
python -m pip install -q "transformers>=4.43.0,<5" "datasets>=2.18,<5" huggingface_hub accelerate tqdm pyyaml sentencepiece safetensors pytest
python -m pip install -q -e . --no-deps
```

This keeps Kaggle's preinstalled CUDA-compatible PyTorch intact, avoids known stale `torchao` / PEFT conflicts, and makes the local repository importable without replacing the GPU stack.

## What You Need To Build

You will add your own training and evaluation code on top of this starter. At minimum, your project should implement:

- SFT training on 30,000 examples.
- From-scratch DPO on HH-RLHF-style preference data.
- From-scratch DPO on UltraFeedback-style preference data.
- TRL `DPOTrainer` + LoRA training on the provided persona preference data.
- Generation and DPO-vs-SFT evaluation outputs.
- A short final report with plots and analysis.

Read `docs/assignment_manual.pdf` for the full requirements and suggested workflow.

## Submission Adapter And Public Tests

Keep the six function signatures in `submission_adapter.py` unchanged. Implement
each function as a small wrapper around your own SFT/DPO code. The grader uses
this stable interface so your internal file and class organization can remain
your choice.

Run the public tests from the repository root:

```bash
python -m pip install -q pytest
PYTHONPATH=. python -m pytest -q tests/test_public_adapter_smoke.py
```

The tests use a tiny local tokenizer and model, so they do not download a model,
load the assignment datasets, or require a GPU. The starter adapter raises
`NotImplementedError` until you connect it to your implementation. Passing the
public tests checks the interface and a few basic invariants; the private grader
also checks numerical correctness and harder edge cases described in the manual.

For `compute_sft_loss`, the assignment accepts either the mean of per-example
summed response negative log probabilities or the mean over all unmasked response
tokens. Use one reduction consistently and document it in the report.

## Fixed Files

- `eval_sets/hh_rlhf_eval_300_seed42.json` is the fixed synthetic helpful/harmless evaluation set.
- For AlpacaEval-style evaluation, use 300 prompts with seed `42` so comparisons are consistent.
- `data/persona_lora/ramsay_style_preferences.json` is the small preference dataset for the LoRA-DPO persona task.

## Saving Work On Kaggle

Kaggle sessions are temporary. Write important outputs under `/kaggle/working`, save notebook versions, and archive checkpoints/generations as zip files or Kaggle output datasets before ending a session.

## Implementation In This Repository

```text
alignment/sft.py        prompt+response tokenization, response-only labels, log-probs, SFT loss
alignment/dpo.py        chosen/rejected batches, tensor-level DPO loss, model-level DPO loss
alignment/data.py       prompt template and dataset filtering (UltraChat, HH-RLHF, UltraFeedback)
alignment/utils.py      seeding, JSON helpers, padding trim, cosine schedule
submission_adapter.py   thin wrappers around alignment/sft.py and alignment/dpo.py
scripts/train_sft.py            SFT training loop
scripts/train_dpo.py            from-scratch DPO training loop (--dataset hh | ultrafeedback)
scripts/train_lora_dpo.py       TRL DPOTrainer + LoRA persona run
scripts/make_alpaca_subset.py   builds eval_sets/alpaca_eval_300_seed42.json
scripts/generate.py             greedy generation on a fixed evaluation set
scripts/build_pairs.py          randomized A/B pairwise files with hidden model identity
scripts/judge_pairwise_kaggle.py  resumable pairwise judging through kaggle_benchmarks
scripts/plot_metrics.py         training curves
notebooks/01-06                 Kaggle notebooks, run in order
tests/test_edge_cases.py        extra checks for truncation, padding, row order and DPO loss
```

Kaggle order: `01_sft` -> `02_dpo_hh` and `03_dpo_ultrafeedback` (input: 01) -> `04_lora_persona` ->
`05_generate` (inputs: 01, 02, 03) -> `06_judge` (utility benchmark notebook, inputs: 04, 05).
Metrics, generations, pairwise files and judgments are saved under `outputs/`. Model checkpoints are kept
in the Kaggle notebook outputs and are not committed.

Run all tests locally with:

```bash
PYTHONPATH=. python -m pytest -q tests/
```
