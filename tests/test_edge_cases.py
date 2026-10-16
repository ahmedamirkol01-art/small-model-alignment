import copy
import math

import pytest
import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from alignment.dpo import build_dpo_batch, concat_pair, dpo_loss, dpo_loss_from_models, pair_logprobs
from alignment.sft import build_sft_batch, response_logprobs, sft_loss

PROMPTS = ["<|im_start|>user\nName a fruit.<|im_end|>\n<|im_start|>assistant\n", "Q: 2+2?\nA:"]
RESPONSES = ["An apple is a fruit.", " 4"]


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")


@pytest.fixture
def model():
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=151936,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=True,
    )
    return Qwen2ForCausalLM(config).eval()


def test_eos_is_kept_at_exact_length_and_dropped_when_truncated(tokenizer):
    n = len(tokenizer(PROMPTS[0], add_special_tokens=False).input_ids) + len(
        tokenizer(RESPONSES[0], add_special_tokens=False).input_ids
    ) + 1
    exact = build_sft_batch(tokenizer, PROMPTS[:1], RESPONSES[:1], n)
    assert exact["input_ids"][0, -1] == tokenizer.eos_token_id
    assert exact["labels"][0, -1] == tokenizer.eos_token_id
    cut = build_sft_batch(tokenizer, PROMPTS[:1], RESPONSES[:1], n - 1)
    assert cut["input_ids"].shape == (1, n - 1)
    assert tokenizer.eos_token_id not in cut["labels"][0].tolist()


def test_pad_equal_to_eos_still_supervises_eos(tokenizer):
    assert tokenizer.pad_token_id == tokenizer.eos_token_id
    batch = build_sft_batch(tokenizer, PROMPTS, RESPONSES, 64)
    for row in range(2):
        length = int(batch["attention_mask"][row].sum())
        assert batch["labels"][row, length - 1] == tokenizer.eos_token_id
        assert (batch["labels"][row, length:] == -100).all()
        assert (batch["input_ids"][row, length:] == tokenizer.pad_token_id).all()


def test_prompt_tokens_are_masked(tokenizer):
    batch = build_sft_batch(tokenizer, PROMPTS, RESPONSES, 64)
    prompt_length = len(tokenizer(PROMPTS[0], add_special_tokens=False).input_ids)
    assert (batch["labels"][0, :prompt_length] == -100).all()
    assert (batch["labels"][0, prompt_length:prompt_length + 3] != -100).all()


def test_left_padding_side_is_ignored(tokenizer):
    left = copy.deepcopy(tokenizer)
    left.padding_side = "left"
    a = build_sft_batch(tokenizer, PROMPTS, RESPONSES, 64)
    b = build_sft_batch(left, PROMPTS, RESPONSES, 64)
    assert all(torch.equal(a[key], b[key]) for key in a)


def test_prompt_longer_than_max_length_gives_zero_score(tokenizer, model):
    batch = build_sft_batch(tokenizer, PROMPTS[:1], RESPONSES[:1], 4)
    assert (batch["labels"] == -100).all()
    assert response_logprobs(model, batch).item() == 0.0
    loss = sft_loss(model, batch)
    assert torch.isfinite(loss) and loss.item() == 0.0


def test_logprobs_match_manual_computation(tokenizer, model):
    batch = build_sft_batch(tokenizer, PROMPTS, RESPONSES, 64)
    scores = response_logprobs(model, batch)
    for row, (prompt, response) in enumerate(zip(PROMPTS, RESPONSES)):
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        response_ids = tokenizer(response, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
        ids = torch.tensor([prompt_ids + response_ids])
        logprobs = model(input_ids=ids).logits[0].float().log_softmax(-1)
        expected = sum(logprobs[len(prompt_ids) + i - 1, token].item() for i, token in enumerate(response_ids))
        assert scores[row].item() == pytest.approx(expected, abs=1e-4)


def test_row_order_and_padding_do_not_change_scores(tokenizer, model):
    together = response_logprobs(model, build_sft_batch(tokenizer, PROMPTS, RESPONSES, 64))
    reversed_scores = response_logprobs(model, build_sft_batch(tokenizer, PROMPTS[::-1], RESPONSES[::-1], 64))
    alone = [response_logprobs(model, build_sft_batch(tokenizer, [p], [r], 32)) for p, r in zip(PROMPTS, RESPONSES)]
    assert torch.allclose(together, reversed_scores.flip(0), atol=1e-4)
    assert torch.allclose(together, torch.cat(alone), atol=1e-4)


def test_sft_loss_is_token_mean(tokenizer, model):
    batch = build_sft_batch(tokenizer, PROMPTS, RESPONSES, 64)
    tokens = (batch["labels"][:, 1:] != -100).sum()
    expected = -response_logprobs(model, batch).sum() / tokens
    assert sft_loss(model, batch).item() == pytest.approx(expected.item(), abs=1e-5)


def test_tensor_dpo_loss_matches_formula():
    pc, pr = torch.tensor([-3.0, -5.0]), torch.tensor([-4.0, -2.0])
    rc, rr = torch.tensor([-3.5, -4.0]), torch.tensor([-3.0, -3.0])
    loss, chosen_rewards, rejected_rewards = dpo_loss(pc, pr, rc, rr, 0.1)
    expected = sum(-math.log(1 / (1 + math.exp(-0.1 * ((a - b) - (c - d))))) for a, b, c, d in zip(pc, pr, rc, rr)) / 2
    assert loss.item() == pytest.approx(float(expected), abs=1e-6)
    assert torch.allclose(chosen_rewards, 0.1 * (pc - rc))
    assert torch.allclose(rejected_rewards, 0.1 * (pr - rr))


def test_dpo_starts_at_log2_and_only_policy_gets_gradients(tokenizer, model):
    reference = copy.deepcopy(model)
    loss = dpo_loss_from_models(model, reference, tokenizer, PROMPTS, RESPONSES, ["No idea.", " 5"], 0.1, 64)
    assert loss.item() == pytest.approx(math.log(2), abs=1e-5)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert all(p.grad is None for p in reference.parameters())


def test_one_step_increases_preference_margin(tokenizer, model):
    reference = copy.deepcopy(model)
    pair_batch = concat_pair(build_dpo_batch(tokenizer, PROMPTS, RESPONSES, ["No idea.", " 5"], 64))
    with torch.no_grad():
        reference_chosen, reference_rejected = pair_logprobs(reference, pair_batch)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    margins = []
    for _ in range(2):
        policy_chosen, policy_rejected = pair_logprobs(model, pair_batch)
        loss, chosen_rewards, rejected_rewards = dpo_loss(
            policy_chosen, policy_rejected, reference_chosen, reference_rejected, 0.1
        )
        margins.append((chosen_rewards - rejected_rewards).mean().item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert margins[1] > margins[0]
