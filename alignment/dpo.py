import torch
import torch.nn.functional as F

from alignment.sft import build_sft_batch, response_logprobs


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    chosen = build_sft_batch(tokenizer, prompts, chosen_responses, max_length)
    rejected = build_sft_batch(tokenizer, prompts, rejected_responses, max_length)
    return {"chosen": chosen, "rejected": rejected}


def concat_pair(dpo_batch):
    chosen, rejected = dpo_batch["chosen"], dpo_batch["rejected"]
    return {key: torch.cat([chosen[key], rejected[key]]) for key in chosen}


def pair_logprobs(model, pair_batch):
    logprobs = response_logprobs(model, pair_batch)
    half = logprobs.shape[0] // 2
    return logprobs[:half], logprobs[half:]


def dpo_loss(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta):
    device = policy_chosen_logps.device
    reference_chosen_logps = reference_chosen_logps.to(device)
    reference_rejected_logps = reference_rejected_logps.to(device)
    policy_gap = policy_chosen_logps - policy_rejected_logps
    reference_gap = reference_chosen_logps - reference_rejected_logps
    loss = -F.logsigmoid(beta * (policy_gap - reference_gap)).mean()
    chosen_rewards = (beta * (policy_chosen_logps - reference_chosen_logps)).detach()
    rejected_rewards = (beta * (policy_rejected_logps - reference_rejected_logps)).detach()
    return loss, chosen_rewards, rejected_rewards


def dpo_loss_from_models(policy_model, reference_model, tokenizer, prompts, chosen_responses, rejected_responses, beta, max_length):
    pair_batch = concat_pair(build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length))
    policy_chosen, policy_rejected = pair_logprobs(policy_model, pair_batch)
    with torch.no_grad():
        reference_chosen, reference_rejected = pair_logprobs(reference_model, pair_batch)
    loss, _, _ = dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta)
    return loss
