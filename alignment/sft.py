import torch


def tokenize_example(tokenizer, prompt, response, max_length):
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    response_ids = tokenizer(response, add_special_tokens=False).input_ids
    if tokenizer.eos_token_id is not None:
        response_ids = response_ids + [tokenizer.eos_token_id]
    input_ids = (prompt_ids + response_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + response_ids)[:max_length]
    return input_ids, labels


def get_pad_id(tokenizer):
    if tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        return tokenizer.eos_token_id
    return 0


def build_sft_batch(tokenizer, prompts, responses, max_length):
    pad_id = get_pad_id(tokenizer)
    input_ids, attention_mask, labels = [], [], []
    for prompt, response in zip(prompts, responses):
        ids, row_labels = tokenize_example(tokenizer, prompt, response, max_length)
        padding = max_length - len(ids)
        input_ids.append(ids + [pad_id] * padding)
        attention_mask.append([1] * len(ids) + [0] * padding)
        labels.append(row_labels + [-100] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def token_logprobs(model, batch):
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"][:, 1:].to(device)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1].float()
    mask = labels != -100
    targets = labels.masked_fill(~mask, 0).unsqueeze(-1)
    logprobs = torch.gather(logits.log_softmax(dim=-1), 2, targets).squeeze(-1)
    return logprobs * mask, mask


def response_logprobs(model, batch):
    logprobs, _ = token_logprobs(model, batch)
    return logprobs.sum(dim=-1)


def sft_loss(model, batch):
    logprobs, mask = token_logprobs(model, batch)
    return -logprobs.sum() / mask.sum().clamp(min=1)
