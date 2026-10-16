from datasets import load_dataset

SYSTEM_PROMPT = "You are a helpful assistant."


def format_prompt(instruction):
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def parse_ultrachat(row):
    messages = row["messages"]
    if len(messages) < 2 or messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
        return None
    instruction = messages[0]["content"].strip()
    response = messages[1]["content"].strip()
    if not instruction or not response:
        return None
    return {"prompt": format_prompt(instruction), "response": response}


def split_hh(text):
    if text.count("\n\nHuman:") != 1 or text.count("\n\nAssistant:") != 1:
        return None
    human, assistant = text.split("\n\nAssistant:")
    if not human.startswith("\n\nHuman:"):
        return None
    return human[len("\n\nHuman:"):].strip(), assistant.strip()


def parse_hh(row):
    chosen = split_hh(row["chosen"])
    rejected = split_hh(row["rejected"])
    if chosen is None or rejected is None or chosen[0] != rejected[0]:
        return None
    instruction, chosen_text, rejected_text = chosen[0], chosen[1], rejected[1]
    if not instruction or not chosen_text or not rejected_text or chosen_text == rejected_text:
        return None
    return {"prompt": format_prompt(instruction), "chosen": chosen_text, "rejected": rejected_text}


def parse_ultrafeedback(row):
    if len(row["chosen"]) != 2 or len(row["rejected"]) != 2:
        return None
    if row["score_chosen"] <= row["score_rejected"]:
        return None
    instruction = row["prompt"].strip()
    chosen_text = row["chosen"][1]["content"].strip()
    rejected_text = row["rejected"][1]["content"].strip()
    if not instruction or not chosen_text or not rejected_text or chosen_text == rejected_text:
        return None
    return {"prompt": format_prompt(instruction), "chosen": chosen_text, "rejected": rejected_text}


SOURCES = {
    "ultrachat": ("HuggingFaceH4/ultrachat_200k", "train_sft", "test_sft", parse_ultrachat),
    "hh": ("Anthropic/hh-rlhf", "train", "test", parse_hh),
    "ultrafeedback": ("HuggingFaceH4/ultrafeedback_binarized", "train_prefs", "test_prefs", parse_ultrafeedback),
}


def count_tokens(tokenizer, text):
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def fits(tokenizer, example, max_prompt_length, max_length):
    prompt_length = count_tokens(tokenizer, example["prompt"])
    if prompt_length > max_prompt_length:
        return False
    if max_length is None:
        return True
    return prompt_length + count_tokens(tokenizer, example["response"]) + 1 <= max_length


def select_examples(dataset, parse, tokenizer, size, max_prompt_length, max_length, seed):
    selected, scanned = [], 0
    for row in dataset.shuffle(seed=seed):
        scanned += 1
        example = parse(row)
        if example is not None and fits(tokenizer, example, max_prompt_length, max_length):
            selected.append(example)
            if len(selected) == size:
                break
    return selected, scanned


def load_examples(name, tokenizer, train_size, val_size, max_prompt_length, max_length, seed):
    path, train_split, val_split, parse = SOURCES[name]
    train, train_scanned = select_examples(
        load_dataset(path, split=train_split), parse, tokenizer, train_size, max_prompt_length, max_length, seed
    )
    val, val_scanned = select_examples(
        load_dataset(path, split=val_split), parse, tokenizer, val_size, max_prompt_length, max_length, seed
    )
    summary = {
        "dataset": path,
        "train_split": train_split,
        "val_split": val_split,
        "train_examples": len(train),
        "val_examples": len(val),
        "train_rows_scanned": train_scanned,
        "val_rows_scanned": val_scanned,
        "max_prompt_length": max_prompt_length,
        "max_length_filter": max_length,
        "seed": seed,
    }
    return train, val, summary
