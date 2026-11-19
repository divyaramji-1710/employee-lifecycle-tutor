"""Ordered event-token encoding for causal language-model training."""
from transformers import AutoTokenizer
from app.event_data import build_histories

MODEL_NAME = "distilgpt2"

# Fixed vocabulary from the synthetic scenario definitions.
EVENTS = sorted({
    event
    for row in build_histories()
    for event in row["events"]
})
EVENT_TOKENS = {
    event: f"<event:{event}>" for event in EVENTS
}

def prepare_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({
        "additional_special_tokens": list(EVENT_TOKENS.values())
    })
    return tokenizer

def encode_history(tokenizer, row):
    events = row["events"]
    if not events:
        raise ValueError("A history must contain at least one event.")

    unknown = [event for event in events if event not in EVENT_TOKENS]
    if unknown:
        raise ValueError(f"Unknown events: {unknown}")

    event_ids = [
        tokenizer.convert_tokens_to_ids(EVENT_TOKENS[event])
        for event in events
    ]
    ids = (
        [tokenizer.bos_token_id]
        + event_ids
        + [tokenizer.eos_token_id]
    )

    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "labels": ids.copy(),
    }
