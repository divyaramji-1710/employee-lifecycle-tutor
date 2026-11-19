"""Continued pretraining on ordered synthetic HR histories."""
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, set_seed

from app.event_encoding import (
    MODEL_NAME, prepare_tokenizer, encode_history,
)

def train_model(
    output_dir="artifacts/hr-model",
    max_steps=200,
    batch_size=8,
    learning_rate=1e-4,
):
    set_seed(42)
    torch.set_num_threads(2)
    rng = random.Random(42)
    tokenizer = prepare_tokenizer()

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model = model.float().cpu()
    model.resize_token_embeddings(
        len(tokenizer), mean_resizing=False
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    def load(name):
        rows = [
            json.loads(line)
            for line in (Path("data") / f"{name}.jsonl")
                .read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
        return [encode_history(tokenizer, row) for row in rows]

    training = load("train")
    validation = load("validation")

    def collate(rows):
        width = max(len(row["input_ids"]) for row in rows)
        batch = {}
        for key in ("input_ids", "attention_mask", "labels"):
            fill = {
                "input_ids": tokenizer.pad_token_id,
                "attention_mask": 0,
                "labels": -100,
            }[key]
            batch[key] = torch.tensor([
                row[key] + [fill] * (width - len(row[key]))
                for row in rows
            ], dtype=torch.long)
        return batch

    @torch.inference_mode()
    def validation_loss():
        model.eval()
        total_loss = total_targets = 0
        for start in range(0, len(validation), batch_size):
            batch = collate(validation[start:start + batch_size])
            targets = (batch["labels"][:, 1:] != -100).sum().item()
            loss = model(**batch, use_cache=False).loss
            total_loss += loss.item() * targets
            total_targets += targets
        return total_loss / total_targets

    initial_loss = validation_loss()
    print("Training histories:", len(training), flush=True)
    print("Validation histories:", len(validation), flush=True)
    print(f"Initial validation loss: {initial_loss:.4f}", flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01
    )
    history = []
    started = time.perf_counter()

    for step in range(1, max_steps + 1):
        model.train()
        selected = rng.sample(
            training, min(batch_size, len(training))
        )
        batch = collate(selected)

        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch, use_cache=False).loss
        if not torch.isfinite(loss):
            raise RuntimeError("Training loss is not finite.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        history.append({"step": step, "loss": loss.item()})
        if step == 1 or step % 5 == 0:
            print(
                f"Step {step}/{max_steps} | loss {loss.item():.4f} | "
                f"elapsed {time.perf_counter() - started:.1f}s",
                flush=True,
            )

    optimizer.zero_grad(set_to_none=True)
    final_loss = validation_loss()
    training_seconds = time.perf_counter() - started

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination)
    tokenizer.save_pretrained(destination)

    report = {
        "base_model": MODEL_NAME,
        "steps": max_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "training_histories": len(training),
        "validation_histories": len(validation),
        "initial_validation_loss": initial_loss,
        "final_validation_loss": final_loss,
        "training_and_final_evaluation_seconds": training_seconds,
        "history": history,
    }
    (destination / "training_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print(f"Final validation loss: {final_loss:.4f}", flush=True)
    print(f"Training and final evaluation: {training_seconds:.1f}s")
    print("Saved model, tokenizer and report:", destination)
    return report
