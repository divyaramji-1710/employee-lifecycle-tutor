"""Next-event predictions from the locally trained language model."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.event_encoding import EVENT_TOKENS

class LifecycleTutor:
    def __init__(self, model_dir="artifacts/hr-model"):
        torch.set_num_threads(2)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir).eval()
        self.event_ids = {
            event: self.tokenizer.convert_tokens_to_ids(token)
            for event, token in EVENT_TOKENS.items()
        }
        self.id_to_event = {
            token_id: event for event, token_id in self.event_ids.items()
        }

    @torch.inference_mode()
    def predict(self, events, top_k=3):
        if not events:
            raise ValueError("Enter at least one event.")
        unknown = [event for event in events if event not in self.event_ids]
        if unknown:
            raise ValueError(f"Unknown events: {unknown}")
        if not 1 <= top_k <= len(self.event_ids) + 1:
            raise ValueError("Invalid number of suggestions.")

        ids = [self.tokenizer.bos_token_id] + [
            self.event_ids[event] for event in events
        ]
        if len(ids) >= self.model.config.n_positions:
            raise ValueError("History exceeds the model context limit.")

        inputs = torch.tensor([ids], dtype=torch.long)
        output = self.model(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            use_cache=False,
        )
        logits = output.logits[0, -1].float()

        # Keep the unrestricted prediction visible, including unknown output.
        unrestricted_id = logits.argmax().item()
        if unrestricted_id == self.tokenizer.eos_token_id:
            unrestricted = "end_of_history"
        else:
            unrestricted = self.id_to_event.get(unrestricted_id, "unknown")

        candidate_names = list(self.event_ids) + ["end_of_history"]
        candidate_ids = list(self.event_ids.values()) + [
            self.tokenizer.eos_token_id
        ]
        candidate_logits = logits[candidate_ids]
        probabilities = torch.softmax(candidate_logits, dim=0)
        values, positions = probabilities.topk(top_k)

        return {
            "unrestricted_prediction": unrestricted,
            "suggestions": [
                {
                    "event": candidate_names[position],
                    "conditional_score": float(value),
                }
                for value, position in zip(values.tolist(), positions.tolist())
            ],
        }
