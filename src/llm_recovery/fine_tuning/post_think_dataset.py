from typing import List, Dict
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import torch


class PostThinkDataset(Dataset[Dict[str, torch.Tensor]]):
    """
    A dataset where the prompt ends with <think>, and the model generates reasoning and an answer.
    Loss is computed only on the part of the answer after </think>.
    """

    def __init__(self, instructions: List[str], answers: List[str], tokenizer: PreTrainedTokenizer):
        self.instructions = instructions
        self.answers = answers
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        prompt = self.instructions[idx]
        full_answer = self.answers[idx]

        # Tokenize prompt (includes <think>)
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
        prompt_input_ids = prompt_tokens["input_ids"]
        prompt_attention_mask = prompt_tokens["attention_mask"]

        # Tokenize full answer (may include reasoning and </think>)
        answer_tokens = self.tokenizer(full_answer, add_special_tokens=False)
        answer_input_ids = answer_tokens["input_ids"]
        answer_attention_mask = answer_tokens["attention_mask"]

        # Find index of </think> in tokenized answer
        end_think_ids = self.tokenizer("</think>", add_special_tokens=False)["input_ids"]
        end_idx = -1
        for i in range(len(answer_input_ids) - len(end_think_ids) + 1):
            if answer_input_ids[i:i + len(end_think_ids)] == end_think_ids:
                end_idx = i + len(end_think_ids)
                break

        # Apply label masking: everything before and including </think> is ignored
        if end_idx != -1:
            label_ids = [-100] * end_idx + answer_input_ids[end_idx:]
        else:
            # If </think> not found, treat entire answer as label
            label_ids = answer_input_ids

        input_ids = prompt_input_ids + answer_input_ids
        attention_mask = prompt_attention_mask + answer_attention_mask
        labels = [-100] * len(prompt_input_ids) + label_ids

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def collate(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = [b["input_ids"] for b in batch]
        attention_mask = [b["attention_mask"] for b in batch]
        labels = [b["labels"] for b in batch]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
