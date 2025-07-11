from typing import List, Dict
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import torch


class ExamplesDataset(Dataset[Dict[str, torch.Tensor]]):
    """
    A torch dataset of prompt-answer examples
    """

    def __init__(self, instructions: List["str"], answers: List["str"], tokenizer: PreTrainedTokenizer):
        """
        Initializes the dataset with given lists of instructions and answers

        :param instructions: the list of instructions
        :param answers: the list of answers
        :param tokenizer: the LLM tokenizer
        """
        self.instructions = instructions
        self.answers = answers
        self.tokenizer = tokenizer

    def __len__(self):
        """
        :return: The length of the dataset
        """
        return len(self.instructions)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        prompt = self.instructions[idx]
        answer = self.answers[idx]
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
        answer_tokens = self.tokenizer(answer, add_special_tokens=False)
        input_ids = prompt_tokens["input_ids"] + answer_tokens["input_ids"]
        attention_mask = prompt_tokens["attention_mask"] + answer_tokens["attention_mask"]
        # Label: -100 for prompt tokens (ignored), actual ids for answer tokens
        labels = [-100] * len(prompt_tokens["input_ids"]) + answer_tokens["input_ids"]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def collate(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = [b["input_ids"] for b in batch]
        attention_mask = [b["attention_mask"] for b in batch]
        labels = [b["labels"] for b in batch]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True,
                                                    padding_value=self.tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
