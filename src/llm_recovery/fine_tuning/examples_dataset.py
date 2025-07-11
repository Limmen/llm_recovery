from typing import List, Dict
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import torch
import llm_recovery.constants.constants as constants


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
        input_ids = prompt_tokens[constants.GENERAL.INPUT_IDS] + answer_tokens[constants.GENERAL.INPUT_IDS]
        attention_mask = (prompt_tokens[constants.GENERAL.ATTENTION_MASK] +
                          answer_tokens[constants.GENERAL.ATTENTION_MASK])
        # Label: -100 for prompt tokens (ignored), actual ids for answer tokens
        labels = [-100] * len(prompt_tokens[constants.GENERAL.INPUT_IDS]) + answer_tokens[constants.GENERAL.INPUT_IDS]
        return {
            constants.GENERAL.INPUT_IDS: torch.tensor(input_ids, dtype=torch.long),
            constants.GENERAL.ATTENTION_MASK: torch.tensor(attention_mask, dtype=torch.long),
            constants.GENERAL.LABELS: torch.tensor(labels, dtype=torch.long),
        }

    def collate(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Takes a batch of tokenized samples, pads them so they have the same length, and  returns a dictionary of
        input_ids (tokenized ids), attention_mask (tokenized mask), and labels, which can be used for supervised
        fine-tuning.

        :param batch: the batch to process
        :return: the processed batch
        """
        input_ids = [b[constants.GENERAL.INPUT_IDS] for b in batch]
        attention_mask = [b[constants.GENERAL.ATTENTION_MASK] for b in batch]
        labels = [b[constants.GENERAL.LABELS] for b in batch]
        input_ids_tensor = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True,
                                                           padding_value=self.tokenizer.pad_token_id)
        attention_mask_tensor = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels_tensor = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return {
            constants.GENERAL.INPUT_IDS: input_ids_tensor,
            constants.GENERAL.ATTENTION_MASK: attention_mask_tensor,
            constants.GENERAL.LABELS: labels_tensor
        }
