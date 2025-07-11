from typing import List, Dict
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import torch
import llm_recovery.constants.constants as constants


class PostThinkDataset(Dataset[Dict[str, torch.Tensor]]):
    """
    A dataset where the prompt ends with <think>, and the model generates reasoning and an answer.
    Loss is computed only on the part of the answer after </think>.
    """

    def __init__(self, instructions: List[str], answers: List[str], tokenizer: PreTrainedTokenizer):
        """
        Initializes the dataset

        :param instructions: the intrusions for instruction-fine-tuning
        :param answers: the answers (labels) for training
        :param tokenizer: the tokenize
        """
        self.instructions = instructions
        self.answers = answers
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """
        :return: the number of instructions
        """
        return len(self.instructions)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Gets on item from the dataset with a specific index

        :param idx: the index of the item to get
        :return: the item with the specified index
        """
        prompt = self.instructions[idx]
        full_answer = self.answers[idx]

        # Tokenize prompt (includes <think>)
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
        prompt_input_ids = prompt_tokens[constants.GENERAL.INPUT_IDS]
        prompt_attention_mask = prompt_tokens[constants.GENERAL.ATTENTION_MASK]

        # Tokenize full answer (may include reasoning and </think>)
        answer_tokens = self.tokenizer(full_answer, add_special_tokens=False)
        answer_input_ids = answer_tokens[constants.GENERAL.INPUT_IDS]
        answer_attention_mask = answer_tokens[constants.GENERAL.ATTENTION_MASK]

        # Find index of </think> in tokenized answer
        end_think_ids = self.tokenizer("</think>", add_special_tokens=False)[constants.GENERAL.INPUT_IDS]
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
        :return: processes batch
        """
        input_ids = [b[constants.GENERAL.INPUT_IDS] for b in batch]
        attention_mask = [b[constants.GENERAL.ATTENTION_MASK] for b in batch]
        labels = [b[constants.GENERAL.LABELS] for b in batch]

        input_ids_tensor = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask_tensor = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels_tensor = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        return {
            constants.GENERAL.INPUT_IDS: input_ids_tensor,
            constants.GENERAL.ATTENTION_MASK: attention_mask_tensor,
            constants.GENERAL.LABELS: labels_tensor,
        }
