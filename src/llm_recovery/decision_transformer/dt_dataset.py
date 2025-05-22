from typing import List, Dict, Any, Union
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import llm_recovery.constants.constants as constants
import torch


class DTDataset(Dataset[Dict[str, torch.Tensor]]):
    """
    A torch dataset of synthetic data
    """

    def __init__(self, samples: List["str"], tokenizer: PreTrainedTokenizer):
        """
        Initializes the dataset with a given list of samples.

        :param samples: the list of data samples
        :param tokenizer: the LLM tokenizer
        """
        self.samples = samples
        self.tokenizer = tokenizer

    def __len__(self):
        """
        :return: The length of the dataset
        """
        return len(self.samples)

    def __getitem__(self, i):
        """
        Retrieves the ith data sample in tokenized form

        :param i: the index of the data sample
        :return: a dictionary with input ids (the token ids and an attention mask)
        """
        sample = self.samples[i]
        # Get token ids as PyTorch tensors (pt) with the attention masks.
        tokenized_input_sample = self.tokenizer(sample, return_tensors=constants.GENERAL.PYTORCH)
        return {constants.GENERAL.INPUT_IDS: tokenized_input_sample.input_ids[0],
                constants.GENERAL.ATTENTION_MASK: tokenized_input_sample.attention_mask[0]}

    def collate(self, batch: List[Any]) -> Dict[str, Union[List[Any], torch.Tensor]]:
        """
        Takes a batch of tokenized samples, pads them so they have the same length, and  returns a dictionary of
        input_ids (tokenized ids), attention_mask (tokenized mask), and labels, which can be used for supervised
        fine-tuning.

        :param batch: the batch of tokenized samples
        :return: the dictionary with input ids, attention mask, and labels.
        """
        ids = [b[constants.GENERAL.INPUT_IDS] for b in batch]
        mask = [b[constants.GENERAL.ATTENTION_MASK] for b in batch]
        ids_tensor = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True,
                                                     padding_value=self.tokenizer.pad_token_id)
        mask_tensor = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True, padding_value=0)
        labels = ids_tensor.clone()
        return {constants.GENERAL.INPUT_IDS: ids_tensor, constants.GENERAL.ATTENTION_MASK: mask_tensor,
                constants.GENERAL.LABELS: labels}
