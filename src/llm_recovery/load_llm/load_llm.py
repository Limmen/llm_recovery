from typing import Dict, Union, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerFast, PreTrainedModel
import torch

class LoadLLM:
    """
    Class with utility functions to load LLM models.
    """

    @staticmethod
    def load_llm(llm_name: str, device_map: Union[Dict[str, int], str] = "auto") \
            -> Tuple[PreTrainedTokenizerFast, PreTrainedModel]:
        """
        Utility function for loading a pretrained LLM from huggingface.

        :param llm_name: the name of the pretrained LLM.
        :param device_map: the device map for loading the LLM, i.e., which GPUs to load it on.
        :return: The tokenizer and the LLM
        """
        tokenizer = AutoTokenizer.from_pretrained(llm_name, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token  # avoid pad-token warnings
        llm = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype=torch.float16, device_map=device_map)
        return tokenizer, llm
