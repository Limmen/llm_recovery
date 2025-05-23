from typing import Dict, Union, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel, BitsAndBytesConfig
import torch


class LoadLLM:
    """
    Class with utility functions to load LLM models.
    """

    @staticmethod
    def load_llm(llm_name: str, device_map: Union[Dict[str, str], str] = "auto") \
            -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
        """
        Utility function for loading a pretrained LLM from huggingface.

        :param llm_name: the name of the pretrained LLM.
        :param device_map: the device map for loading the LLM, i.e., which GPUs to load it on.
        :return: The tokenizer and the LLM
        """
        tokenizer = AutoTokenizer.from_pretrained(llm_name, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        llm = AutoModelForCausalLM.from_pretrained(llm_name, device_map=device_map,
                                                   quantization_config=quantization_config)
        llm.use_memory_efficient_attention = True
        return tokenizer, llm
