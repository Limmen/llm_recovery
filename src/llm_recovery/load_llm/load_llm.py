from typing import Dict, Union, Tuple
from transformers import (AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel,
                          BitsAndBytesConfig, AutoConfig)
import torch
import llm_recovery.constants.constants as constants


class LoadLLM:
    """
    Class with utility functions to load LLM models.
    """

    @staticmethod
    def load_llm(llm_name: str, device_map: Union[Dict[str, int], str] = "auto", num_gpus: int = 1,
                 use_quantization: bool = True) \
            -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
        """
        Utility function for loading a pretrained LLM from huggingface.

        :param llm_name: the name of the pretrained LLM.
        :param device_map: the device map for loading the LLM, i.e., which GPUs to load it on.
        :param use_quantization: boolean flag indicating whether to use quantization.
        :return: The tokenizer and the LLM
        """
        tokenizer = AutoTokenizer.from_pretrained(llm_name, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token
        if device_map == constants.GPU.DISTRIBUTED:
            device_map = LoadLLM.create_device_map(num_gpus=num_gpus, llm_name=llm_name)
        if use_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type=constants.GPU.NF4,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            llm = AutoModelForCausalLM.from_pretrained(llm_name, device_map=device_map,
                                                       quantization_config=quantization_config,
                                                       attn_implementation=constants.GPU.SDPA,
                                                       torch_dtype=torch.bfloat16)
            llm.use_memory_efficient_attention = True
        else:
            llm = AutoModelForCausalLM.from_pretrained(llm_name, device_map=device_map, torch_dtype=torch.float16)
        return tokenizer, llm

    @staticmethod
    def create_device_map(num_gpus: int, llm_name: str) -> Dict[str, int]:
        """
        Utiltiy function for automatically creating a device map for distributed training.

        :param num_gpus: the number of GPUs
        :param llm_name: the name of the LLM
        :return: the device map
        """
        config = AutoConfig.from_pretrained(llm_name, trust_remote_code=True)
        num_layers = config.num_hidden_layers
        base = num_layers // num_gpus
        remainder = num_layers % num_gpus

        layers_each = [base] * num_gpus
        for i in range(remainder):
            layers_each[i] += 1

        if layers_each[0] > 0:
            layers_each[0] -= 1
            tgt_gpu = layers_each.index(min(layers_each))
            layers_each[tgt_gpu] += 1

        assert sum(layers_each) == num_layers

        device_map = {"model.embed_tokens": 0, "model.norm": 0, "lm_head": 0}
        layer_idx = 0
        for gpu, n in enumerate(layers_each):
            for _ in range(n):
                device_map[f"model.layers.{layer_idx}"] = gpu
                layer_idx += 1
        return device_map
