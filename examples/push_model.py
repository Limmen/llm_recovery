import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils.quantization_config import BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import llm_recovery.constants.constants as constants
from huggingface_hub import upload_folder

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(constants.LLM.DEEPSEEK_14B_QWEN, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type=constants.GPU.NF4,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    llm = AutoModelForCausalLM.from_pretrained(constants.LLM.DEEPSEEK_14B_QWEN,
                                               device_map={"": 0},
                                               quantization_config=quantization_config,
                                               attn_implementation=constants.GPU.SDPA,
                                               torch_dtype=torch.bfloat16)
    llm.use_memory_efficient_attention = True
    output_dir = "/home/kim/recovery_backups/backup_17_june/checkpoint-1704"
    peft_config = PeftConfig.from_pretrained(output_dir)
    assert isinstance(llm, torch.nn.Module)
    model = PeftModel.from_pretrained(llm, output_dir)
    model.save_pretrained("/home/kim/recovery_backups/pretrained/")
    tokenizer.save_pretrained("/home/kim/recovery_backups/pretrained/")
    upload_folder(folder_path="/home/kim/recovery_backups/pretrained/",
                  repo_id="kimhammar/LLMIncidentResponse", repo_type="model")
