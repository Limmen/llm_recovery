from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
import torch, random

ckpt = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"   # ← same as your script

# ---- 1. tokenizer ----------------------------------------------------------
tok = AutoTokenizer.from_pretrained(ckpt, use_fast=True)
tok.pad_token = tok.eos_token        # avoid pad-token warnings

# ---- 2. (optional) 4-bit quant to save VRAM -------------------------------
bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,           # ← comment out for plain FP16
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16)

base = AutoModelForCausalLM.from_pretrained(
        ckpt,
        torch_dtype=torch.float16,           # FP16 weights
        quantization_config=bnb_cfg,         # remove arg if not quantising
        device_map={"": 0},                  # *** SINGLE GPU ***
        low_cpu_mem_usage=True)

# ---- 3. add LoRA -----------------------------------------------------------
lora_cfg = LoraConfig(
        r=8, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type="CAUSAL_LM")
model = get_peft_model(base, lora_cfg)
model.print_trainable_parameters()

# ░░ 2. tiny text-encoded Decision-Transformer-style dataset ░░
T = 100                                            # horizon
ACTIONS = ["isolate host", "rotate keys", "block IP",
           "reimage server", "collect memory dump", "notify IR lead"]

def toy_episode(idx):
    states  = [f"[{idx}:{t}] IDS alert X{t}" for t in range(T)]
    rewards = [1 if random.random()>.3 else 0 for _ in range(T)]
    rtg     = list(reversed([sum(rewards[t:]) for t in range(T)]))
    actions = [random.choice(ACTIONS) for _ in range(T)]
    seq = []
    for s,a,r in zip(states,actions,rtg):
        seq.append(f"<state> {s} <action> {a} <rtg> {r}")
    seq.append("<end>")
    return " ".join(seq)

class TextDT(Dataset):
    def __init__(self,n): self.samples=[toy_episode(i) for i in range(n)]
    def __len__(self):    return len(self.samples)
    def __getitem__(self,i):
        enc = tok(self.samples[i], return_tensors="pt")
        return {"input_ids": enc.input_ids[0],
                "attention_mask": enc.attention_mask[0]}

train_data = TextDT(3000)

def collate(batch):
    ids  = [b["input_ids"]      for b in batch]
    mask = [b["attention_mask"] for b in batch]
    ids  = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True,
                                           padding_value=tok.pad_token_id)
    mask = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True, padding_value=0)
    labels = ids.clone()
    return {"input_ids":ids, "attention_mask":mask, "labels":labels}

# ░░ 3. LoRA fine-tune ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
args = TrainingArguments(
        output_dir="ds-dt-lora", bf16=True,
        per_device_train_batch_size=1, num_train_epochs=3,
        learning_rate=5e-5, logging_steps=1, save_strategy="no")

trainer = Trainer(model=model, args=args,
                  train_dataset=train_data, data_collator=collate)
trainer.train()

# ░░ 4. inference: ask “what next action?” ░░
prompt = """
<state> [INC42:5] outbound C2 to 185.220.101.1
<action> isolate host
<rtg> 3
"""
gen = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**gen, max_new_tokens=20,
                     eos_token_id=tok.eos_token_id)
print(tok.decode(out[0], skip_special_tokens=True))
