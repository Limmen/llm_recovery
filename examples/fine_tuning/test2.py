import json
from llm_recovery.load_llm.load_llm import LoadLLM
from llm_recovery.fine_tuning.lora import LORA
from llm_recovery.decision_transformer.dt_generator import DTGenerator
import llm_recovery.constants.constants as constants
from llm_recovery.decision_transformer.dt_dataset import DTDataset

if __name__ == '__main__':
    with open("attack_sequences.json", "r") as f:
        loaded_sequences = json.load(f)
    print(loaded_sequences[0])
    tokenizer, llm = LoadLLM.load_llm(llm_name=constants.LLM.DEEPSEEK_7B, device_map={"": 0})
    llm = LORA.setup_llm_for_fine_tuning(llm=llm)
    dataset = DTDataset(samples=loaded_sequences, tokenizer=tokenizer)
    prompt = ("<observation>940</observation><action>Reconfigure firewall to block port 80 on target "
              "host IPs=15.9.7.17 "
              "OS=Ubuntu 20 Services/Ports=SSH:22/SNMP:161/SMTP:25/Postgres:5432/NTP:123</action><rtg>-403.0</rtg> "
              "<observation>2231</observation>")
    lr = 5e-5
    # lr = 5e-4
    per_device_batch_size = 2
    num_train_epochs = 4
    prompt_logging_frequency = 20
    max_generation_tokens = 500
    logging_steps = 1
    running_average_window = 100
    LORA.supervised_fine_tuning(llm=llm, dataset=dataset, learning_rate=lr,
                                per_device_train_batch_size=per_device_batch_size,
                                num_train_epochs=num_train_epochs, logging_steps=logging_steps, prompt=prompt,
                                prompt_logging=True,
                                running_average_window=running_average_window,
                                max_generation_tokens=max_generation_tokens,
                                prompt_logging_frequency=prompt_logging_frequency)
    output = DTGenerator.generate(prompt=prompt, llm=llm, tokenizer=tokenizer)
    print(output)
