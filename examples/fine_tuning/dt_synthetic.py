from llm_recovery.load_llm.load_llm import LoadLLM
from llm_recovery.fine_tuning.lora import LORA
from llm_recovery.decision_transformer.synthetic_dataset_generator import SyntheticDatasetGenerator
from llm_recovery.decision_transformer.dt_generator import DTGenerator
import llm_recovery.constants.constants as constants

if __name__ == '__main__':
    actions = ["isolate host", "rotate keys", "block IP",
               "reimage server", "collect memory dump", "notify IR lead"]
    num_episodes = 10
    time_horizon = 100
    tokenizer, llm = LoadLLM.load_llm(llm_name=constants.LLM.DEEPSEEK_7B, device_map={"": 0})
    llm = LORA.setup_llm_for_fine_tuning(llm=llm)
    dataset = SyntheticDatasetGenerator.generate_synthetic_dataset(tokenizer=tokenizer, num_episodes=num_episodes,
                                                                   actions=actions, time_horizon=time_horizon)
    LORA.supervised_fine_tuning(llm=llm, dataset=dataset)
    prompt = """
    <state> [INC42:5] outbound C2 to 185.220.101.1
    <action> isolate host
    <rtg> 3
    """
    output = DTGenerator.generate(prompt=prompt, llm=llm, tokenizer=tokenizer)
    print(output)
