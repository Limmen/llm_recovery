"""
Constants for llm_recovery
"""


class LLM:
    """
    LLM constants
    """
    DEEPSEEK_32B_QWEN = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    DEEPSEEK_14B_QWEN = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    DEEPSEEK_7B_QWEN = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    DEEPSEEK_1_5B_QWEN = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    DEEPSEEK_8B_LLAMA = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    LLAMA_1B_INSTRUCT = "meta-llama/Llama-3.2-1B-Instruct"
    MISTRAL_7B_INSTRUCT = "mistralai/Mistral-7B-Instruct-v0.3"
    QWEN_14B = "Qwen/Qwen3-14B"


class GENERAL:
    """
    General string constants
    """
    INPUT_IDS = "input_ids"
    ATTENTION_MASK = "attention_mask"
    PYTORCH = "pt"
    LABELS = "labels"
    LEARNING_RATE = "learning_rate"
    GRAD_NORM = "grad_norm"
    LOSS = "loss"
    MODEL = "model"
    MAX_NEW_TOKENS = "max_new_tokens"
    N_A = "n/a"


class GPU:
    """String constants related to GPUs"""
    DISTRIBUTED = "distributed"
    SDPA = "sdpa"
    NF4 = "nf4"
    AUTO = "auto"
    MODEL_EMBED_TOKENS = "model.embed_tokens"
    MODEL_NORM = "model.norm"
    LM_HEAD = "lm_head"
    MODEL_LAYERS = "model.layers"


class LORA:
    """
    String constants related to LORA
    """
    SAVE_STRATEGY_NO = "no"
    SAVE_STRATEGY_STEPS = "steps"


class DECISION_TRANSFORMER:
    """
    String constants related to decision transformer
    """
    STATE_OPEN_DELIMITER = "<s>"
    STATE_CLOSE_DELIMITER = "</s>"
    OBSERVATION_OPEN_DELIMITER = "<obs>"
    OBSERVATION_CLOSE_DELIMITER = "</o>"
    ACTION_OPEN_DELIMITER = "<action>"
    ACTION_CLOSE_DELIMITER = "</a>"
    COST_TO_GO_OPEN_DELIMITER = "<cost>"
    COST_TO_GO_CLOSE_DELIMITER = "</cost>"
    TASK_DESCRIPTION_OPEN_DELIMITER = "<task>"
    TASK_DESCRIPTION_CLOSE_DELIMITER = "</task>"
    SEQUENCE_DESCRIPTION_OPEN_DELIMITER = " "
    ACTION_SPACE_INSTRUCTION_OPEN_DELIMITER = "<actions>"
    ACTION_SPACE_INSTRUCTION_CLOSE_DELIMITER = "</actions>"
    SYSTEM_INSTRUCTION_OPEN_DELIMITER = "<system>"
    SYSTEM_INSTRUCTION_CLOSE_DELIMITER = "</system>"
    SEQUENCE_START = "<history>"
    SEQUENCE_END = "</history>"
    SEQUENCE_INSTRUCTION = (" The system can be modeled as a POMDP. The following is a POMDP history. Continue it")
    TASK_INSTRUCTION = (" You are a security operator selecting recovery actions for a system.")
    SYSTEM_INSTRUCTION = (" These are the system's hosts:")
    ACTION_SPACE_INSTRUCTION = (" List of per-host recovery actions and their costs:")
