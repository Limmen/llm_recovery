"""
Constants for llm_recovery
"""


class LLM:
    """
    LLM constants
    """
    DEEPSEEK_7B = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    MISTRAL_7B = "mistralai/Mistral-7B-v0.1"


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


class LORA:
    """
    String constants related to LORA
    """
    SAVE_STRATEGY_NO = "no"


class DECISION_TRANSFORMER:
    """
    String constants related to decision transformer
    """
    STATE_OPEN_DELIMITER = "<state>"
    STATE_CLOSE_DELIMITER = "</state>"
    OBSERVATION_OPEN_DELIMITER = "<observation>"
    OBSERVATION_CLOSE_DELIMITER = "</observation>"
    ACTION_OPEN_DELIMITER = "<action>"
    ACTION_CLOSE_DELIMITER = "</action>"
    RTG_OPEN_DELIMITER = "<rtg>"
    RTG_CLOSE_DELIMITER = "</rtg>"
    SEQUENCE_END = "<end>"
