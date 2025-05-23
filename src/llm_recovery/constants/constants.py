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
    COST_TO_GO_OPEN_DELIMITER = "<cost-to-go>"
    COST_TO_GO_CLOSE_DELIMITER = "</cost-to-go>"
    TASK_DESCRIPTION_OPEN_DELIMITER = "<task>"
    TASK_DESCRIPTION_CLOSE_DELIMITER = "</task>"
    SEQUENCE_DESCRIPTION_OPEN_DELIMITER = "<sequence-description>"
    SEQUENCE_DESCRIPTION_CLOSE_DELIMITER = "</sequence-description>"
    ACTION_SPACE_INSTRUCTION_OPEN_DELIMITER = "<action-space>"
    ACTION_SPACE_INSTRUCTION_CLOSE_DELIMITER = "</action-space>"
    SYSTEM_INSTRUCTION_OPEN_DELIMITER = "<system>"
    SYSTEM_INSTRUCTION_CLOSE_DELIMITER = "</system>"
    SEQUENCE_START = "<sequence>"
    SEQUENCE_END = "<sequence/>"
    SEQUENCE_INSTRUCTION = ("The system is a partially observed discrete-time dynamical system. "
                            "The following is a trace of observations, actions, and cost-to-go."
                            " Your task is to continue predicting the sequence.")
    TASK_INSTRUCTION = ("You are a security operator selecting recovery actions.")
    SYSTEM_INSTRUCTION = ("Hosts: ")
    ACTION_SPACE_INSTRUCTION = ("This is the list of per-host recovery actions and associated costs.")
