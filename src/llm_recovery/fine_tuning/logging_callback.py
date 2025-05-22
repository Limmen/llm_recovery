from typing import Dict, Any, Deque
import torch
from collections import deque
from transformers import (TrainerCallback, TrainerControl, TrainerState, TrainingArguments, PreTrainedModel,
                          PreTrainedTokenizer)
import llm_recovery.constants.constants as constants
from llm_recovery.decision_transformer.dt_dataset import DTDataset


class LoggingCallback(TrainerCallback):
    """
    Callback for logging during LORA  training.
    """

    def __init__(self, prompt: str, tokenizer: PreTrainedTokenizer, dataset: DTDataset,
                 window: int = 100, gen_kwargs: Dict[str, Any] | None = None, prompt_logging: bool = False,
                 prompt_logging_frequency: int = 1) -> None:
        """
        Initializes the callback.

        :param prompt: the prompt to use for testing during training
        :param tokenizer: the tokenizer for the LLM
        :param window: the length of the training window for computing running averages
        :param gen_kwargs: keyword arguments to use for test generation with the LLM
        :param dataset: dataset for training
        :param prompt_logging: Boolean flag indicating whether to log test-prompts during training
        :param prompt_logging_frequency: frequency of prompt logging
        """
        self.prompt = prompt
        self.tokenizer = tokenizer
        self.window = window
        self.dataset = dataset
        self.prompt_logging = prompt_logging
        self.losses: Deque[float] = deque(maxlen=window)
        self.gen_kwargs = gen_kwargs or {constants.GENERAL.MAX_NEW_TOKENS: 64}
        self.prompt_logging_frequency = prompt_logging_frequency

    @torch.no_grad()
    def _sample(self, llm: PreTrainedModel) -> str:
        """
        Utility function to sample from the LLM.

        :param llm: the LLM to sample from
        :return: the sampled output
        """
        model_was_training = llm.training
        llm.eval()
        inputs = self.tokenizer(self.prompt, return_tensors=constants.GENERAL.PYTORCH).to(llm.device)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        output_ids = llm.generate(**inputs, pad_token_id=pad_id, **self.gen_kwargs)
        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if model_was_training:
            llm.train()
        return str(text)

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs) \
            -> None:
        """
        This function is called by the trainer during the training phase

        :param args: training arguments
        :param state: state of the training process
        :param control: object that can be used for controlling the training (e.g., early stopping)
        :param logs: the logs stored during training
        :param kwargs: keyword arguments for the training
        :return:
        """
        try:
            loss = logs.get(constants.GENERAL.LOSS)
            self.losses.append(float(loss))
            rolling_loss = sum(self.losses) / len(self.losses)
            lr = logs.get(constants.GENERAL.LEARNING_RATE, constants.GENERAL.N_A)
            gn = logs.get(constants.GENERAL.GRAD_NORM, constants.GENERAL.N_A)
            model_output = "-"
            if self.prompt_logging and state.global_step % self.prompt_logging_frequency == 0:
                model = kwargs[constants.GENERAL.MODEL]
                model_output = self._sample(model)
            progress = state.global_step / state.max_steps * 100
            print(f"Step: {state.global_step}, Progress: {round(progress, 2)}%, Avg_loss={rolling_loss:.4f}, "
                  f"LR={lr:.8f}, "
                  f"Grad_norm={gn:.4f}, sample: {model_output}", flush=True)
        except Exception:
            pass
