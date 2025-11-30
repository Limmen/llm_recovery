from typing import Dict, Any, Deque, List, Union, Optional
import random
import torch
from collections import deque
from transformers import (TrainerCallback, TrainerControl, TrainerState, TrainingArguments, PreTrainedModel,
                          PreTrainedTokenizer)
import llm_recovery.constants.constants as constants
from llm_recovery.decision_transformer.dt_dataset import DTDataset
from llm_recovery.fine_tuning.examples_dataset import ExamplesDataset
from llm_recovery.fine_tuning.post_think_dataset import PostThinkDataset
import time
import json


class LoggingCallback(TrainerCallback):
    """
    Callback for logging during LORA  training.
    """

    def __init__(self, prompts: List[str], answers: List[str], tokenizer: PreTrainedTokenizer,
                 dataset: Union[DTDataset, ExamplesDataset, PostThinkDataset],
                 window: int = 100, gen_kwargs: Optional[Dict[str, Any]] = None, prompt_logging: bool = False,
                 prompt_logging_frequency: int = 1, progress_save_frequency: int = 1, seed: int = 29015) -> None:
        """
        Initializes the callback.

        :param prompts: List of prompts to use for testing during training
        :param answers: List of answers to use for testing during training
        :param tokenizer: the tokenizer for the LLM
        :param window: the length of the training window for computing running averages
        :param gen_kwargs: keyword arguments to use for test generation with the LLM
        :param dataset: dataset for training
        :param prompt_logging: Boolean flag indicating whether to log test-prompts during training
        :param prompt_logging_frequency: frequency of prompt logging
        :param progress_save_frequency: frequency of saving the training progress to disk
        :param seed: the random seed
        """
        self.prompts = prompts
        self.answers = answers
        self.tokenizer = tokenizer
        self.window = window
        self.dataset = dataset
        self.prompt_logging = prompt_logging
        self.losses: Deque[float] = deque(maxlen=window)
        self.gen_kwargs = gen_kwargs or {constants.GENERAL.MAX_NEW_TOKENS: 64}
        self.prompt_logging_frequency = prompt_logging_frequency
        self.avg_losses_logging: List[float] = []
        self.losses_logging: List[float] = []
        self.grad_norms: List[float] = []
        self.learning_rates: List[float] = []
        self.epochs: List[int] = []
        self.start_time: float = time.time()
        self.times_passed: List[float] = []
        self.steps: List[int] = []
        self.progress_save_frequency = progress_save_frequency
        self.seed = seed

    @torch.no_grad()
    def _sample(self, llm: PreTrainedModel, prompt: str) -> str:
        """
        Utility function to sample from the LLM.

        :param llm: the LLM to sample from
        :param prompt: the prompt
        :return: the sampled output
        """
        model_was_training = llm.training
        llm.eval()
        inputs = self.tokenizer(prompt, return_tensors=constants.GENERAL.PYTORCH).to(llm.device)
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
            if self.prompt_logging and state.global_step % self.prompt_logging_frequency == 0:
                model = kwargs[constants.GENERAL.MODEL]
                prompt_idx = random.randint(0, len(self.prompts) - 1)
                prompt = self.prompts[prompt_idx]
                answer = self.answers[prompt_idx]
                model_output = self._sample(model, prompt=prompt)
                model_output = model_output[len(prompt):]
                print(f"prediction:\n{model_output}\nlabel:\n{answer}\n", flush=True)
            minutes_passed = (time.time() - self.start_time) / 60
            progress = state.global_step / state.max_steps * 100
            print(f"Step: {state.global_step}, Epoch: {state.epoch:.4f}, Progress: {round(progress, 2)}%, "
                  f"Avg_loss={rolling_loss:.4f}, "
                  f"LR={lr:.8f}, Grad_norm={gn:.4f}, minutes: {minutes_passed:.4f}", flush=True)
            self.avg_losses_logging.append(rolling_loss)
            self.losses_logging.append(float(loss))
            self.grad_norms.append(gn)
            self.learning_rates.append(lr)
            self.times_passed.append(minutes_passed)
            self.steps.append(state.global_step)
            if state.global_step % self.progress_save_frequency == 0:
                with open("training.json", "w") as f:
                    training_state = {
                        "steps": self.steps,
                        "running_avg": self.window,
                        "seed": self.seed,
                        "avg_losses": self.avg_losses_logging,
                        "losses": self.losses_logging,
                        "grad_norms": self.grad_norms,
                        "learning_rates": self.learning_rates,
                        "times_passed": self.times_passed
                    }
                    json.dump(training_state, f)
        except Exception:
            pass
