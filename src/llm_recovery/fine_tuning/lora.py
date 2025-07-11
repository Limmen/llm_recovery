from typing import Union, List
from transformers import PreTrainedModel, Trainer, TrainingArguments, PrinterCallback, ProgressCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from llm_recovery.decision_transformer.dt_dataset import DTDataset
from llm_recovery.fine_tuning.examples_dataset import ExamplesDataset
from llm_recovery.fine_tuning.post_think_dataset import PostThinkDataset
import llm_recovery.constants.constants as constants
from llm_recovery.fine_tuning.logging_callback import LoggingCallback
from peft import TaskType


class LORA:
    """
    Class with utility functions for fine-tuning with LORA
    """

    @staticmethod
    def setup_llm_for_fine_tuning(llm: PreTrainedModel, r: int = 8, lora_alpha: int = 32,
                                  lora_dropout: float = 0.05, use_quantization: bool = True):
        """
        Sets up a given LLM for fine-tuning with LORA

        :param llm: the LLM to fine-tune
        :param r: The LORA dimension, i.e.,  the rank
        :param lora_alpha: The alpha parameter for Lora scaling.
        :param lora_dropout: The dropout probability for Lora layers.
        :return: the LLM prepared for fine-tuning
        """
        if use_quantization:
            llm = prepare_model_for_kbit_training(llm)
        lora_cfg = LoraConfig(r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                              bias="none", task_type=TaskType.CAUSAL_LM)
        llm_for_fine_tuning = get_peft_model(llm, lora_cfg)
        if use_quantization:
            llm.gradient_checkpointing_enable()
        trainable_params = [p for p in llm_for_fine_tuning.parameters() if p.requires_grad]
        total_trainable = sum(p.numel() for p in trainable_params)
        print(f"Trainable parameters: {total_trainable}")
        return llm_for_fine_tuning

    @staticmethod
    def supervised_fine_tuning(llm: PreTrainedModel, dataset: Union[DTDataset, ExamplesDataset, PostThinkDataset],
                               output_dir: str = "ds-dt-lora",
                               learning_rate: float = 5e-5, logging_steps: int = 1,
                               per_device_train_batch_size: int = 1, num_train_epochs: int = 3,
                               prompt_logging: bool = False, prompts: List[str] = None,
                               answers: List[str] = None,
                               max_generation_tokens: int = 32,
                               running_average_window: int = 100, prompt_logging_frequency: int = 1,
                               temperature: float = 0.7, save_steps: int = 100, save_limit: int = 2,
                               gradient_accumulation_steps: int = 1, progress_save_frequency: int = 1,
                               seed: int = 91501) -> None:
        """
        Performs supervised fine-tuning of a given llm based on a given dataset

        :param llm: The LLM to fine-tune
        :param dataset: the dataset to use for the fine-tuning
        :param output_dir: The output directory to save the trained weights
        :param learning_rate: The learning rate to use for the fine-tuning
        :param logging_steps: The number of steps to logging the fine-tuning
        :param per_device_train_batch_size: The number of samples to use per device for fine-tuning
        :param prompt_logging: Boolean flag indicating whether to log test-prompts during training
        :param prompts: The prompts to use for prompt logging
        :param answers: The prompts to use for prompt logging
        :param max_generation_tokens: The maximum number of tokens to generate for prompt logging
        :param running_average_window: length of the window to compute running averages
        :param prompt_logging_frequency: frequency of prompt logging
        :param temperature: controls the randomness of the LLM outputs
        :param save_steps: controls how frequently to checkpoint the model
        :param save_limit: controls the maximum number of saved copies of the model
        :param gradient_accumulation_steps: how many gradients to accumulate before updating model
        :param use_quantization: boolean flag indicating whether quantization should be used
        :param progress_save_frequency: frequency of saving training progress to disk
        :param seed: the random seed for reproducibility
        :return: None
        """
        gen_kwargs = dict(max_new_tokens=max_generation_tokens, temperature=temperature, do_sample=True)
        args = TrainingArguments(
            output_dir=output_dir, bf16=True,
            per_device_train_batch_size=per_device_train_batch_size, num_train_epochs=num_train_epochs,
            learning_rate=learning_rate, logging_steps=logging_steps,
            save_strategy=constants.LORA.SAVE_STRATEGY_STEPS, save_steps=save_steps,
            save_total_limit=save_limit, gradient_accumulation_steps=gradient_accumulation_steps, seed=seed)
        callback = LoggingCallback(prompts=prompts, tokenizer=dataset.tokenizer, window=running_average_window,
                                   gen_kwargs=gen_kwargs, dataset=dataset,
                                   prompt_logging=prompt_logging, prompt_logging_frequency=prompt_logging_frequency,
                                   answers=answers, progress_save_frequency=progress_save_frequency, seed=seed)
        trainer = Trainer(model=llm, args=args, train_dataset=dataset, data_collator=dataset.collate,
                          callbacks=[callback])
        trainer.remove_callback(PrinterCallback)
        trainer.remove_callback(ProgressCallback)
        trainer.train()
        trainer.save_model(output_dir=output_dir)
