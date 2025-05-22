from transformers import PreTrainedModel, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from llm_recovery.decision_transformer.dt_dataset import DTDataset


class LORA:
    """
    Class with utility functions for fine-tuning with LORA
    """

    @staticmethod
    def setup_llm_for_fine_tuning(llm: PreTrainedModel, r: int = 8, lora_alpha: int = 32,
                                  lora_dropout: float = 0.05):
        """
        Sets up a given LLM for fine-tuning with LORA

        :param llm: the LLM to fine-tune
        :param r: The LORA dimension, i.e.,  the rank
        :param lora_alpha: The alpha parameter for Lora scaling.
        :param lora_dropout: The dropout probability for Lora layers.
        :return:
        """
        lora_cfg = LoraConfig(r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        llm_for_fine_tuning = get_peft_model(llm, lora_cfg)
        return llm_for_fine_tuning

    @staticmethod
    def supervised_fine_tuning(llm: PreTrainedModel, dataset: DTDataset) -> None:
        """
        Performs supervised fine-tuning of a given llm based on a given dataset

        :param llm: The LLM to fine-tune
        :param dataset: the dataset to use for the fine-tuning
        :return: None
        """
        args = TrainingArguments(
            output_dir="ds-dt-lora", bf16=True,
            per_device_train_batch_size=1, num_train_epochs=3,
            learning_rate=5e-5, logging_steps=1, save_strategy="no")
        trainer = Trainer(model=llm, args=args, train_dataset=dataset, data_collator=dataset.collate)
        trainer.train()
