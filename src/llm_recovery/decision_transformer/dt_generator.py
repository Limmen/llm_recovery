from transformers import PreTrainedModel, PreTrainedTokenizer
import llm_recovery.constants.constants as constants


class DTGenerator:
    """
    Class with utility functions related to generating outputs with a fine-tuned decision transformer.
    """

    @staticmethod
    def generate(prompt: str, llm: PreTrainedModel, tokenizer: PreTrainedTokenizer,
                 max_new_tokens: int = 20) -> str:
        """
        Uses an LLM fine-tuned with decision transformer to generate outputs based on a given prompt.

        :param prompt: the prompt
        :param llm: the fine-tuned LLM
        :param tokenizer: the tokenizer
        :param max_new_tokens: the maximum number of new tokens to generate
        :return: the output of the fine-tuned LLM
        """
        gen = tokenizer(prompt, return_tensors=constants.GENERAL.PYTORCH).to(llm.device)
        out = llm.generate(**gen, max_new_tokens=max_new_tokens, eos_token_id=tokenizer.eos_token_id,
                           pad_token_id=tokenizer.eos_token_id)
        return str(tokenizer.decode(out[0], skip_special_tokens=True))
