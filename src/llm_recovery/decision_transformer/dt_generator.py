from transformers import PreTrainedModel, PreTrainedTokenizerFast


class DTGenerator:
    """
    Class with utility functions related to generating outputs with a fine-tuned decision transformer.
    """

    @staticmethod
    def generate(prompt: str, llm: PreTrainedModel, tokenizer: PreTrainedTokenizerFast) -> str:
        """
        Uses an LLM fine-tuned witth decision transformer to generate outputs based on a given prompt.

        :param prompt: the prompt
        :param llm: the fine-tuned LLM
        :param tokenizer: the tokenizer
        :return: the output of the fine-tuned LLM
        """
        gen = tokenizer(prompt, return_tensors="pt").to(llm.device)
        out = llm.generate(**gen, max_new_tokens=20, eos_token_id=tokenizer.eos_token_id)
        return str(tokenizer.decode(out[0], skip_special_tokens=True))
