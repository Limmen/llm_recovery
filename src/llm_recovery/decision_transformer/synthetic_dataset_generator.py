from typing import List
import random
from transformers import PreTrainedTokenizer
from llm_recovery.decision_transformer.dt_dataset import DTDataset
import llm_recovery.constants.constants as constants


class SyntheticDatasetGenerator:
    """
    Class with utility functions for generating synthetic datasets for training a Decision Transformer
    """

    @staticmethod
    def synthetic_episode(episode_idx: int, time_horizon: int, actions: List[str]) -> str:
        """
        Generates a synthetic episode for training a decisio Transformer

        :param episode_idx: the index of the episode
        :param time_horizon: the time horizon of the episode
        :param actions: the action space
        :return: the synthetic episode encoded as a long string
        """
        states = [f"[{episode_idx}:{t}] IDS alert X{t}" for t in range(time_horizon)]
        rewards = [1 if random.random() > .3 else 0 for _ in range(time_horizon)]
        rtg = list(reversed([sum(rewards[t:]) for t in range(time_horizon)]))
        actions = [random.choice(actions) for _ in range(time_horizon)]
        seq = []
        for s, a, r in zip(states, actions, rtg):
            seq.append(f"{constants.DECISION_TRANSFORMER.OBSERVATION_OPEN_DELIMITER}{s}"
                       f"{constants.DECISION_TRANSFORMER.ACTION_OPEN_DELIMITER}{a}"
                       f"{constants.DECISION_TRANSFORMER.RTG_OPEN_DELIMITER}{r}")
        seq.append(constants.DECISION_TRANSFORMER.SEQUENCE_END)
        return " ".join(seq)

    @staticmethod
    def generate_synthetic_dataset(tokenizer: PreTrainedTokenizer, actions: List[str], num_episodes: int = 10,
                                   time_horizon: int = 100) -> DTDataset:
        """
        Generates a synthetic dataset for training a Decision Transformer

        :param tokenizer: the tokenizer used for tokenization of the dataset
        :param num_episodes: the number of episodes in the dataset
        :param time_horizon: the time horizon of an episode
        :param actions: the list of actions
        :return: the dataset as a TorchDataset object
        """
        samples = [SyntheticDatasetGenerator.synthetic_episode(i, time_horizon=time_horizon, actions=actions)
                   for i in range(num_episodes)]
        dataset = DTDataset(samples=samples, tokenizer=tokenizer)
        return dataset
