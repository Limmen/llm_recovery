import random
import json
import llm_recovery.constants.constants as constants
from llm_recovery.envs.IntrusionEnv import IntrusionEnv

if __name__ == '__main__':
    env = IntrusionEnv()
    env.reset()
    observations = []
    actions = []
    costs = []
    num_episodes = 1
    episodes = []
    for episode in range(num_episodes):
        done = False
        while not done:
            action = random.randint(0, len(env.actions) - 1)
            o_prime, cost, done, done, info = env.step(action, llm=True)
            host_id, recovery_action_id = env.action_id_to_host_and_recovery_id[action]
            host_str = env.hosts[host_id]
            action_str = env.recovery_actions[recovery_action_id]
            actions.append(action_str + f", host={host_str.split(',')[0]}")
            costs.append(cost)
            observations.append(o_prime)
        costs_to_go = list(reversed([sum(costs[t:]) for t in range(len(costs))]))
        seq = [constants.DECISION_TRANSFORMER.TASK_DESCRIPTION_OPEN_DELIMITER,
               constants.DECISION_TRANSFORMER.TASK_INSTRUCTION,
               constants.DECISION_TRANSFORMER.SYSTEM_INSTRUCTION_OPEN_DELIMITER,
               constants.DECISION_TRANSFORMER.SYSTEM_INSTRUCTION,
               ",".join(env.hosts),
               constants.DECISION_TRANSFORMER.ACTION_SPACE_INSTRUCTION_OPEN_DELIMITER,
               constants.DECISION_TRANSFORMER.ACTION_SPACE_INSTRUCTION,
               ",".join(env.recovery_actions_and_costs_strings),
               constants.DECISION_TRANSFORMER.SEQUENCE_DESCRIPTION_OPEN_DELIMITER,
               constants.DECISION_TRANSFORMER.SEQUENCE_INSTRUCTION,
               constants.DECISION_TRANSFORMER.SEQUENCE_START
               ]
        for o, a, c in zip(observations, actions, costs_to_go):
            seq.append(f"{constants.DECISION_TRANSFORMER.OBSERVATION_OPEN_DELIMITER}{o}"
                       f"{constants.DECISION_TRANSFORMER.ACTION_OPEN_DELIMITER}{a}"
                       f"{constants.DECISION_TRANSFORMER.COST_TO_GO_OPEN_DELIMITER}{c}")
        seq.append(constants.DECISION_TRANSFORMER.SEQUENCE_END)
        episodes.append("".join(seq))
    with open("attack_sequences.json", "w") as f:
        json.dump(episodes, f)
