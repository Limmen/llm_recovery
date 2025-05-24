import random
import json
import llm_recovery.constants.constants as constants
from llm_recovery.envs.IntrusionEnv import IntrusionEnv

if __name__ == '__main__':
    print("Loading environment...")
    env = IntrusionEnv()
    print("Environment loaded.")
    num_episodes = 100000
    num_optimal_episodes = 20000
    episodes = []
    for episode in range(num_episodes):
        print(f"Generating episode {episode}/{num_episodes}")
        observations = []
        actions = []
        costs = []
        env.reset()
        done = False
        while not done:
            if episode < num_optimal_episodes:
                action = 0
                if env.state[6] == 1:
                    action = env.host_and_recovery_id_to_action_id[(6, 43)]
                if env.state[10] == 1:
                    action = env.host_and_recovery_id_to_action_id[(10, 9)]
                if env.state[12] == 1:
                    action = env.host_and_recovery_id_to_action_id[(12, 9)]
                if env.state[7] == 1:
                    action = env.host_and_recovery_id_to_action_id[(7, 43)]
            else:
                action = random.randint(0, len(env.actions) - 1)
            o_prime, cost, done, done, info = env.step(action, llm=True)
            host_id, recovery_action_id = env.action_id_to_host_and_recovery_id[action]
            host_str = env.hosts[host_id]
            action_str = env.recovery_actions[recovery_action_id]
            actions.append(action_str + f",{host_str.split(',')[0]}")
            costs.append(cost)
            observations.append(o_prime)
        costs_to_go = list([sum(costs[t:]) for t in range(len(costs))])
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
        print(f"Sequence length: {len(seq)}")
        episodes.append("".join(seq))
        print(f"seq length: {len(''.join(seq))}")
    with open("attack_sequences.json", "w") as f:
        json.dump(episodes, f)
