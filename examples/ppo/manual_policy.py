from llm_recovery.envs.IntrusionEnv import IntrusionEnv
import numpy as np

if __name__ == '__main__':
    print(f"Loading environment...")
    env = IntrusionEnv()
    print(f"Environment loaded.")
    episodes = 1
    costs = []
    for i in range(episodes):
        env.reset()
        done = False
        t = 0
        C = 0
        while not done:
            a = 0
            if env.state[6] == 1:
                a = env.host_and_recovery_id_to_action_id[(6, 43)]
            if env.state[10] == 1:
                a = env.host_and_recovery_id_to_action_id[(10, 9)]
            if env.state[12] == 1:
                a = env.host_and_recovery_id_to_action_id[(12, 9)]
            if env.state[7] == 1:
                a = env.host_and_recovery_id_to_action_id[(7, 43)]
            host_id, recovery_id = env.action_id_to_host_and_recovery_id[a]
            o_prime, cost, done, done, info = env.step(a, llm=True)
            print(f"t: {t}, cost: {cost}, o: {o_prime}, state: {sum(env.state)}, action: {env.recovery_actions[recovery_id]}, host: {env.hosts[host_id]}, attack state: {env.attack_state}")
            print(env.state)
            t+=1
            C+=cost
        costs.append(C)
        print(f"ep {i+1}/{episodes}, avg cost: {np.mean(costs)}")
