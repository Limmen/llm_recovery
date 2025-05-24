import numpy as np
import time
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from llm_recovery.envs.IntrusionEnv import IntrusionEnv


class DiscountedRewardLoggerCallback(BaseCallback):
    """
    Logs the average DISCOUNTED episode returns every `log_freq` episodes,
    and also prints the elapsed time in seconds.
    """

    def __init__(self, gamma=0.99, log_freq=10, verbose=0):
        super().__init__(verbose)
        self.gamma = gamma
        self.log_freq = log_freq
        self.episode_discounted_returns = []
        self.episode_count = 0
        self.current_discounted_return = 0.0
        self.episode_step = 0
        self.start_time = time.time()

    def _on_step(self) -> bool:
        done = self.locals["dones"][0]
        reward = self.locals["rewards"][0]
        self.current_discounted_return += (self.gamma ** self.episode_step) * reward
        self.episode_step += 1
        if done:
            self.episode_count += 1
            self.episode_discounted_returns.append(self.current_discounted_return)
            if (self.episode_count % self.log_freq) == 0:
                avg_10 = np.mean(self.episode_discounted_returns[-100:])
                elapsed = time.time() - self.start_time
                print(f"Episode={self.episode_count}, "
                      f"DiscountedReturn={self.current_discounted_return:.2f}, "
                      f"AvgLast100={avg_10:.2f}, "
                      f"Elapsed={elapsed:.2f}s")
            self.current_discounted_return = 0.0
            self.episode_step = 0
        return True


def main():
    print(f"Loading environment...")
    env = IntrusionEnv(costs=False)
    print(f"Environment loaded.")
    env = Monitor(env)
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=0,
        policy_kwargs={
            "net_arch": [64, 64]
        },
        learning_rate=1e-4,
        n_steps=5012,
        batch_size=128,
        gamma=0.95,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device='cpu'
    )
    callback = DiscountedRewardLoggerCallback(log_freq=100, gamma=1)
    model.learn(total_timesteps=5000000, callback=callback)


if __name__ == "__main__":
    main()
