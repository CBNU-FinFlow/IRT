"""
Custom Stable-Baselines3 callbacks used across FinFlow.
"""

from __future__ import annotations

import os
from typing import Dict, List, Union

import numpy as np
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import sync_envs_normalization


class StrictEvalCallback(EvalCallback):
    """
    EvalCallback variant with stable comparison semantics.

    Differences from the upstream version:
    - Treats identical floating point values as ties (tolerance = epsilon).
    - Emits high precision logs when no improvement is detected.
    """

    def __init__(self, *args, reward_improvement_epsilon: float = 1e-6, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_improvement_epsilon = float(reward_improvement_epsilon)

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as exc:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from exc

            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                assert isinstance(episode_lengths, list)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                extra_logs: Dict[str, Union[List, np.ndarray]] = {}
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    extra_logs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **extra_logs,
                )

            mean_reward = float(np.mean(episode_rewards))
            std_reward = float(np.std(episode_rewards))
            mean_ep_length = float(np.mean(episode_lengths))
            std_ep_length = float(np.std(episode_lengths))
            self.last_mean_reward = float(mean_reward)

            if self.verbose >= 1:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, "
                    f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
                )
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")

            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/mean_ep_length", mean_ep_length)
            self.logger.record("eval/mean_reward_raw", mean_reward, exclude="tensorboard")
            # 평가 표준편차와 최고 성과를 함께 기록한다.
            self.logger.record("eval/std_reward", std_reward)
            self.logger.record("eval/best_mean_reward", self.best_mean_reward)

            if len(self._is_success_buffer) > 0:
                success_rate = float(np.mean(self._is_success_buffer))
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            self.logger.record(
                "time/total_timesteps",
                self.num_timesteps,
                exclude="tensorboard",
            )
            self.logger.dump(self.num_timesteps)

            improved = mean_reward > (self.best_mean_reward + self.reward_improvement_epsilon)
            # 최고 성과 대비 차이를 별도로 기록해 진전 여부를 확인한다.
            delta_from_best = mean_reward - self.best_mean_reward
            self.logger.record("eval/delta_from_best", delta_from_best)
            self.logger.record("eval/improvement_threshold", self.best_mean_reward + self.reward_improvement_epsilon)
            
            if improved:
                if self.verbose >= 1:
                    print(f"New best mean reward! (delta: {delta_from_best:+.6f})")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = float(mean_reward)
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()
            else:
                if self.verbose >= 2:
                    print(
                        (
                            "[EvalCallback] No improvement: "
                            f"mean_reward={mean_reward:.6f}, best={self.best_mean_reward:.6f}, "
                            f"delta={delta_from_best:+.6f}, epsilon={self.reward_improvement_epsilon:.6f}"
                        )
                    )

            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training
