"""Example of using RLlib's debug callbacks.
Here we use callbacks to track the average CartPole pole angle magnitude as a
custom metric.
"""

from typing import Dict, Tuple
import argparse
import numpy as np
import os

import ray
from ray import air, tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch

parser = argparse.ArgumentParser()
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "tfe", "torch"],
    default="tf",
    help="The DL framework specifier.",
)
parser.add_argument("--stop-iters", type=int, default=2000)


class MyCallbacks(DefaultCallbacks):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # Make sure this episode has just been started (only initial obs
        # logged so far).
        assert episode.length == 0, (
            "ERROR: `on_episode_start()` callback should be called right "
            "after env reset!"
        )
        print("episode {} (env-idx={}) started.".format(episode.episode_id, env_index))
        episode.user_data["is_success"] = []
        episode.hist_data["is_success"] = []

    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # Make sure this episode is ongoing.
        assert episode.length > 0, (
            "ERROR: `on_episode_step()` callback should not be called right "
            "after env reset!"
        )
        success = episode.last_info_for()['success']
        episode.user_data["is_success"].append(bool(success))
#        episode.user_data["is_success"] |= bool(success)

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # Check if there are multiple episodes in a batch, i.e.
        # "batch_mode": "truncate_episodes".
        if worker.policy_config["batch_mode"] == "truncate_episodes":
            # Make sure this episode is really done.
            assert episode.batch_builder.policy_collectors["default_policy"].batches[
                -1
            ]["dones"][-1], (
                "ERROR: `on_episode_end()` should only be called "
                "after episode is done!"
            )
            
        is_success = True if True in episode.user_data["is_success"] else False
        print(
            "episode {} (env-idx={}) ended with length {} and "
            "sucess {}".format(
                episode.episode_id, env_index, episode.length, is_success
            )
        )
        episode.custom_metrics["is_success"] = is_success
        episode.hist_data["is_success"] = episode.user_data["is_success"]


    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        print(
            "Algorithm.train() result: {} -> {} episodes".format(
                algorithm, result["episodes_this_iter"]
            )
        )
        #print("success: {}".format(result["custom_metrics"]["is_success"]))
        # you can mutate the result dict to add new fields to return
        result["callback_ok"] = True

    # def on_learn_on_batch(
    #     self, *, policy: Policy, train_batch: SampleBatch, result: dict, **kwargs
    # ) -> None:
    #     result["sum_actions_in_train_batch"] = np.sum(train_batch["actions"].cpu().numpy())
    #     print(
    #         "policy.learn_on_batch() result: {} -> sum actions: {}".format(
    #             policy, result["sum_actions_in_train_batch"]
    #         )
    #     )

    def on_postprocess_trajectory(
        self,
        *,
        worker: RolloutWorker,
        episode: Episode,
        agent_id: str,
        policy_id: str,
        policies: Dict[str, Policy],
        postprocessed_batch: SampleBatch,
        original_batches: Dict[str, Tuple[Policy, SampleBatch]],
        **kwargs
    ):
        #print("postprocessed {} steps".format(postprocessed_batch.count))
        if "num_batches" not in episode.custom_metrics:
            episode.custom_metrics["num_batches"] = 0
        episode.custom_metrics["num_batches"] += 1

if __name__ == "__main__":
    args = parser.parse_args()

    ray.init()
    tuner = tune.Tuner(
        "PG",
        run_config=air.RunConfig(
            stop={
                "training_iteration": args.stop_iters,
            },
        ),
        param_space={
            "env": "CartPole-v0",
            "num_envs_per_worker": 2,
            "callbacks": MyCallbacks,
            "framework": args.framework,
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        },
    )
    # there is only one trial involved.
    result = tuner.fit().get_best_result()

    # Verify episode-related custom metrics are there.
    custom_metrics = result.metrics["custom_metrics"]
    print(custom_metrics)
    assert "pole_angle_mean" in custom_metrics
    assert "pole_angle_min" in custom_metrics
    assert "pole_angle_max" in custom_metrics
    assert "num_batches_mean" in custom_metrics
    assert "callback_ok" in result.metrics