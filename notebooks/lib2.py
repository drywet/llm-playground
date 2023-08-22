import collections.abc
import dataclasses
import os
import pathlib
import random
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Final

import gymnasium as gym
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from gymnasium import Space
from numpy import ndarray

__all__ = ["frames_to_gif_file", "store_dqn_images", "archive", "unarchive", "Transition", "DQN"]

from tensorflow.python.keras import Model
from tensorflow.python.keras.utils import losses_utils


def run(cmd: str, input):
    res = subprocess.run(cmd.split(' '), capture_output=True, input=input)
    if res.returncode != 0:
        raise Exception(f"Error calling a command: {res.stderr}")
    return res


def frames_to_gif_file(frames: list[ndarray], output_file_path: str, fps: int = 60):
    assert ' ' not in output_file_path
    assert len(frames) > 0
    assert len(frames[0].shape) == 3
    width = frames[0].shape[1]
    height = frames[0].shape[0]
    os.makedirs(pathlib.Path(output_file_path).parent, exist_ok=True)
    # For mp4 output:
    # -f mp4 -c:v libx264 -pix_fmt yuv420p -an
    run(
        f"ffmpeg -y -f rawvideo -pix_fmt rgb24 -s {width}x{height} -r {fps} -i pipe:0 -f gif {output_file_path}",
        input=np.array(frames).tobytes()
    )


def archive(dir: Path):
    with zipfile.ZipFile(f"{dir}.zip", "w", zipfile.ZIP_DEFLATED) as zip_file:
        for entry in dir.rglob("*"):
            zip_file.write(entry, entry.relative_to(dir))


def unarchive(file: Path):
    with zipfile.ZipFile(file, "r", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.extractall(file.with_suffix(''))


def observation_callback_1(observation):
    # x, y, vx, vy, r, vr, landedl, landedr = observation
    # print(f"")
    # print(f"i={frame_index}")
    # # print(f"observation={observation}")
    # print(f"x={x}")
    # print(f"y={y}")
    # # print(f"vx={vx}")
    # # print(f"vy={vy}")
    # # print(f"r={r}")
    # # print(f"vr={vr}")
    # # print(f"landedl={landedl}")
    # # print(f"landedr={landedr}")
    # print(f"reward={reward}")
    # # print(f"info={info}")
    pass


@dataclass(frozen=True)
class Transition:
    observation: ndarray
    action: int
    reward: float
    next_observation: ndarray
    terminal: bool
    frame: ndarray | None
    # For charts; not updated during learning
    observation_original_q_values: ndarray
    next_observation_original_q_values: ndarray


def store_dqn_images(episode: int, transitions: list[Transition], file_path: pathlib.Path):
    if transitions[0].frame is not None:
        # Frames
        filename = f"{file_path}_{episode:04}"
        frames_to_gif_file([t.frame for t in transitions], f"{filename}.gif")
        # Charts
        color_map_1 = mpl.colormaps['spring']
        color_map_2 = mpl.colormaps['winter']
        indices = list(range(len(transitions)))
        rewards = [t.reward for t in transitions]
        observation_original_q_values_list: ndarray = (
            np.array([t.observation_original_q_values for t in transitions]).transpose()
        )
        next_observation_original_q_values_list: ndarray = (
            np.array([t.next_observation_original_q_values for t in transitions]).transpose()
        )
        fig, (ax1, ax2) = plt.subplots(nrows=2)
        ax1.set_title("Reward")
        ax1.plot(indices, rewards)
        ax2.set_title("Action values")
        for index, values in enumerate(observation_original_q_values_list):
            ax2.plot(indices, values, color=color_map_1(index / len(observation_original_q_values_list)))
        for index, values in enumerate(next_observation_original_q_values_list):
            ax2.plot(indices, values, color=color_map_2(index / len(next_observation_original_q_values_list)))
        plt.savefig(f"{filename}.png", dpi=300)
        plt.cla()
        plt.close(fig)
        return None


class DqnMeanSquaredError(tf.keras.losses.MeanSquaredError):
    """A variation that replaces -inf y_true params with corresponding params from y_pred"""

    def __init__(self, reduction=losses_utils.ReductionV2.AUTO, name="dqn_mean_squared_error"):
        super().__init__(reduction, name)

    def call(self, y_true, y_pred):
        # print(f"loss function call, \n\ty_true: {y_true}, \n\ty_pred: {y_pred}")
        stacked = tf.stack([y_true, y_pred], axis=1)
        # print(f"stacked {stacked}")
        y_true_new = tf.vectorized_map(
            lambda yt_yp: tf.where(
                tf.vectorized_map(lambda y: y != np.NINF, yt_yp[0]),
                yt_yp[0],
                yt_yp[1],
            ),
            stacked
        )
        # print(f"y_true_new {y_true_new}")
        res = super().call(y_true_new, y_pred)
        # print(f"res {res}")
        return res


class DQN:
    # max_transitions_stored: Final[int] = 1000
    random_generator = np.random.default_rng()

    episode: int = 0
    all_transitions: Final[list[Transition]] = []
    n_actions: Final[int]
    observation_space_size: Final[int]
    output_size: Final[int]
    q_network: Final[tf.keras.Model]

    @staticmethod
    def num_actions(action_space: Space):
        assert len(action_space.shape) == 0
        assert action_space.__class__ == gym.spaces.discrete.Discrete
        # noinspection PyTypeChecker
        action_space_: gym.spaces.discrete.Discrete = action_space
        return action_space_.n - action_space_.start

    @staticmethod
    def get_observation_space_size(observation_space: Space):
        assert len(observation_space.shape) == 1
        assert observation_space.__class__ == gym.spaces.box.Box
        return observation_space.shape[0]

    @staticmethod
    def create_q_network(observation_space: Space, n_actions: int, output_size: int) -> Model:
        # Checking the idea of predicting the next state and the reward along with the state value.
        # 1. Optimizing for predicting extra stuff can serve as a clue for the network helping to find better policy.
        # 2. It's interesting to compare action values of the current state vs
        # estimated action reward + max action value of the next predicted state given the action.
        # 3. If rewards and states turn out to be predicted better than action values,
        # prediction of rewards + states can be used as a learned model that can be used to traverse the state tree
        # and plan ahead.
        # Result: no significant performance improvement; state and reward predictions are quite off.

        # If observation space was too large, the algorithm would need to use a compressed representation,
        # i.e. an embedding
        model = tf.keras.Sequential([
            tf.keras.layers.Input(observation_space.shape, name="model_input", dtype=float),
            tf.keras.layers.Dense(512, activation=tf.keras.activations.gelu, name="dense_1"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation=tf.keras.activations.gelu, name="dense_2"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation=tf.keras.activations.gelu, name="dense_3"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation=tf.keras.activations.gelu, name="dense_4"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(64, activation=tf.keras.activations.gelu, name="dense_5"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(n_actions * output_size, activation=tf.keras.activations.linear, name=f"output"),
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss=DqnMeanSquaredError(),
        )
        return model

    def __init__(self, env: gym.Env, observation_callback: Callable[[ndarray], None] = lambda _: None):
        self.env = env
        self.observation_callback = observation_callback
        self.n_actions = DQN.num_actions(env.action_space)
        self.observation_space_size = DQN.get_observation_space_size(env.observation_space)
        # action value + reward + observation space
        self.output_size = 2 + self.observation_space_size
        self.q_network = DQN.create_q_network(env.observation_space, self.n_actions, self.output_size)

    def predict(self, observation: ndarray) -> ndarray:
        assert observation.shape == (self.observation_space_size,)
        res: ndarray = self.q_network(np.expand_dims(observation, axis=0))[0]
        assert len(res) == self.n_actions * self.output_size
        return res

    def choose_random_action(self, action_values: ndarray) -> int:
        return self.env.action_space.sample()

    def choose_action(self, action_values: ndarray, epsilon: float) -> int:
        assert action_values.shape == (self.n_actions,)
        assert 0 <= epsilon <= 1
        if random.random() < epsilon:
            action = self.env.action_space.sample()
            print(f"Random action: {action}")
            return action
        else:
            # noinspection PyTypeChecker
            action: int = np.argmax(action_values)
            # softmax = tf.math.softmax(action_values).numpy().astype(np.float32)
            # action: int = self.random_generator.choice(len(action_values), p=softmax)
            # print(f"Action values: {action_values}; best action: {action}")
            return action

    def extract_action_values(self, prediction: ndarray) -> ndarray:
        assert prediction.shape == (self.n_actions * self.output_size,)
        res = prediction[::self.output_size]
        assert len(res) == self.n_actions
        return res

    def extract_rewards(self, prediction: ndarray) -> ndarray:
        assert prediction.shape == (self.n_actions * self.output_size,)
        res = prediction[1::self.output_size]
        assert len(res) == self.n_actions
        return res

    def extract_next_observations(self, prediction: ndarray) -> ndarray:
        assert prediction.shape == (self.n_actions * self.output_size,)
        res = tf.reshape(prediction, (self.n_actions, self.output_size))[:, 2:]
        assert len(res) == self.n_actions
        assert len(res[0]) == self.observation_space_size
        return res

    def play_one_episode(self,
                         render_frames: bool = False,
                         predict: Callable[[ndarray], ndarray] | None = None,
                         choose_action: Callable[[ndarray], int] | None = None,
                         learn: Callable[[None], None] | None = None,
                         learning_period: int = 1,
                         ) -> list[Transition]:
        if predict is None:
            predict = lambda _: np.zeros(self.n_actions * self.output_size, dtype=float)
        if choose_action is None:
            choose_action = self.choose_random_action
        if learn is None:
            learn: Callable[[None], None] = lambda: None
        assert learning_period > 0

        terminated, truncated = False, False
        transitions = []
        observation, _ = self.env.reset()
        num_observations = 1
        while not (terminated or truncated):
            prediction = predict(observation)
            # print(f"prediction: {prediction}")
            action_values = self.extract_action_values(prediction)

            predicted_rewards = self.extract_rewards(prediction)
            predicted_next_observations = self.extract_next_observations(prediction)
            # TODO model.predict() for efficiency
            predicted_next_max_action_values = [np.max(self.extract_action_values(predict(observation)))
                                                for observation in predicted_next_observations]
            predicted_with_future_action_values = [reward + max_action_value
                                                   for reward, max_action_value
                                                   in zip(predicted_rewards, predicted_next_max_action_values)]
            immediate_vs_future_predicted_action_value = tf.metrics.mean_squared_error(action_values,
                                                                                       predicted_with_future_action_values)
            # print(f"immediate_vs_future_predicted_action_value: {immediate_vs_future_predicted_action_value}")
            # print(f"action_value: {action_values}")
            # print(f"predicted_with_future_action_values: {predicted_with_future_action_values}")

            action = choose_action(action_values + predicted_with_future_action_values)
            next_observation, reward, terminated, truncated, info = self.env.step(action)

            # Experiment: adjust reward so that it directs the ship towards the landing pad
            # Well, actually the env already has such a reward
            # landing_pad_proximity_reward = -math.sqrt(next_observation[0] ** 2 + next_observation[1] ** 2) * 5
            # print(f"landing_pad_proximity_reward: {landing_pad_proximity_reward}")
            # reward += landing_pad_proximity_reward

            num_observations += 1
            frame = self.env.render() if render_frames else None
            # noinspection PyTypeChecker
            reward_float: float = reward
            observation_original_q_values = action_values
            next_observation_original_q_values = self.extract_action_values(predict(next_observation))

            # print(f"actual_reward: {reward_float}")
            # print(f"predicted_reward: {predicted_rewards[action]}")
            # print(f"_____observation: {observation}")
            # print(f"next_observation: {next_observation}")
            # print(f"predicted_next_observation: {predicted_next_observations[action]}")

            transition = Transition(
                observation=observation,
                action=action,
                reward=reward_float,
                next_observation=next_observation,
                terminal=terminated or truncated,
                frame=frame,
                observation_original_q_values=observation_original_q_values,
                next_observation_original_q_values=next_observation_original_q_values,
            )
            transitions.append(transition)
            self.all_transitions.append(dataclasses.replace(transition, frame=None))
            observation = next_observation
            self.observation_callback(next_observation)

            if (len(self.all_transitions) - 1) % learning_period == 0:
                learn()

            if terminated or truncated:
                reason = "terminated" if terminated else "truncated"
                print(f"Episode {self.episode} ended after {num_observations} iterations: {reason}")
                break
        return transitions

    @staticmethod
    def make_y_vector(action: int,
                      q_value: float,
                      reward: float,
                      observation: ndarray,
                      n_actions: int,
                      ) -> ndarray:
        output_size = 2 + len(observation)
        y = np.full(n_actions * output_size, np.NINF)
        y[action * output_size] = q_value
        y[action * output_size + 1] = reward
        y[(action * output_size + 2):(action * output_size + 2 + len(observation))] = observation
        return y

    def learn_after_episode(self,
                            max_learning_batch_size: int,
                            gamma: float,
                            ):
        assert max_learning_batch_size > 0
        assert 0 <= gamma <= 1
        batch_size = min(int(len(self.all_transitions) * 0.2), max_learning_batch_size)
        print(f"batch_size: {batch_size}")
        batch: list[Transition] = np.random.choice(self.all_transitions, batch_size).tolist()
        non_terminal_transitions: list[Transition] = [t for t in batch if not t.terminal]
        terminal_transitions: list[Transition] = [t for t in batch if t.terminal]
        # Need to compute action-value function for next observation only for non-terminal transitions.
        # For terminal transitions it equals zero
        non_terminal_next_observation_predictions: ndarray
        # print(f"len(non_terminal_transitions): {len(non_terminal_transitions)}")
        # print(f"len(terminal_transitions): {len(terminal_transitions)}")
        if len(non_terminal_transitions) > 0:
            # batch, action, output
            non_terminal_next_observation_predictions = np.array(self.q_network.predict(
                np.array([t.next_observation for t in non_terminal_transitions])
            ))
        else:
            non_terminal_next_observation_predictions = np.empty((0, self.n_actions * self.output_size))
        assert len(non_terminal_next_observation_predictions.shape) == 2
        assert non_terminal_next_observation_predictions.shape[0] == len(non_terminal_transitions)
        assert non_terminal_next_observation_predictions.shape[1] == self.n_actions * self.output_size
        non_terminal_next_observation_max_q: list[float] = [np.max(prediction[::self.output_size])
                                                            for prediction in non_terminal_next_observation_predictions]
        assert len(non_terminal_transitions) == len(non_terminal_next_observation_max_q)
        # y is a pair of the action performed and its value.
        # The custom loss function expects such input format for y_target
        y_non_terminal: list[ndarray] = [
            DQN.make_y_vector(
                action=t.action,
                q_value=t.reward + gamma * next_observation_max_q,
                reward=t.reward,
                observation=t.next_observation,
                n_actions=self.n_actions,
            )
            for t, next_observation_max_q in
            zip(non_terminal_transitions, non_terminal_next_observation_max_q)]
        y_terminal: list[ndarray] = [DQN.make_y_vector(
            action=t.action,
            q_value=t.reward,
            reward=t.reward,
            observation=t.next_observation,
            n_actions=self.n_actions,
        ) for t in terminal_transitions]
        # action, batch, output
        y = np.array(y_non_terminal + y_terminal)
        x_non_terminal: list[ndarray] = [t.observation for t in non_terminal_transitions]
        x_terminal: list[ndarray] = [t.observation for t in terminal_transitions]
        x: ndarray = np.array(x_non_terminal + x_terminal)
        # print(f"len(x): {len(x)}")
        # print(f"len(y): {len(y)}")
        assert len(x) == len(y)
        if len(x) > 0:
            self.q_network.fit(x, y, batch_size=x.size, epochs=1)

    def learn(
            self,
            render_period: int = 0,
            learning_period: int = 1,
            max_learning_batch_size: int = 1000,
            epsilon: float = 0.05,
            gamma: float = 0.9,
    ) -> collections.abc.Iterator[(int, list[Transition])]:
        assert render_period >= 0
        assert learning_period > 0
        assert max_learning_batch_size > 0
        assert 0 <= epsilon <= 1
        assert 0 <= gamma <= 1
        while True:
            print(f"Episode {self.episode}")
            render_frames = (render_period > 0) and (self.episode % render_period == 0)
            transitions = self.play_one_episode(
                render_frames=render_frames,
                predict=self.predict,
                choose_action=lambda action_values: self.choose_action(action_values, epsilon),
                learn=lambda: self.learn_after_episode(
                    max_learning_batch_size=max_learning_batch_size,
                    gamma=gamma,
                ),
                learning_period=learning_period,
            )
            yield self.episode, transitions
            self.episode += 1
