import os
import pathlib
import subprocess
from dataclasses import dataclass
from typing import Callable

import gymnasium as gym
import numpy as np
from numpy import ndarray

__all__ = ["frames_to_gif_file", "Sample", "DQN"]


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
        input=np.asarray(frames).tobytes()
    )


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
class Sample:
    observation: ndarray
    action: int
    reward: float
    next_observation: ndarray
    frame: ndarray | None


class DQN:
    def __init__(self, env: gym.Env, observation_callback: Callable[[ndarray], None] = lambda _: None):
        self.env = env
        self.observation_callback = observation_callback

    def choose_action(self) -> int:
        return self.env.action_space.sample()

    def play_one_session(self, render_frames: bool = False):
        terminated, truncated = False, False
        num_iterations = 0
        samples = []
        observation, _ = self.env.reset()
        while not (terminated or truncated):
            num_iterations += 1
            action = self.choose_action()
            next_observation, reward, terminated, truncated, info = self.env.step(action)
            frame = self.env.render() if render_frames else None
            sample = Sample(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                frame=frame,
            )
            samples.append(sample)
            observation = next_observation
            self.observation_callback(next_observation)

            if terminated or truncated:
                reason = "terminated" if terminated else "truncated"
                print(f"Session ended after {num_iterations} iterations: {reason}")
                break
        return samples
