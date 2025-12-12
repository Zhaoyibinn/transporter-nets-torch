# coding=utf-8
# Adapted from Ravens - Transporter Networks, Zeng et al., 2021
# https://github.com/google-research/ravens
"""Data collection script."""

import os
import numpy as np
from absl import app, flags, logging

from ravens_torch import tasks
from ravens_torch.constants import EXPERIMENTS_DIR, ENV_ASSETS_DIR
from ravens_torch.dataset import Dataset
from ravens_torch.environments.environment import Environment
# import logging
logging.set_verbosity(logging.DEBUG)


flags.DEFINE_string('assets_root', ENV_ASSETS_DIR, '')
flags.DEFINE_string('data_dir', EXPERIMENTS_DIR, '')
flags.DEFINE_bool('disp', False, '')
flags.DEFINE_bool('shared_memory', False, '')
flags.DEFINE_string('task', 'block-insertion', '')
flags.DEFINE_string('mode', 'test', '')
flags.DEFINE_integer('n', 1000, '')
flags.DEFINE_float('sim_speed', -1, '')

flags.DEFINE_bool('all', False, '') 
# 是否从头开始再来一遍

flags.DEFINE_string('gs_render', '', '')
# GS渲染的保存路径 如果不指定就不保存

flags.DEFINE_string('own_scene', '', '')

flags.DEFINE_bool("gs_engine", False, "")
# 是否需要开启GS的渲染

FLAGS = flags.FLAGS


def main(unused_argv):
    assert not(FLAGS.gs_render!= '' and not FLAGS.gs_engine), "If you want to use GS rendering, please specify gs_render path and set gs_engine to True"
    # Initialize environment and task.
    env = Environment(
        FLAGS.assets_root,
        disp=FLAGS.disp,
        shared_memory=FLAGS.shared_memory,
        hz=480,
        sim_speed=FLAGS.sim_speed,
        gs_render=FLAGS.gs_render,
        own_scene=FLAGS.own_scene,
        gs_engine = FLAGS.gs_engine
        )
    task = tasks.names[FLAGS.task]()
    task.mode = FLAGS.mode

    # Initialize scripted oracle agent and dataset.
    agent = task.oracle(env)
    if FLAGS.gs_engine:

        dataset = Dataset(os.path.join(
            FLAGS.data_dir, f'{FLAGS.task}-GS-{task.mode}'),all_flag=FLAGS.all)
    else:
        dataset = Dataset(os.path.join(
            FLAGS.data_dir, f'{FLAGS.task}-{task.mode}'),all_flag=FLAGS.all)

    # Train seeds are even and test seeds are odd.
    seed = dataset.max_seed
    if seed < 0:
        seed = -1 if (task.mode == 'test') else -2

    # Collect training data from oracle demonstrations.
    # dataset.n_episodes = 0
    
    while dataset.n_episodes < FLAGS.n:
        print(f'Oracle demonstration: {dataset.n_episodes + 1}/{FLAGS.n}')
        episode, total_reward = [], 0
        seed += 2
        np.random.seed(seed)
        env.set_task(task)
        obs = env.reset()
        info = None
        reward = 0
        # 运动前采集一次 运动后采集一次
        # 运动前后都采集观察
        # 运动前采集agent生成的目标位姿act 运动后采集奖励和运动之后的位姿
        for _ in range(task.max_steps):
            act = agent.act(obs, info)
            # print('Acting...', act)
            episode.append((obs, act, reward, info))
            obs, reward, done, info = env.step(act)
            total_reward += reward
            print(f'Total Reward: {total_reward} Done: {done}')
            if done:
                break
        episode.append((obs, None, reward, info))

        # Only save completed demonstrations.
        if total_reward > 0.99:
            dataset.add(seed, episode)


if __name__ == '__main__':
    app.run(main)
