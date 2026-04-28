#!/usr/bin/env python
"""Continual SAC training on serial dm_control state-obs tasks.

Each task is trained for ``--task_steps`` environment steps, then the agent
switches to the next task (keeping all weights) and continues training.
The agent sees a fixed observation_space / action_space across all tasks;
observations are zero-padded and actions are sliced to the real task dim.

Example
-------
    python train_continual.py \\
        --tasks cheetah_run,walker_walk,hopper_hop \\
        --obs_dim 64 --act_dim 6 \\
        --task_steps 500000 \\
        --save_dir ./logdir/continual_sac/test/seed_42
"""

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

import tqdm
import wandb
import numpy as np
from absl import app, flags
from ml_collections import config_flags

from jaxrl2.agents import SACLearner
from jaxrl2.data import ReplayBuffer
from jaxrl2.evaluation import evaluate
from jaxrl2.wrappers import wrap_gym
from jaxrl2.envs import ContinualDMCEnv

FLAGS = flags.FLAGS

flags.DEFINE_string("tasks",
                    "cheetah_run,walker_walk,hopper_hop",
                    "Comma-separated list of 'domain_task' strings.")
flags.DEFINE_integer("obs_dim",   64,  "Fixed (padded) observation dimension.")
flags.DEFINE_integer("act_dim",    6,  "Fixed (padded) action dimension.")
flags.DEFINE_integer("task_steps", 100_000,
                     "Environment steps to train on each task.")
flags.DEFINE_integer("task_repeats", 2,
                     "How many times to cycle through the full task list.")
flags.DEFINE_integer("start_training", 1_000,
                     "Steps before training begins (random actions).")
flags.DEFINE_integer("batch_size",  256, "Mini-batch size.")
flags.DEFINE_integer("utd",           1, "Updates per environment step.")
flags.DEFINE_integer("eval_episodes", 5, "Episodes per eval.")
flags.DEFINE_integer("eval_interval", 20_000, "Steps between evals.")
flags.DEFINE_integer("log_interval",  10_000, "Steps between log flushes.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("save_dir",
                    "./logdir/continual_sac/default/seed_none",
                    "Directory for logs.")
flags.DEFINE_string("vd_mode", "disabled",
                    "Value-Disturbance mode. One of 'disabled', 'RI'('first'/'last'), "
                    "'RA'('first'/'last','gaussian')")
flags.DEFINE_boolean("tqdm",  False,  "Show tqdm progress bar.")
flags.DEFINE_boolean("wandb", True,  "Log to wandb.")
config_flags.DEFINE_config_file(
    "config",
    "configs/continual_sac.py",
    "Training hyperparameter config.",
    lock_config=False,
)


def make_env(task_list, obs_dim, act_dim, seed):
    """Create a ContinualDMCEnv and apply standard SAC wrappers."""
    env = ContinualDMCEnv(task_list, obs_dim=obs_dim, act_dim=act_dim, seed=seed)
    env = wrap_gym(env, rescale_actions=False)   # actions already in [-1,1]
    env = __import__('gym').wrappers.RecordEpisodeStatistics(env, deque_size=1)
    return env


def make_eval_env(task_name, obs_dim, act_dim, seed):
    """Single-task eval env (same spaces as training env)."""
    env = ContinualDMCEnv([task_name], obs_dim=obs_dim, act_dim=act_dim, seed=seed)
    env = wrap_gym(env, rescale_actions=False)
    return env


def main(_):
    kwargs = dict(FLAGS.config)
    kwargs.pop("jax_mem_fraction", None)
    redo_cfg = dict(kwargs.pop("redo", {}))
    redo_cfg["frequency"] = redo_cfg["frequency"] * FLAGS.utd
    print(f"Effective ReDo frequency (in env steps): {redo_cfg['frequency']}")

    task_list = [t.strip() for t in FLAGS.tasks.split(',')]
    # Expand task list by repeating: [t1, t2, t3] * 2 → [t1, t2, t3, t1, t2, t3]
    task_schedule = task_list * FLAGS.task_repeats
    total_steps = FLAGS.task_steps * len(task_schedule)

    project, group, run = FLAGS.save_dir.split('/')[-3:]
    if FLAGS.wandb:
        wandb.init(project=project, group=group, name=run,
                   dir=FLAGS.save_dir)
        wandb.config.update(FLAGS)

    env      = make_env(task_list, FLAGS.obs_dim, FLAGS.act_dim, FLAGS.seed)
    eval_env = make_eval_env(task_schedule[0], FLAGS.obs_dim, FLAGS.act_dim, FLAGS.seed + 42)
    env.seed(FLAGS.seed)

    agent = SACLearner(
        FLAGS.seed,
        env.observation_space,
        env.action_space,
        redo=redo_cfg,
        vd_mode=FLAGS.vd_mode,
        **kwargs,
    )

    replay_buffer = ReplayBuffer(
        env.observation_space, env.action_space,
        FLAGS.task_steps          # buffer per task; reset between tasks
    )
    replay_buffer.seed(FLAGS.seed)

    task_idx = 0
    # Load first task in the unwrapped env, then reset through all wrappers
    # so that RecordEpisodeStatistics (and other wrappers) are properly initialised.
    env.unwrapped._load_task(task_schedule[0])
    obs = env.reset()

    global_step = 0   # steps across all tasks
    task_local  = 0   # steps within the current task

    pbar = tqdm.tqdm(total=total_steps, smoothing=0.1, disable=not FLAGS.tqdm)

    while global_step < total_steps:
        # ---- task switch ----
        if task_local >= FLAGS.task_steps + FLAGS.start_training:
            task_idx += 1
            if task_idx >= len(task_schedule):
                break
            cur_task = task_schedule[task_idx]
            repeat_idx = task_idx // len(task_list)  # which cycle we are in
            print(f'\n[Step {global_step}] Cycle {repeat_idx + 1}/{FLAGS.task_repeats} '
                  f'→ task: {cur_task}')
            # Switch task in the inner env, then reset through all wrappers.
            env.unwrapped._load_task(cur_task)
            env.unwrapped.task_step = 0
            obs = env.reset()
            # Rebuild a fresh replay buffer for the new task.
            replay_buffer = ReplayBuffer(
                env.observation_space, env.action_space, FLAGS.task_steps)
            replay_buffer.seed(FLAGS.seed + task_idx)
            # Rebuild eval env for the new task.
            eval_env = make_eval_env(
                cur_task, FLAGS.obs_dim, FLAGS.act_dim, FLAGS.seed + 42)
            task_local = 0
            # reset_all: reinitialise the full SAC agent on every task switch.
            if 'reset_all' in FLAGS.vd_mode:
                agent.reset_agent()
                print(f'  → agent fully reset (vd_mode=reset_all)')

        # ---- collect one step ----
        if task_local < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(obs)

        next_obs, reward, done, info = env.step(action)

        mask = 0.0 if (done and 'TimeLimit.truncated' not in info) else 1.0
        replay_buffer.insert(dict(
            observations=obs,
            actions=action,
            rewards=reward,
            masks=mask,
            dones=done,
            next_observations=next_obs,
        ))
        obs = next_obs

        if done:
            obs = env.reset()

        # ---- training ----
        if task_local >= FLAGS.start_training:
            update_info = {}
            utd_obs, utd_actions = [], []
            for _ in range(FLAGS.utd):
                batch     = replay_buffer.sample(FLAGS.batch_size)
                step_info = agent.update(batch)
                update_info.update(step_info)   # redo metrics appear in whichever call they fire
                utd_obs.append(np.asarray(batch['observations']))
                utd_actions.append(np.asarray(batch['actions']))

            # Aggregated noised-activation rank analysis (fires at same frequency as ReDo).
            if any('redo' in k for k in update_info):
                update_info.update(agent.collect_noised_act_stats(
                    np.concatenate(utd_obs,     axis=0),
                    np.concatenate(utd_actions, axis=0),
                ))

            has_redo = any('redo' in k for k in update_info) or any('grad_redo' in k for k in update_info)
            if FLAGS.wandb and (global_step % FLAGS.log_interval == 0 or has_redo):
                log_dict = {}
                for k, v in update_info.items():
                    if '/redo_noised_agg/' in k:
                        net, _, rest = k.partition('/redo_noised_agg/')
                        log_dict[f'redo_noised_agg/{net}/{rest}'] = v
                    elif '/redo_noised/' in k:
                        net, _, rest = k.partition('/redo_noised/')
                        log_dict[f'redo_noised/{net}/{rest}'] = v
                    elif '/redo/' in k:
                        net, _, rest = k.partition('/redo/')
                        log_dict[f'redo/{net}/{rest}'] = v
                    elif '/grad_redo/' in k:
                        net, _, rest = k.partition('/grad_redo/')
                        log_dict[f'grad_redo/{net}/{rest}'] = v
                    else:
                        log_dict[f'train/{k}'] = v
                wandb.log(log_dict, step=global_step)

        # ---- evaluation ----
        if global_step % FLAGS.eval_interval == 0:
            eval_info = evaluate(agent, eval_env, num_episodes=FLAGS.eval_episodes)
            cur_task = task_schedule[task_idx]
            print(f'[{cur_task} | step {global_step}] return={eval_info["return"]:.1f}')
            if FLAGS.wandb:
                wandb.log({f'performance/{cur_task}/{k}': v
                           for k, v in eval_info.items()}, step=global_step)

        global_step += 1
        task_local  += 1
        pbar.update(1)

    pbar.close()
    env.close()


if __name__ == '__main__':
    app.run(main)
