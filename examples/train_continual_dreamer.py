#!/usr/bin/env python
"""Continual SAC-Dreamer training on serial dm_control state-obs tasks.

Identical to train_continual.py but uses SACDreamerLearner (DreamerV3
size50m-aligned networks: 3 × 512 SiLU, no norm) instead of SACLearner.

Example
-------
    python examples/train_continual_dreamer.py \\
        --tasks finger_spin,walker_walk,cheetah_run,reacher_easy \\
        --obs_dim 32 --act_dim 6 \\
        --task_steps 200000 \\
        --save_dir ./logdir/continual_sac_dreamer/test/seed_42
"""

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

import tqdm
import wandb
import numpy as np
from absl import app, flags
from ml_collections import config_flags

from jaxrl2.agents import SACDreamerLearner
from jaxrl2.data import ReplayBuffer
from jaxrl2.evaluation import evaluate
from jaxrl2.wrappers import wrap_gym
from jaxrl2.envs import ContinualDMCEnv

FLAGS = flags.FLAGS

flags.DEFINE_string("tasks",
                    "finger_spin,walker_walk,cheetah_run,reacher_easy",
                    "Comma-separated list of 'domain_task' strings.")
flags.DEFINE_integer("obs_dim",   32,  "Fixed (padded) observation dimension.")
flags.DEFINE_integer("act_dim",    6,  "Fixed (padded) action dimension.")
flags.DEFINE_integer("task_steps", 200_000,
                     "Environment steps to train on each task.")
flags.DEFINE_integer("task_repeats", 10,
                     "How many times to cycle through the full task list.")
flags.DEFINE_integer("start_training", 10_000,
                     "Steps before training begins (random actions).")
flags.DEFINE_integer("batch_size",  256, "Mini-batch size.")
flags.DEFINE_integer("utd",           1, "Updates per environment step.")
flags.DEFINE_integer("eval_episodes", 5, "Episodes per eval.")
flags.DEFINE_integer("eval_interval", 20_000, "Steps between evals.")
flags.DEFINE_integer("log_interval",  10_000, "Steps between log flushes.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("save_dir",
                    "./logdir/continual_sac_dreamer/default/seed_none",
                    "Directory for logs.")
flags.DEFINE_string("vd_mode", "disabled",
                    "Value-Disturbance mode (passed through but not used by "
                    "SACDreamerLearner's networks; kept for API compatibility).")
flags.DEFINE_boolean("tqdm",  False,  "Show tqdm progress bar.")
flags.DEFINE_boolean("wandb", True,   "Log to wandb.")
config_flags.DEFINE_config_file(
    "config",
    "configs/continual_sac_dreamer.py",
    "Training hyperparameter config.",
    lock_config=False,
)


def make_env(task_list, obs_dim, act_dim, seed):
    env = ContinualDMCEnv(task_list, obs_dim=obs_dim, act_dim=act_dim, seed=seed)
    env = wrap_gym(env, rescale_actions=False)
    env = __import__('gym').wrappers.RecordEpisodeStatistics(env, deque_size=1)
    return env


def make_eval_env(task_name, obs_dim, act_dim, seed):
    env = ContinualDMCEnv([task_name], obs_dim=obs_dim, act_dim=act_dim, seed=seed)
    env = wrap_gym(env, rescale_actions=False)
    return env


def main(_):
    kwargs = dict(FLAGS.config)
    kwargs.pop("jax_mem_fraction", None)
    redo_cfg = dict(kwargs.pop("redo", {}))
    redo_cfg["frequency"] = redo_cfg["frequency"] * FLAGS.utd
    opt_cfg = dict(kwargs.pop("opt", {}))
    # model_size is resolved inside SACDreamerLearner; pass it through.
    # If not present in config, default to None (use hidden_dims directly).
    kwargs.setdefault("model_size", None)

    task_list     = [t.strip() for t in FLAGS.tasks.split(',')]
    task_schedule = task_list * FLAGS.task_repeats
    total_steps   = FLAGS.task_steps * len(task_schedule)

    project, group, run = FLAGS.save_dir.split('/')[-3:]
    if FLAGS.wandb:
        wandb.init(project=project, group=group, name=run,
                   dir=FLAGS.save_dir)
        wandb.config.update(FLAGS)

    env      = make_env(task_list, FLAGS.obs_dim, FLAGS.act_dim, FLAGS.seed)
    eval_env = make_eval_env(task_schedule[0], FLAGS.obs_dim, FLAGS.act_dim,
                             FLAGS.seed + 42)
    env.seed(FLAGS.seed)

    agent = SACDreamerLearner(
        FLAGS.seed,
        env.observation_space,
        env.action_space,
        redo=redo_cfg,
        vd_mode=FLAGS.vd_mode,
        opt=opt_cfg,
        **kwargs,
    )

    replay_buffer = ReplayBuffer(
        env.observation_space, env.action_space, FLAGS.task_steps)
    replay_buffer.seed(FLAGS.seed)

    task_idx   = 0
    env.unwrapped._load_task(task_schedule[0])
    obs = env.reset()

    global_step = 0
    task_local  = 0

    pbar = tqdm.tqdm(total=total_steps, smoothing=0.1, disable=not FLAGS.tqdm)

    while global_step < total_steps:
        # ---- task switch ----
        if task_local >= FLAGS.task_steps + FLAGS.start_training:
            task_idx += 1
            if task_idx >= len(task_schedule):
                break
            cur_task   = task_schedule[task_idx]
            repeat_idx = task_idx // len(task_list)
            print(f'\n[Step {global_step}] Cycle {repeat_idx + 1}/{FLAGS.task_repeats} '
                  f'→ task: {cur_task}')
            env.unwrapped._load_task(cur_task)
            env.unwrapped.task_step = 0
            obs = env.reset()
            replay_buffer = ReplayBuffer(
                env.observation_space, env.action_space, FLAGS.task_steps)
            replay_buffer.seed(FLAGS.seed + task_idx)
            eval_env = make_eval_env(
                cur_task, FLAGS.obs_dim, FLAGS.act_dim, FLAGS.seed + 42)
            task_local = 0
            if 'reset_all' in FLAGS.vd_mode:
                agent.reset_agent()
                print('  → agent fully reset (vd_mode=reset_all)')

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
            for _ in range(FLAGS.utd):
                batch     = replay_buffer.sample(FLAGS.batch_size)
                step_info = agent.update(batch)
                update_info.update(step_info)

            has_redo = any('redo' in k for k in update_info)
            if FLAGS.wandb and (global_step % FLAGS.log_interval == 0 or has_redo):
                log_dict = {}
                # Metric name mapping — mirrors dreamerv3/continual_train.py categories:
                #   loss/   : per-component losses   (dreamer: loss/policy, loss/value, …)
                #   train/  : scalar training stats  (dreamer: train/ent/action, …)
                #   act_redo/: activation-ReDo stats (dreamer: act_redo/…)
                #   grad_redo/: gradient-ReDo stats  (dreamer: grad_redo/…)
                _LOSS_REMAP = {
                    'actor_loss':      'loss/policy',   # dreamer: loss/policy
                    'critic_loss':     'loss/value',    # dreamer: loss/value
                    'temperature_loss': 'loss/temp',    # dreamer: loss/temp (no direct equiv.)
                }
                _TRAIN_REMAP = {
                    'entropy':     'train/ent/action',  # dreamer: train/ent/action
                    'temperature': 'train/temperature', # dreamer: train/temperature
                }
                for k, v in update_info.items():
                    if k in _LOSS_REMAP:
                        log_dict[_LOSS_REMAP[k]] = v
                    elif k in _TRAIN_REMAP:
                        log_dict[_TRAIN_REMAP[k]] = v
                    elif '/redo/' in k:
                        # Convert to dreamer naming: act_redo/{rank_type}/{net_prefix}_linear{i}
                        # e.g. actor/redo/erank/layer_0_act → act_redo/erank/pol_mlp_linear0
                        #      critic/redo/srank/layer_1_act → act_redo/srank/val_mlp_linear1
                        _NET_PREFIX = {'actor': 'pol_mlp', 'critic': 'val_mlp'}
                        net, _, rest = k.partition('/redo/')
                        rank_type, _, lname_raw = rest.partition('/')
                        if lname_raw.startswith('layer_') and lname_raw.endswith('_act'):
                            idx = lname_raw[len('layer_'):-len('_act')]
                            net_prefix = _NET_PREFIX.get(net, net)
                            log_dict[f'act_redo/{rank_type}/{net_prefix}_linear{idx}'] = v
                        else:
                            # fallback for non-standard names (e.g. Dormant_*, Act_Mean/*)
                            log_dict[f'act_redo/{net}/{rest}'] = v
                    elif '/grad_redo/' in k:
                        # Convert to dreamer naming: grad_redo/{rank_type}/{net_prefix}_linear{i}
                        # e.g. actor/grad_redo/GradDormant_0.1/layer_0 → grad_redo/GradDormant_0.1/pol_mlp_linear0
                        #      critic_0/grad_redo/Grad_Mean/layer_1    → grad_redo/Grad_Mean/val_mlp_linear1
                        # critic_1 (second Q ensemble member) is skipped — only critic_0 is logged.
                        _NET_PREFIX_G = {'actor': 'pol_mlp', 'critic': 'val_mlp'}
                        net, _, rest = k.partition('/grad_redo/')
                        rank_type, _, lname_raw = rest.partition('/')
                        # handle vmapped suffix: critic_0 → base=critic, keep; critic_1 → skip
                        parts = net.rsplit('_', 1)
                        if len(parts) == 2 and parts[1].isdigit():
                            if parts[1] != '0':
                                continue  # skip critic_1, critic_2, …
                            net_base = parts[0]
                        else:
                            net_base = net
                        if lname_raw.startswith('layer_'):
                            idx = lname_raw[len('layer_'):]
                            net_prefix = _NET_PREFIX_G.get(net_base, net_base)
                            log_dict[f'grad_redo/{rank_type}/{net_prefix}_linear{idx}'] = v
                        else:
                            # fallback for non-standard names
                            log_dict[f'grad_redo/{net_base}/{rest}'] = v
                    else:
                        log_dict[f'train/{k}'] = v
                wandb.log(log_dict, step=global_step)

        # ---- evaluation ----
        if global_step % FLAGS.eval_interval == 0:
            eval_info = evaluate(agent, eval_env,
                                 num_episodes=FLAGS.eval_episodes)
            cur_task = task_schedule[task_idx]
            print(f'[{cur_task} | step {global_step}] '
                  f'return={eval_info["return"]:.1f}')
            if FLAGS.wandb:
                wandb.log(
                    # Use 'score' to match dreamer's performance/{task}/score.
                    {f'performance/{cur_task}/{"score" if k == "return" else k}': v
                     for k, v in eval_info.items()},
                    step=global_step)

        global_step += 1
        task_local  += 1
        pbar.update(1)

    pbar.close()
    env.close()


if __name__ == '__main__':
    app.run(main)
