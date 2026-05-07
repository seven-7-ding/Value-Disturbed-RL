#!/usr/bin/env python
"""Continual SAC-Dreamer-Dist training: distributional critic + parallel envs.

Uses SACDreamerDistLearner (TwoHot distributional critic, DreamerV3-aligned
SiLU+RMSNorm networks) with gym.vector parallel environments.

Parallel-env design (mirrors DreamerV3 batch_size / envs pattern):
  - num_envs  worker processes each hold an independent ContinualDMCEnv.
  - One env.step() call advances all workers simultaneously, collecting
    num_envs transitions into the replay buffer.
  - start_training threshold is reached after start_training / num_envs
    *vectorised* steps (i.e. start_training individual transitions).
  - task_steps counts individual transitions (consistent with serial script).

Example
-------
    python examples/train_continual_dreamer_dist.py \\
        --tasks finger_spin,walker_walk,cheetah_run,reacher_easy \\
        --obs_dim 32 --act_dim 6 \\
        --task_steps 200000 \\
        --num_envs 16 \\
        --save_dir ./logdir/continual_sac_dreamer_dist/test/seed_42
"""

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

import tqdm
import wandb
import numpy as np
import gym
import gym.vector
from absl import app, flags
from ml_collections import config_flags

from jaxrl2.agents import SACDreamerDistLearner
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
                     "Environment steps (individual transitions) per task.")
flags.DEFINE_integer("task_repeats", 10,
                     "How many times to cycle through the full task list.")
flags.DEFINE_integer("start_training", 10_000,
                     "Transitions before training begins (random actions).")
flags.DEFINE_integer("batch_size",  1024, "Mini-batch size.")
flags.DEFINE_integer("utd",           1, "Updates per environment step.")
flags.DEFINE_integer("num_envs",     16,
                     "Number of parallel environment workers.")
flags.DEFINE_integer("eval_episodes", 5, "Episodes per eval.")
flags.DEFINE_integer("eval_interval", 20_000, "Steps between evals.")
flags.DEFINE_integer("log_interval",  10_000, "Steps between log flushes.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("save_dir",
                    "./logdir/continual_sac_dreamer_dist/default/seed_none",
                    "Directory for logs.")
flags.DEFINE_string("vd_mode", "disabled",
                    "Value-Disturbance mode (kept for API compatibility).")
flags.DEFINE_boolean("tqdm",  False,  "Show tqdm progress bar.")
flags.DEFINE_boolean("wandb", True,   "Log to wandb.")
config_flags.DEFINE_config_file(
    "config",
    "configs/continual_sac_dreamer_dist.py",
    "Training hyperparameter config.",
    lock_config=False,
)


# ---------------------------------------------------------------------------
# Environment factories
# ---------------------------------------------------------------------------

def _make_single_env(task_list, obs_dim, act_dim, seed):
    """Factory for a single (serial) ContinualDMCEnv."""
    env = ContinualDMCEnv(task_list, obs_dim=obs_dim, act_dim=act_dim, seed=seed)
    env = wrap_gym(env, rescale_actions=False)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    return env


def make_vec_env(task_list, obs_dim, act_dim, seed, num_envs):
    """Create a vectorised environment with num_envs workers.

    Uses SyncVectorEnv so that workers share the same process and the
    underlying ContinualDMCEnv objects remain directly accessible via
    vec_env.envs[i].unwrapped for task switching.
    Each worker gets a unique seed offset to ensure independent exploration.
    """
    def _env_fn(worker_seed):
        def _make():
            return _make_single_env(task_list, obs_dim, act_dim, worker_seed)
        return _make

    env_fns = [_env_fn(seed + i) for i in range(num_envs)]
    vec_env = gym.vector.SyncVectorEnv(env_fns)
    return vec_env


def make_eval_env(task_name, obs_dim, act_dim, seed):
    env = ContinualDMCEnv([task_name], obs_dim=obs_dim, act_dim=act_dim, seed=seed)
    env = wrap_gym(env, rescale_actions=False)
    return env


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(_):
    kwargs = dict(FLAGS.config)
    kwargs.pop("jax_mem_fraction", None)
    redo_cfg = dict(kwargs.pop("redo", {}))
    redo_cfg["frequency"] = redo_cfg["frequency"] * FLAGS.utd
    opt_cfg = dict(kwargs.pop("opt", {}))
    kwargs.setdefault("model_size", None)

    num_envs  = FLAGS.num_envs
    task_list = [t.strip() for t in FLAGS.tasks.split(',')]
    task_schedule = task_list * FLAGS.task_repeats
    # Total individual transitions (num_envs transitions per vec-step).
    total_steps = FLAGS.task_steps * len(task_schedule)

    os.makedirs(FLAGS.save_dir, exist_ok=True)
    project, group, run = FLAGS.save_dir.split('/')[-3:]
    if FLAGS.wandb:
        wandb.init(project=project, group=group, name=run,
                   dir=FLAGS.save_dir)
        wandb.config.update(FLAGS)

    # Build vectorised env.
    vec_env = make_vec_env(task_list, FLAGS.obs_dim, FLAGS.act_dim,
                           FLAGS.seed, num_envs)
    eval_env = make_eval_env(task_schedule[0], FLAGS.obs_dim, FLAGS.act_dim,
                             FLAGS.seed + 42)

    # Use single-env spaces for the agent / replay buffer.
    single_obs_space = vec_env.single_observation_space
    single_act_space = vec_env.single_action_space

    agent = SACDreamerDistLearner(
        FLAGS.seed,
        single_obs_space,
        single_act_space,
        redo=redo_cfg,
        vd_mode=FLAGS.vd_mode,
        opt=opt_cfg,
        **kwargs,
    )

    replay_buffer = ReplayBuffer(
        single_obs_space, single_act_space, 5_000_000)
    replay_buffer.seed(FLAGS.seed)

    task_idx   = 0
    # Switch all workers to the first task.
    # vec_env.envs[i] is RecordEpisodeStatistics(wrap_gym(ContinualDMCEnv(...)));
    # .unwrapped walks the wrapper chain down to ContinualDMCEnv.
    for w in vec_env.envs:
        w.unwrapped._load_task(task_schedule[0])

    obs = vec_env.reset()   # [num_envs, obs_dim]

    # global_step counts *individual* transitions (not vec-steps).
    global_step = 0
    task_local  = 0         # individual transitions on this task

    pbar = tqdm.tqdm(total=total_steps, smoothing=0.1, disable=not FLAGS.tqdm)

    while global_step < total_steps:
        # ---- task switch ----
        if task_local >= FLAGS.task_steps:
            task_idx += 1
            if task_idx >= len(task_schedule):
                break
            cur_task   = task_schedule[task_idx]
            repeat_idx = task_idx // len(task_list)
            print(f'\n[Step {global_step}] Cycle {repeat_idx + 1}/{FLAGS.task_repeats} '
                  f'→ task: {cur_task}')
            # Switch all workers — .unwrapped reaches ContinualDMCEnv.
            for w in vec_env.envs:
                w.unwrapped._load_task(cur_task)
                w.unwrapped.task_step = 0
            obs = vec_env.reset()
            replay_buffer = ReplayBuffer(
                single_obs_space, single_act_space, FLAGS.task_steps)
            replay_buffer.seed(FLAGS.seed + task_idx)
            eval_env = make_eval_env(
                cur_task, FLAGS.obs_dim, FLAGS.act_dim, FLAGS.seed + 42)
            task_local = 0
            if 'reset_all' in FLAGS.vd_mode:
                agent.reset_agent()
                print('  → agent fully reset (vd_mode=reset_all)')

        # ---- collect one vectorised step (num_envs transitions) ----
        if task_local < FLAGS.start_training:
            # Random actions during warm-up.
            actions = np.array([single_act_space.sample() for _ in range(num_envs)])
        else:
            # Agent selects actions for all envs in one batched call.
            actions = agent.sample_actions(obs)   # [num_envs, act_dim]

        next_obs, rewards, dones, infos = vec_env.step(actions)
        # next_obs: [num_envs, obs_dim]  (auto-reset applied by AsyncVectorEnv)

        # Normalise infos to a list-of-dicts regardless of gym version.
        # Older gym returns a list; newer gym (>=0.26) returns a dict-of-arrays.
        if isinstance(infos, list):
            per_env_infos = infos
        else:
            per_env_infos = [{} for _ in range(num_envs)]
            for _key, _values in infos.items():
                if _key.startswith('_'):
                    continue  # skip boolean mask arrays like '_TimeLimit.truncated'
                _mask = infos.get(f'_{_key}', None)
                for _j in range(num_envs):
                    if _mask is None or _mask[_j]:
                        per_env_infos[_j][_key] = _values[_j]

        for i in range(num_envs):
            # gym.vector sets 'TimeLimit.truncated' only when a true timeout
            # (not a terminal done) fires.  The final_observation key holds
            # the real last obs before auto-reset.
            truncated = per_env_infos[i].get('TimeLimit.truncated', False)
            mask = 0.0 if (dones[i] and not truncated) else 1.0

            # When the episode was auto-reset, the real next_obs for the
            # *transition* is the terminal obs, not the reset obs.
            if dones[i]:
                # gym.vector stores the terminal observation in info.
                terminal_obs = per_env_infos[i].get('terminal_observation', next_obs[i])
            else:
                terminal_obs = next_obs[i]

            replay_buffer.insert(dict(
                observations=obs[i],
                actions=actions[i],
                rewards=float(rewards[i]),
                masks=mask,
                dones=bool(dones[i]),
                next_observations=terminal_obs,
            ))

        obs = next_obs
        global_step += num_envs
        task_local  += num_envs
        pbar.update(num_envs)

        # ---- training ----
        if task_local >= FLAGS.start_training:
            # Perform UTD gradient updates; each update uses num_envs transitions.
            update_info = {}
            for _ in range(FLAGS.utd * num_envs):
                if len(replay_buffer) >= FLAGS.batch_size:
                    batch     = replay_buffer.sample(FLAGS.batch_size)
                    step_info = agent.update(batch)
                    update_info.update(step_info)

            has_redo = any('redo' in k for k in update_info)
            if FLAGS.wandb and (global_step % FLAGS.log_interval < num_envs or has_redo):
                log_dict = {}
                _LOSS_REMAP = {
                    'actor_loss':       'loss/policy',
                    'critic_loss':      'loss/value',
                    'temperature_loss': 'loss/temp',
                }
                _TRAIN_REMAP = {
                    'entropy':     'train/ent/action',
                    'temperature': 'train/temperature',
                    'q':           'train/q_mean',
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
        if global_step % FLAGS.eval_interval < num_envs and task_local >= FLAGS.start_training:
            eval_info = evaluate(agent, eval_env, FLAGS.eval_episodes)
            cur_task = task_schedule[task_idx]
            print(f'[{cur_task} | step {global_step}] '
                  f'return={eval_info["return"]:.1f}')
            if FLAGS.wandb:
                wandb.log(
                    {f'performance/{cur_task}/{"score" if k == "return" else k}': v
                     for k, v in eval_info.items()},
                    step=global_step)

    pbar.close()
    vec_env.close()
    print("Training complete.")


if __name__ == "__main__":
    app.run(main)
