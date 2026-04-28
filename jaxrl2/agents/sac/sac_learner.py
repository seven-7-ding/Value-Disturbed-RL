"""Implementations of algorithms for continuous control."""

import copy
import functools
from typing import Dict, Optional, Sequence, Tuple

import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState

from jaxrl2.agents.agent import Agent
from jaxrl2.agents.sac.actor_updater import update_actor
from jaxrl2.agents.sac.critic_updater import update_critic
from jaxrl2.agents.sac.temperature import Temperature
from jaxrl2.agents.sac.temperature_updater import update_temperature
from jaxrl2.networks.normal_tanh_policy import NormalTanhPolicy
from jaxrl2.networks.values import StateActionEnsemble
from jaxrl2.types import Params, PRNGKey
from jaxrl2.utils.target_update import soft_target_update
from jaxrl2.utils.redo import SACReDo, SACGradientReDo


@functools.partial(jax.jit, static_argnames=("backup_entropy", "critic_reduction"))
def _update_jit(
    rng: PRNGKey,
    actor: TrainState,
    critic: TrainState,
    target_critic_params: Params,
    temp: TrainState,
    batch: FrozenDict,
    discount: float,
    tau: float,
    target_entropy: float,
    backup_entropy: bool,
    critic_reduction: str,
) -> Tuple[PRNGKey, TrainState, TrainState, Params, TrainState, Dict[str, float]]:

    rng, key = jax.random.split(rng)
    target_critic = critic.replace(params=target_critic_params)
    new_critic, critic_info = update_critic(
        key,
        actor,
        critic,
        target_critic,
        temp,
        batch,
        discount,
        backup_entropy=backup_entropy,
        critic_reduction=critic_reduction,
    )
    new_target_critic_params = soft_target_update(
        new_critic.params, target_critic_params, tau
    )

    rng, key = jax.random.split(rng)
    new_actor, actor_info = update_actor(key, actor, new_critic, temp, batch)
    new_temp, alpha_info = update_temperature(
        temp, actor_info["entropy"], target_entropy
    )

    return (
        rng,
        new_actor,
        new_critic,
        new_target_critic_params,
        new_temp,
        {**critic_info, **actor_info, **alpha_info},
    )


def _collect_activations(train_state: TrainState, obs: np.ndarray) -> Dict:
    """Run a forward pass with mutable='intermediates' to collect sow'd activations.

    Returns a flat dict  ``{'MLP_0/Dense_0': act, ...}``  suitable for ReDo.
    The sow calls in MLP store post-activation tensors under 'intermediates'.
    """
    _, state = train_state.apply_fn(
        {'params': train_state.params},
        obs,
        mutable=['intermediates'],
    )
    # state['intermediates'] is a nested FrozenDict; flatten to slash-paths.
    intermediates = state.get('intermediates', {})

    flat = {}
    def _walk(node, prefix):
        if not (isinstance(node, dict) or hasattr(node, 'items')):
            # leaf – unwrap sow tuple (sow accumulates as tuples)
            flat[prefix] = node[0] if isinstance(node, tuple) and len(node) == 1 else node
            return
        for k, v in node.items():
            _walk(v, f'{prefix}/{k}' if prefix else k)

    _walk(intermediates, '')
    return flat


class SACLearner(Agent):
    def __init__(
        self,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        target_entropy: Optional[float] = None,
        backup_entropy: bool = True,
        critic_reduction: str = "min",
        init_temperature: float = 1.0,
        vd_mode: str = "disabled",
        # ReDo config – mirrors the 'redo' block in dreamerv3/configs.yaml
        redo: Optional[Dict] = None,
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """

        action_dim = action_space.shape[-1]

        if target_entropy is None:
            self.target_entropy = -action_dim / 2
        else:
            self.target_entropy = target_entropy

        self.backup_entropy = backup_entropy
        self.critic_reduction = critic_reduction

        self.tau = tau
        self.discount = discount

        observations = observation_space.sample()
        actions = action_space.sample()

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, critic_noise_key, temp_key = jax.random.split(rng, 5)

        if np.all(action_space.low == -1) and np.all(action_space.high == 1):
            low = None
            high = None
        else:
            low = action_space.low
            high = action_space.high

        actor_def = NormalTanhPolicy(hidden_dims, action_dim, low=low, high=high)
        actor_params = actor_def.init(actor_key, observations)["params"]
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=actor_lr),
        )

        critic_def = StateActionEnsemble(hidden_dims, num_qs=2, vd_mode=vd_mode)
        critic_params = critic_def.init(
            {"params": critic_key, "noise": critic_noise_key}, observations, actions
        )["params"]
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=critic_lr),
        )
        target_critic_params = copy.deepcopy(critic_params)

        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)["params"]
        temp = TrainState.create(
            apply_fn=temp_def.apply,
            params=temp_params,
            tx=optax.adam(learning_rate=temp_lr),
        )

        self._actor = actor
        self._critic = critic
        self._target_critic_params = target_critic_params
        self._temp = temp
        self._rng = rng
        self._vd_mode = vd_mode

        # Save construction state for reset_agent().
        self._actor_def       = actor_def
        self._critic_def      = critic_def
        self._temp_def        = temp_def
        self._actor_lr        = actor_lr
        self._critic_lr       = critic_lr
        self._temp_lr         = temp_lr
        self._init_temperature = init_temperature
        self._observations_sample = observations
        self._actions_sample      = actions

        # Build ReDo objects from config dict (mirrors configs.yaml 'redo' block).
        redo = redo or {}
        redo_kw = dict(
            tau=redo.get('tau', 0.05),
            mode=redo.get('mode', 'threshold'),
            frequency=redo.get('frequency', 1000),
            log_item=redo.get('log_item', 'disabled'),
            rank_threshold=redo.get('rank_threshold', 0.99),
            reset_start=redo.get('reset_start', 0),
            reset_end=redo.get('reset_end', 0),
        )
        self._rank_threshold = redo_kw['rank_threshold']
        # Activation-based ReDo (one instance per network).
        skip = redo.get('skip_last_layer', False)
        self._actor_redo  = SACReDo(name='actor',  **redo_kw, skip_last_layer=skip) \
            if redo.get('redo_enabled', False) else None
        self._critic_redo = SACReDo(name='critic', **redo_kw, skip_last_layer=skip) \
            if redo.get('redo_enabled', False) else None
        # Gradient-based ReDo (one instance per network).
        grad_kw = {k: v for k, v in redo_kw.items() if k != 'rank_threshold'}
        self._actor_grad_redo  = SACGradientReDo(name='actor',  **grad_kw) \
            if redo.get('grad_redo_enabled', False) else None
        self._critic_grad_redo = SACGradientReDo(name='critic', **grad_kw) \
            if redo.get('grad_redo_enabled', False) else None

    # ------------------------------------------------------------------
    # Agent reset
    # ------------------------------------------------------------------

    def reset_agent(self) -> None:
        """Reinitialise all network parameters and optimiser states from scratch.

        Called when vd_mode == 'reset_all' at every task switch.
        ReDo step counters are also reset so frequency gating restarts cleanly.
        """
        self._rng, actor_key, critic_key, critic_noise_key, temp_key = \
            jax.random.split(self._rng, 5)

        actor_params = self._actor_def.init(actor_key, self._observations_sample)["params"]
        self._actor = self._actor.replace(
            params=actor_params,
            opt_state=optax.adam(learning_rate=self._actor_lr).init(actor_params),
            step=0,
        )

        critic_params = self._critic_def.init(
            {"params": critic_key, "noise": critic_noise_key},
            self._observations_sample, self._actions_sample
        )["params"]
        self._critic = self._critic.replace(
            params=critic_params,
            opt_state=optax.adam(learning_rate=self._critic_lr).init(critic_params),
            step=0,
        )
        self._target_critic_params = copy.deepcopy(critic_params)

        temp_params = self._temp_def.init(temp_key)["params"]
        self._temp = self._temp.replace(
            params=temp_params,
            opt_state=optax.adam(learning_rate=self._temp_lr).init(temp_params),
            step=0,
        )

        # Reset ReDo step counters.
        for obj in (self._actor_redo, self._critic_redo,
                    self._actor_grad_redo, self._critic_grad_redo):
            if obj is not None:
                obj._step = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_act_redo(self, obs: np.ndarray, actions: np.ndarray) -> Dict:
        """Run activation-based ReDo for actor and/or critic.

        Called only when at least one instance has should_run()==True;
        each instance whose should_run() is True will do analysis+optional reset.
        Instances that are not due this step have already been ticked by update().
        """
        info = {}
        rng = self._rng

        if self._actor_redo is not None and self._actor_redo.should_run():
            act_acts = _collect_activations(self._actor, obs)
            rng, key = jax.random.split(rng)
            new_p, mets = self._actor_redo.step(self._actor.params, act_acts, key)
            self._actor = self._actor.replace(params=new_p)
            info.update(mets)

        if self._critic_redo is not None and self._critic_redo.should_run():
            crit_acts = _collect_activations_critic(self._critic, obs, actions)
            rng, key = jax.random.split(rng)
            new_p, mets = self._critic_redo.step(self._critic.params, crit_acts, key)
            self._critic = self._critic.replace(params=new_p)
            info.update(mets)

        self._rng = rng
        return info

    def collect_noised_act_stats(self, observations: np.ndarray, actions: np.ndarray) -> Dict:
        """Compute erank/srank of critic noised activations over aggregated UTD data.

        Runs a single critic forward pass on all concatenated mini-batches from
        the UTD loop, then analyses only ``*_noised_act`` intermediates.
        Returns an empty dict when the critic has no noised activations
        (e.g. vd_mode == 'disabled').
        """
        from jaxrl2.utils.redo import _effective_rank, _stable_rank, f32
        noise_key = jax.random.PRNGKey(0)
        _, state = self._critic.apply_fn(
            {'params': self._critic.params},
            observations, actions,
            mutable=['intermediates'],
            rngs={'noise': noise_key},
        )
        intermediates = state.get('intermediates', {})
        metrics = {}

        def _walk(node, prefix):
            if not (isinstance(node, dict) or hasattr(node, 'items')):
                v = node[0] if isinstance(node, tuple) and len(node) == 1 else node
                if prefix.endswith('_noised_act') and hasattr(v, 'ndim') and v.ndim >= 2:
                    lname = prefix.rsplit('/', 1)[-1]
                    # vmap stacks along axis 0 → shape (num_qs, batch, hidden)
                    if v.ndim == 3:
                        for qi in range(v.shape[0]):
                            act_2d = f32(v[qi]).reshape(-1, v[qi].shape[-1])
                            sv = jnp.linalg.svd(act_2d, compute_uv=False)
                            pfx = f'critic_{qi}'
                            metrics[f'{pfx}/redo_noised_agg/erank/{lname}'] = float(_effective_rank(sv))
                            metrics[f'{pfx}/redo_noised_agg/srank/{lname}'] = float(
                                _stable_rank(sv, self._rank_threshold))
                    else:  # ndim == 2: (batch, hidden) — non-vmapped fallback
                        act_2d = f32(v)
                        sv = jnp.linalg.svd(act_2d, compute_uv=False)
                        metrics[f'critic/redo_noised_agg/erank/{lname}'] = float(_effective_rank(sv))
                        metrics[f'critic/redo_noised_agg/srank/{lname}'] = float(
                            _stable_rank(sv, self._rank_threshold))
                return
            for k, v_node in node.items():
                _walk(v_node, f'{prefix}/{k}' if prefix else k)

        _walk(intermediates, '')
        return metrics

    def update(self, batch: FrozenDict) -> Dict[str, float]:
        (
            new_rng,
            new_actor,
            new_critic,
            new_target_critic_params,
            new_temp,
            info,
        ) = _update_jit(
            self._rng,
            self._actor,
            self._critic,
            self._target_critic_params,
            self._temp,
            batch,
            self.discount,
            self.tau,
            self.target_entropy,
            self.backup_entropy,
            self.critic_reduction,
        )

        self._rng = new_rng
        self._actor = new_actor
        self._critic = new_critic
        self._target_critic_params = new_target_critic_params
        self._temp = new_temp

        # --- Activation-based ReDo ---
        # Gate the call so np.asarray is only done when needed.
        act_redo_due = (
            (self._actor_redo  is not None and self._actor_redo.should_run()) or
            (self._critic_redo is not None and self._critic_redo.should_run())
        )
        # Always tick counters even if we skip work.
        if self._actor_redo is not None and not self._actor_redo.should_run():
            self._actor_redo._step += 1
        if self._critic_redo is not None and not self._critic_redo.should_run():
            self._critic_redo._step += 1
        if act_redo_due:
            obs     = np.asarray(batch['observations'])
            actions = np.asarray(batch['actions'])
            info.update(self._apply_act_redo(obs, actions))

        # --- Gradient-based ReDo ---
        # Re-compute gradients only when should_run() is True.
        if self._actor_grad_redo is not None:
            if self._actor_grad_redo.should_run():
                self._rng, key = jax.random.split(self._rng)
                new_p, _, mets = self._actor_grad_redo.step(
                    self._actor.params,
                    _compute_actor_grads(self._actor, self._critic, self._temp, batch),
                    key,
                )
                self._actor = self._actor.replace(params=new_p)
                info.update(mets)
            else:
                self._actor_grad_redo._step += 1  # keep counter in sync

        if self._critic_grad_redo is not None:
            if self._critic_grad_redo.should_run():
                self._rng, key = jax.random.split(self._rng)
                new_p, _, mets = self._critic_grad_redo.step(
                    self._critic.params,
                    _compute_critic_grads(
                        self._actor, self._critic, self._target_critic_params,
                        self._temp, batch, self.discount, self.backup_entropy, self.critic_reduction
                    ),
                    key,
                )
                self._critic = self._critic.replace(params=new_p)
                info.update(mets)
            else:
                self._critic_grad_redo._step += 1  # keep counter in sync

        return info


# ---------------------------------------------------------------------------
# Helpers for activation / gradient collection outside JIT
# ---------------------------------------------------------------------------

def _collect_activations_critic(critic_state: TrainState,
                                 obs: np.ndarray,
                                 actions: np.ndarray) -> Dict:
    """Collect activations from the critic network.

    The critic uses nn.vmap (num_qs=2), so each sow'd tensor has an extra
    leading vmap axis.  We take critic index 0 for dormancy analysis – both
    critics share the same architecture so the result is representative.
    """
    noise_key = jax.random.PRNGKey(0)
    _, state = critic_state.apply_fn(
        {'params': critic_state.params},
        obs, actions,
        mutable=['intermediates'],
        rngs={'noise': noise_key},
    )
    intermediates = state.get('intermediates', {})
    flat = {}
    def _walk(node, prefix):
        if not (isinstance(node, dict) or hasattr(node, 'items')):
            # Sow accumulates as 1-tuples; vmap stacks along axis 0 → shape (num_qs, batch, hidden).
            # Unwrap tuple, then take critic 0.
            v = node[0] if isinstance(node, tuple) and len(node) == 1 else node
            if hasattr(v, 'ndim') and v.ndim >= 2:
                v = v[0]   # take first critic
            flat[prefix] = v
            return
        for k, v in node.items():
            _walk(v, f'{prefix}/{k}' if prefix else k)
    _walk(intermediates, '')
    return flat


def _compute_actor_grads(actor, critic, temp, batch):
    """Re-compute actor gradients for gradient-based ReDo (outside JIT)."""
    from jaxrl2.agents.sac.actor_updater import update_actor
    import jax
    rng = jax.random.PRNGKey(0)
    rng, noise_key = jax.random.split(rng)
    def actor_loss_fn(actor_params):
        dist = actor.apply_fn({'params': actor_params}, batch['observations'])
        actions, log_probs = dist.sample_and_log_prob(seed=rng)
        qs = critic.apply_fn({'params': critic.params}, batch['observations'], actions,
                             rngs={'noise': noise_key})
        q = qs.mean(axis=0)
        return (log_probs * temp.apply_fn({'params': temp.params}) - q).mean(), {}
    grads, _ = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    return grads


def _compute_critic_grads(actor, critic, target_params, temp, batch,
                           discount, backup_entropy, critic_reduction):
    """Re-compute critic gradients for gradient-based ReDo (outside JIT)."""
    import jax
    from jaxrl2.agents.sac.critic_updater import update_critic
    rng = jax.random.PRNGKey(0)
    rng, noise_key1, noise_key2 = jax.random.split(rng, 3)
    target_critic = critic.replace(params=target_params)
    dist = actor.apply_fn({'params': actor.params}, batch['next_observations'])
    next_actions, next_log_probs = dist.sample_and_log_prob(seed=rng)
    next_qs = target_critic.apply_fn({'params': target_params}, batch['next_observations'], next_actions,
                                      rngs={'noise': noise_key1})
    next_q = next_qs.min(axis=0) if critic_reduction == 'min' else next_qs.mean(axis=0)
    target_q = batch['rewards'] + discount * batch['masks'] * next_q
    if backup_entropy:
        target_q -= discount * batch['masks'] * temp.apply_fn({'params': temp.params}) * next_log_probs
    def critic_loss_fn(critic_params):
        qs = critic.apply_fn({'params': critic_params}, batch['observations'], batch['actions'],
                             rngs={'noise': noise_key2})
        return ((qs - target_q) ** 2).mean(), {}
    grads, _ = jax.grad(critic_loss_fn, has_aux=True)(critic.params)
    return grads
