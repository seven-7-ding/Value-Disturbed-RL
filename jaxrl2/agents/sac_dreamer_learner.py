"""SAC agent whose network architecture mirrors DreamerV3.

Network design (matching dreamerv3/configs.yaml defaults):
  - Hidden layers : 3 × units       (configurable via model_size)
  - Activation    : SiLU             (dreamerv3 act: silu)
  - Normalisation : RMSNorm          (dreamerv3 norm: rms)  ← per-layer, after Dense
  - Layer order   : Dense → RMSNorm → SiLU
  - Weight init   : default Xavier uniform (Flax default)
  - Optimiser     : Adam, lr=3e-4 for actor/critic/temp (standard SAC)

Everything else (temperature entropy tuning, ReDo, VD-perturbation, etc.)
is identical to the base SACLearner so the two algorithms can be compared
under the same continual-learning harness.
"""

import copy
import functools
from typing import Dict, Optional, Sequence, Tuple

import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState

from jaxrl2.agents.agent import Agent
from jaxrl2.agents.sac.critic_updater import update_critic
from jaxrl2.agents.sac.temperature import Temperature
from jaxrl2.agents.sac.temperature_updater import update_temperature
from jaxrl2.networks.constants import default_init
from jaxrl2.networks.mlp import MLP, _flatten_dict
from jaxrl2.types import Params, PRNGKey
from jaxrl2.utils.target_update import soft_target_update
from jaxrl2.utils.redo import SACReDo, SACGradientReDo

import distrax


# ---------------------------------------------------------------------------
# Size presets — mirror dreamerv3/configs.yaml (units × 3 layers for policy/value)
# ---------------------------------------------------------------------------

# Each entry: units value from dreamerv3 size config → 3-layer MLP hidden dims.
SAC_SIZES = {
    'size1m':   (64,   64,   64),    # units: 64
    'size12m':  (256,  256,  256),   # units: 256
    'size25m':  (384,  384,  384),   # units: 384
    'size50m':  (512,  512,  512),   # units: 512
    'size100m': (768,  768,  768),   # units: 768
    'size200m': (1024, 1024, 1024),  # units: 1024
    'size400m': (1536, 1536, 1536),  # units: 1536
}


# ---------------------------------------------------------------------------
# SiLU activation shortcut
# ---------------------------------------------------------------------------

silu = nn.silu  # jax.nn.silu is the same; flax.linen.silu also works


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class _SiLUMLP(nn.Module):
    """Plain MLP with SiLU activation and RMSNorm normalisation.

    Mirrors DreamerV3's MLP block: for each hidden layer the order is
        Dense → RMSNorm → SiLU
    matching dreamerv3/configs.yaml  act: silu, norm: rms.
    The final output layer (when activate_final=False) gets no norm or act.
    """
    hidden_dims: Sequence[int]
    activate_final: bool = False
    scale_final: Optional[float] = None
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        x = _flatten_dict(x)
        for i, size in enumerate(self.hidden_dims):
            if i + 1 == len(self.hidden_dims) and self.scale_final is not None:
                x = nn.Dense(size, kernel_init=default_init(self.scale_final),
                             name=f'layer_{i}')(x)
            else:
                x = nn.Dense(size, kernel_init=default_init(), name=f'layer_{i}')(x)

            if i + 1 < len(self.hidden_dims) or self.activate_final:
                # DreamerV3 order: Dense → RMSNorm → SiLU
                x = nn.RMSNorm(name=f'norm_{i}')(x)
                x = silu(x)
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training)
                if self.is_mutable_collection('intermediates'):
                    self.sow('intermediates', f'layer_{i}_act', x)
                    # self.sow('intermediates', f'layer_{i}_noised_act', x)
        return x


class NormalTanhPolicySiLU(nn.Module):
    """Stochastic actor: Gaussian policy with tanh squashing.

    Uses SiLU MLP backbone (no VD perturbation).  Architecture mirrors
    DreamerV3's policy head: layers=3, units=512, act=silu, norm=none.
    """
    hidden_dims: Sequence[int]
    action_dim: int
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    low: Optional[jnp.ndarray] = None
    high: Optional[jnp.ndarray] = None

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 training: bool = False) -> distrax.Distribution:
        outputs = _SiLUMLP(
            self.hidden_dims,
            activate_final=True,
        )(observations, training=training)

        means = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)
        log_stds = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        # Build tanh-squashed Gaussian (optionally rescaled to [low, high]).
        distribution = distrax.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds))

        layers = []
        if self.low is not None and self.high is not None:
            low, high = self.low, self.high

            def rescale(x):
                return (x + 1) / 2 * (high - low) + low

            def fldj(x):
                h = jnp.broadcast_to(high, x.shape)
                l = jnp.broadcast_to(low,  x.shape)
                return jnp.sum(jnp.log(0.5 * (h - l)), -1)

            layers.append(distrax.Lambda(
                rescale,
                forward_log_det_jacobian=fldj,
                event_ndims_in=1, event_ndims_out=1,
            ))
        layers.append(distrax.Block(distrax.Tanh(), 1))
        bijector = distrax.Chain(layers)
        return distrax.Transformed(distribution=distribution, bijector=bijector)


class _StateActionValueSiLU(nn.Module):
    """Single Q-function: (obs, act) → scalar.  SiLU, no norm."""
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray,
                 training: bool = False) -> jnp.ndarray:
        inputs = {"states": observations, "actions": actions}
        x = _SiLUMLP((*self.hidden_dims, 1))(inputs, training=training)
        return jnp.squeeze(x, -1)


class StateActionEnsembleSiLU(nn.Module):
    """Double Q-function ensemble with SiLU activations."""
    hidden_dims: Sequence[int]
    num_qs: int = 2

    @nn.compact
    def __call__(self, states, actions, training: bool = False):
        VmapCritic = nn.vmap(
            _StateActionValueSiLU,
            variable_axes={"params": 0, "intermediates": 0},
            split_rngs={"params": True, "noise": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num_qs,
        )
        return VmapCritic(self.hidden_dims)(states, actions, training)


# ---------------------------------------------------------------------------
# JIT-compiled update step (mirrors sac_learner._update_jit)
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=("backup_entropy", "critic_reduction"))
def _update_jit_dreamer(
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
) -> Tuple[PRNGKey, TrainState, TrainState, Params, TrainState, Dict]:
    rng, key = jax.random.split(rng)
    target_critic = critic.replace(params=target_critic_params)
    new_critic, critic_info = update_critic(
        key, actor, critic, target_critic, temp, batch,
        discount, backup_entropy=backup_entropy, critic_reduction=critic_reduction,
    )
    new_target_critic_params = soft_target_update(
        new_critic.params, target_critic_params, tau)

    rng, key = jax.random.split(rng)
    # Actor update (inline; avoids importing SAC's actor_updater which may use
    # RN perturbations).
    rng, noise_key = jax.random.split(rng)

    def actor_loss_fn(actor_params):
        dist = actor.apply_fn({"params": actor_params}, batch["observations"])
        actions, log_probs = dist.sample_and_log_prob(seed=key)
        qs = new_critic.apply_fn(
            {"params": new_critic.params}, batch["observations"], actions,
            rngs={"noise": noise_key})
        q = qs.mean(axis=0)
        loss = (log_probs * temp.apply_fn({"params": temp.params}) - q).mean()
        return loss, {"actor_loss": loss, "entropy": -log_probs.mean()}

    grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    new_actor = actor.apply_gradients(grads=grads)

    new_temp, alpha_info = update_temperature(
        temp, actor_info["entropy"], target_entropy)

    return (
        rng, new_actor, new_critic, new_target_critic_params, new_temp,
        {**critic_info, **actor_info, **alpha_info},
    )


# ---------------------------------------------------------------------------
# SACDreamerLearner
# ---------------------------------------------------------------------------

class SACDreamerLearner(Agent):
    """SAC agent with DreamerV3-aligned networks.

    Network spec (default size1m):
        hidden_dims = (64, 64, 64)      # 3 layers × 64 units
        activation  = SiLU
        norm        = RMSNorm (per hidden layer, after Dense, before SiLU)

    All other SAC hyperparameters (τ, γ, entropy target, ReDo, …) are
    identical to SACLearner so comparison is fair.
    """

    def __init__(
        self,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        # DreamerV3 size1m default: 3 layers × 64 units
        hidden_dims: Sequence[int] = (64, 64, 64),
        # Pass e.g. 'size1m' / 'size12m' / 'size50m' to override hidden_dims.
        model_size: Optional[str] = None,
        discount: float = 0.99,
        tau: float = 0.005,
        target_entropy: Optional[float] = None,
        backup_entropy: bool = True,
        critic_reduction: str = "min",
        init_temperature: float = 1.0,
        vd_mode: str = "disabled",
        redo: Optional[Dict] = None,
    ):
        action_dim = action_space.shape[-1]

        # Resolve model_size → hidden_dims (model_size takes priority).
        if model_size is not None:
            if model_size not in SAC_SIZES:
                raise ValueError(
                    f"Unknown model_size '{model_size}'. "
                    f"Valid options: {list(SAC_SIZES.keys())}")
            hidden_dims = SAC_SIZES[model_size]

        self.target_entropy = (
            -action_dim / 2 if target_entropy is None else target_entropy)
        self.backup_entropy = backup_entropy
        self.critic_reduction = critic_reduction
        self.tau = tau
        self.discount = discount

        observations = observation_space.sample()
        actions = action_space.sample()

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, critic_noise_key, temp_key = (
            jax.random.split(rng, 5))

        # Bounded action rescaling (if needed).
        if np.all(action_space.low == -1) and np.all(action_space.high == 1):
            low = high = None
        else:
            low = action_space.low
            high = action_space.high

        # ---- networks ----
        actor_def = NormalTanhPolicySiLU(hidden_dims, action_dim, low=low, high=high)
        actor_params = actor_def.init(actor_key, observations)["params"]
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=actor_lr),
        )

        critic_def = StateActionEnsembleSiLU(hidden_dims, num_qs=2)
        critic_params = critic_def.init(
            {"params": critic_key, "noise": critic_noise_key},
            observations, actions,
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

        # Stash for reset_agent().
        self._actor_def = actor_def
        self._critic_def = critic_def
        self._temp_def = temp_def
        self._actor_lr = actor_lr
        self._critic_lr = critic_lr
        self._temp_lr = temp_lr
        self._init_temperature = init_temperature
        self._observations_sample = observations
        self._actions_sample = actions

        # ---- ReDo (same logic as SACLearner) ----
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
        skip = redo.get('skip_last_layer', False)
        self._actor_redo = SACReDo(name='actor', **redo_kw, skip_last_layer=skip) \
            if redo.get('redo_enabled', False) else None
        self._critic_redo = SACReDo(name='critic', **redo_kw, skip_last_layer=skip) \
            if redo.get('redo_enabled', False) else None
        grad_kw = {k: v for k, v in redo_kw.items() if k != 'rank_threshold'}
        self._actor_grad_redo = SACGradientReDo(name='actor', **grad_kw) \
            if redo.get('grad_redo_enabled', False) else None
        self._critic_grad_redo = SACGradientReDo(name='critic', **grad_kw) \
            if redo.get('grad_redo_enabled', False) else None

    # ------------------------------------------------------------------
    # Agent reset
    # ------------------------------------------------------------------

    def reset_agent(self) -> None:
        """Reinitialise all network parameters (called on 'reset_all' vd_mode)."""
        self._rng, actor_key, critic_key, critic_noise_key, temp_key = \
            jax.random.split(self._rng, 5)

        actor_params = self._actor_def.init(
            actor_key, self._observations_sample)["params"]
        self._actor = self._actor.replace(
            params=actor_params,
            opt_state=optax.adam(learning_rate=self._actor_lr).init(actor_params),
            step=0,
        )

        critic_params = self._critic_def.init(
            {"params": critic_key, "noise": critic_noise_key},
            self._observations_sample, self._actions_sample,
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
        for obj in (self._actor_redo, self._critic_redo,
                    self._actor_grad_redo, self._critic_grad_redo):
            if obj is not None:
                obj._step = 0

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def eval_actions(self, observations: np.ndarray) -> np.ndarray:
        """Return deterministic actions: tanh(normal_mean).

        Overrides Agent.eval_actions() because distrax.Transformed.mode() is
        not implemented for bijectors with non-constant Jacobian (Tanh).
        """
        actions = _eval_actions_tanh_mean_jit(self._actor, observations)
        return np.asarray(actions)

    def sample_actions(self, observations: np.ndarray) -> np.ndarray:
        rng, key = jax.random.split(self._rng)
        self._rng = rng
        actions = _sample_actions_jit(key, self._actor, observations)
        return np.asarray(actions)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def update(self, batch: FrozenDict) -> Dict[str, float]:
        (
            new_rng,
            new_actor,
            new_critic,
            new_target_critic_params,
            new_temp,
            info,
        ) = _update_jit_dreamer(
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

        # ---- ReDo analysis ----
        redo_due = (
            (self._actor_redo is not None and self._actor_redo.should_run()) or
            (self._critic_redo is not None and self._critic_redo.should_run()) or
            (self._actor_grad_redo is not None and self._actor_grad_redo.should_run()) or
            (self._critic_grad_redo is not None and self._critic_grad_redo.should_run())
        )
        if redo_due:
            obs     = np.asarray(batch['observations'])
            actions = np.asarray(batch['actions'])
            info.update(self._apply_act_redo(obs, actions))
            info.update(self._apply_grad_redo(batch))
        else:
            for obj in (self._actor_redo, self._critic_redo,
                        self._actor_grad_redo, self._critic_grad_redo):
                if obj is not None:
                    obj._step += 1

        return info

    # ------------------------------------------------------------------
    # Noised activation stats (mirrors SACLearner)
    # ------------------------------------------------------------------

    def collect_noised_act_stats(self, obs: np.ndarray,
                                 actions: np.ndarray) -> Dict:
        """Compute activation rank stats — called by train_continual.py."""
        return {}

    # ------------------------------------------------------------------
    # Internal ReDo helpers (copied from SACLearner pattern)
    # ------------------------------------------------------------------

    def _collect_actor_acts(self, obs: np.ndarray) -> Dict:
        _, state = self._actor.apply_fn(
            {'params': self._actor.params},
            obs,
            mutable=['intermediates'],
        )
        intermediates = state.get('intermediates', {})
        flat = {}
        def _walk(node, prefix):
            if not (isinstance(node, dict) or hasattr(node, 'items')):
                flat[prefix] = node[0] if isinstance(node, tuple) else node
                return
            for k, v in node.items():
                _walk(v, f'{prefix}/{k}' if prefix else k)
        _walk(intermediates, '')
        return flat

    def _collect_critic_acts(self, obs: np.ndarray,
                              actions: np.ndarray) -> Dict:
        noise_key = jax.random.PRNGKey(0)
        _, state = self._critic.apply_fn(
            {'params': self._critic.params},
            obs, actions,
            mutable=['intermediates'],
            rngs={'noise': noise_key},
        )
        intermediates = state.get('intermediates', {})
        flat = {}
        def _walk(node, prefix):
            if not (isinstance(node, dict) or hasattr(node, 'items')):
                v = node[0] if isinstance(node, tuple) and len(node) == 1 else node
                if hasattr(v, 'ndim') and v.ndim >= 2:
                    v = v[0]
                flat[prefix] = v
                return
            for k, v in node.items():
                _walk(v, f'{prefix}/{k}' if prefix else k)
        _walk(intermediates, '')
        return flat

    def _apply_act_redo(self, obs: np.ndarray,
                         actions: np.ndarray) -> Dict:
        info = {}
        if self._actor_redo is not None and self._actor_redo.should_run():
            self._rng, key = jax.random.split(self._rng)
            acts = self._collect_actor_acts(obs)
            new_p, mets = self._actor_redo.step(
                self._actor.params, acts, key)
            self._actor = self._actor.replace(params=new_p)
            info.update(mets)
        elif self._actor_redo is not None:
            self._actor_redo._step += 1

        if self._critic_redo is not None and self._critic_redo.should_run():
            self._rng, key = jax.random.split(self._rng)
            acts = self._collect_critic_acts(obs, actions)
            new_p, mets = self._critic_redo.step(
                self._critic.params, acts, key)
            self._critic = self._critic.replace(params=new_p)
            info.update(mets)
        elif self._critic_redo is not None:
            self._critic_redo._step += 1

        return info

    def _apply_grad_redo(self, batch: FrozenDict) -> Dict:
        info = {}
        if self._actor_grad_redo is not None and self._actor_grad_redo.should_run():
            self._rng, key = jax.random.split(self._rng)
            grads = _compute_actor_grads_dreamer(
                self._actor, self._critic, self._temp, batch)
            new_p, _, mets = self._actor_grad_redo.step(
                self._actor.params, grads, key)
            self._actor = self._actor.replace(params=new_p)
            info.update(mets)
        elif self._actor_grad_redo is not None:
            self._actor_grad_redo._step += 1

        if self._critic_grad_redo is not None and self._critic_grad_redo.should_run():
            self._rng, key = jax.random.split(self._rng)
            grads = _compute_critic_grads_dreamer(
                self._actor, self._critic, self._target_critic_params,
                self._temp, batch, self.discount, self.backup_entropy,
                self.critic_reduction)
            new_p, _, mets = self._critic_grad_redo.step(
                self._critic.params, grads, key)
            self._critic = self._critic.replace(params=new_p)
            info.update(mets)
        elif self._critic_grad_redo is not None:
            self._critic_grad_redo._step += 1

        return info


# ---------------------------------------------------------------------------
# JIT helpers
# ---------------------------------------------------------------------------

@jax.jit
def _sample_actions_jit(key: PRNGKey, actor: TrainState,
                         observations: np.ndarray) -> jnp.ndarray:
    dist = actor.apply_fn({"params": actor.params}, observations)
    return dist.sample(seed=key)


@jax.jit
def _eval_actions_tanh_mean_jit(actor: TrainState,
                                  observations: np.ndarray) -> jnp.ndarray:
    """Deterministic action for evaluation.

    distrax.Transformed does not implement mode() for non-constant-Jacobian
    bijectors (Tanh).  The correct deterministic action for a TanhNormal policy
    is bijector.forward(normal.mean()), i.e. tanh of the Gaussian mean.
    """
    dist = actor.apply_fn({"params": actor.params}, observations)
    return dist.bijector.forward(dist.distribution.mean())


def _compute_actor_grads_dreamer(actor, critic, temp, batch):
    rng = jax.random.PRNGKey(0)
    rng, noise_key = jax.random.split(rng)

    def loss_fn(actor_params):
        dist = actor.apply_fn({'params': actor_params}, batch['observations'])
        actions, log_probs = dist.sample_and_log_prob(seed=rng)
        qs = critic.apply_fn(
            {'params': critic.params}, batch['observations'], actions,
            rngs={'noise': noise_key})
        q = qs.mean(axis=0)
        return (log_probs * temp.apply_fn({'params': temp.params}) - q).mean(), {}

    grads, _ = jax.grad(loss_fn, has_aux=True)(actor.params)
    return grads


def _compute_critic_grads_dreamer(actor, critic, target_params, temp, batch,
                                   discount, backup_entropy, critic_reduction):
    rng = jax.random.PRNGKey(0)
    rng, noise_key1, noise_key2 = jax.random.split(rng, 3)
    target_critic = critic.replace(params=target_params)
    dist = actor.apply_fn({'params': actor.params}, batch['next_observations'])
    next_actions, next_log_probs = dist.sample_and_log_prob(seed=rng)
    next_qs = target_critic.apply_fn(
        {'params': target_params}, batch['next_observations'], next_actions,
        rngs={'noise': noise_key1})
    next_q = (next_qs.min(axis=0) if critic_reduction == 'min'
               else next_qs.mean(axis=0))
    target_q = batch['rewards'] + discount * batch['masks'] * next_q
    if backup_entropy:
        target_q -= (discount * batch['masks'] *
                     temp.apply_fn({'params': temp.params}) * next_log_probs)

    def critic_loss_fn(critic_params):
        qs = critic.apply_fn(
            {'params': critic_params}, batch['observations'], batch['actions'],
            rngs={'noise': noise_key2})
        return ((qs - target_q) ** 2).mean(), {}

    grads, _ = jax.grad(critic_loss_fn, has_aux=True)(critic.params)
    return grads
