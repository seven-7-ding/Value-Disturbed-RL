"""SAC agent whose network architecture mirrors DreamerV3.

Network design (matching dreamerv3/configs.yaml defaults):
  - Hidden layers : 3 × units       (configurable via model_size)
  - Activation    : SiLU             (dreamerv3 act: silu)
  - Normalisation : RMSNorm          (dreamerv3 norm: rms)  ← per-layer, after Dense
  - Layer order   : Dense → RMSNorm → SiLU
  - Weight init   : default Xavier uniform (Flax default)
  - Optimiser     : DreamerV3-aligned (AGC→RMS→momentum→warmup-LR)

Everything else (temperature entropy tuning, ReDo, VD-perturbation, etc.)
is identical to the base SACLearner so the two algorithms can be compared
under the same continual-learning harness.
"""

import copy
import functools
import re as _re
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
# DreamerV3-aligned optimizer
# Mirrors dreamerv3/agent.py _make_opt() + embodied/jax/opt.py primitives.
# Chain: AGC → RMS-scaling → momentum → lr-schedule (with warmup).
# ---------------------------------------------------------------------------

def _clip_by_agc(clip: float = 0.3, pmin: float = 1e-3) -> optax.GradientTransformation:
    """Adaptive gradient clipping (Brock et al., 2021)."""
    def init_fn(params):
        return ()

    def update_fn(updates, state, params=None):
        def fn(param, update):
            unorm = jnp.linalg.norm(update.flatten(), 2)
            pnorm = jnp.linalg.norm(param.flatten(), 2)
            upper = clip * jnp.maximum(pmin, pnorm)
            return update * (1 / jnp.maximum(1.0, unorm / upper))
        updates = jax.tree_util.tree_map(fn, params, updates) if clip else updates
        return updates, ()

    return optax.GradientTransformation(init_fn, update_fn)


def _scale_by_rms(
    beta: float = 0.999,
    eps: float = 1e-20,
) -> optax.GradientTransformation:
    """Adam-style second-moment scaling with bias correction (no sqrt(1-β²) denom)."""
    def init_fn(params):
        nu = jax.tree_util.tree_map(lambda t: jnp.zeros_like(t, jnp.float32), params)
        step = jnp.zeros((), jnp.int32)
        return (step, nu)

    def update_fn(updates, state, params=None):
        step, nu = state
        step = optax.safe_int32_increment(step)
        nu = jax.tree_util.tree_map(
            lambda v, u: beta * v + (1 - beta) * (u * u), nu, updates)
        nu_hat = optax.bias_correction(nu, beta, step)
        updates = jax.tree_util.tree_map(
            lambda u, v: u / (jnp.sqrt(v) + eps), updates, nu_hat)
        return updates, (step, nu)

    return optax.GradientTransformation(init_fn, update_fn)


def _scale_by_momentum(
    beta: float = 0.9,
    nesterov: bool = False,
) -> optax.GradientTransformation:
    """Bias-corrected first-moment (momentum) transform."""
    def init_fn(params):
        mu = jax.tree_util.tree_map(lambda t: jnp.zeros_like(t), params)
        step = jnp.zeros((), jnp.int32)
        return (step, mu)

    def update_fn(updates, state, params=None):
        step, mu = state
        step = optax.safe_int32_increment(step)
        mu = optax.update_moment(updates, mu, beta, 1)
        if nesterov:
            mu2 = optax.update_moment(updates, mu, beta, 1)
            mu_hat = optax.bias_correction(mu2, beta, step)
        else:
            mu_hat = optax.bias_correction(mu, beta, step)
        return mu_hat, (step, mu)

    return optax.GradientTransformation(init_fn, update_fn)


def _make_dreamer_opt(
    lr: float,
    agc: float = 0.3,
    eps: float = 1e-20,
    beta1: float = 0.9,
    beta2: float = 0.999,
    momentum: bool = True,
    nesterov: bool = False,
    wd: float = 0.0,
    wdregex: str = r'/kernel$',
    schedule: str = 'const',
    warmup: int = 1000,
    anneal: int = 0,
) -> optax.GradientTransformation:
    """Reconstruct DreamerV3's optimizer chain.

    Exactly mirrors dreamerv3/agent.py _make_opt() signature and logic.
    Chain: AGC → RMS-scale → (optional) momentum → (optional) WD → lr-schedule.

    Args:
        lr       : Peak learning rate.
        agc      : AGC clip ratio (0 = disabled).  DreamerV3 default: 0.3.
        eps      : RMS denominator epsilon.         DreamerV3 default: 1e-20.
        beta1    : Momentum decay.                  DreamerV3 default: 0.9.
        beta2    : RMS decay.                       DreamerV3 default: 0.999.
        momentum : Whether to apply first-moment.  DreamerV3 default: True.
        nesterov : Nesterov momentum.               DreamerV3 default: False.
        wd       : Weight-decay coefficient.        DreamerV3 default: 0.0.
        wdregex  : Regex matching params to decay.  DreamerV3 default: '/kernel$'.
        schedule : LR schedule ('const'/'linear'/'cosine').
        warmup   : Linear warmup steps.             DreamerV3 default: 1000.
        anneal   : Total steps for non-const schedule (0 = unused).
    """
    chain = []
    chain.append(_clip_by_agc(agc))
    chain.append(_scale_by_rms(beta2, eps))
    if momentum:
        chain.append(_scale_by_momentum(beta1, nesterov))
    if wd:
        pattern = _re.compile(wdregex)
        def _wd_mask(params):
            """Boolean pytree: True for params whose path matches wdregex."""
            try:
                return jax.tree_util.tree_map_with_path(
                    lambda path, _: bool(pattern.search(
                        '/' + '/'.join(
                            getattr(p, 'key', str(p)) for p in path))),
                    params)
            except AttributeError:
                # Fallback for older JAX: apply decay to all leaves.
                return jax.tree_util.tree_map(lambda _: True, params)
        chain.append(optax.add_decayed_weights(wd, _wd_mask))
    assert anneal > 0 or schedule == 'const', \
        f"anneal must be > 0 for schedule='{schedule}'"
    if schedule == 'const':
        sched = optax.constant_schedule(lr)
    elif schedule == 'linear':
        sched = optax.linear_schedule(lr, 0.1 * lr, anneal - warmup)
    elif schedule == 'cosine':
        sched = optax.cosine_decay_schedule(lr, anneal - warmup, 0.1 * lr)
    else:
        raise NotImplementedError(f"Unknown schedule: {schedule!r}")
    if warmup:
        ramp = optax.linear_schedule(0.0, lr, warmup)
        sched = optax.join_schedules([ramp, sched], [warmup])
    chain.append(optax.scale_by_learning_rate(sched))
    return optax.chain(*chain)


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
        opt: Optional[Dict] = None,
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

        # ---- optimizer hyperparams (mirrors dreamerv3/configs.yaml opt block) ----
        opt = opt or {}
        opt_kwargs = dict(
            agc      = opt.get('agc',      0.3),
            eps      = opt.get('eps',      1e-20),
            beta1    = opt.get('beta1',    0.9),
            beta2    = opt.get('beta2',    0.999),
            momentum = opt.get('momentum', True),
            nesterov = opt.get('nesterov', False),
            wd       = opt.get('wd',       0.0),
            wdregex  = opt.get('wdregex',  r'/kernel$'),
            schedule = opt.get('schedule', 'const'),
            warmup   = opt.get('warmup',   1000),
            anneal   = opt.get('anneal',   0),
        )
        self._opt_kwargs = opt_kwargs

        # ---- networks ----
        actor_def = NormalTanhPolicySiLU(hidden_dims, action_dim, low=low, high=high)
        actor_params = actor_def.init(actor_key, observations)["params"]
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=_make_dreamer_opt(actor_lr, **opt_kwargs),
        )

        critic_def = StateActionEnsembleSiLU(hidden_dims, num_qs=2)
        critic_params = critic_def.init(
            {"params": critic_key, "noise": critic_noise_key},
            observations, actions,
        )["params"]
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=_make_dreamer_opt(critic_lr, **opt_kwargs),
        )
        target_critic_params = copy.deepcopy(critic_params)

        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)["params"]
        temp = TrainState.create(
            apply_fn=temp_def.apply,
            params=temp_params,
            tx=_make_dreamer_opt(temp_lr, **opt_kwargs),
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
            opt_state=_make_dreamer_opt(self._actor_lr, **self._opt_kwargs).init(actor_params),
            step=0,
        )

        critic_params = self._critic_def.init(
            {"params": critic_key, "noise": critic_noise_key},
            self._observations_sample, self._actions_sample,
        )["params"]
        self._critic = self._critic.replace(
            params=critic_params,
            opt_state=_make_dreamer_opt(self._critic_lr, **self._opt_kwargs).init(critic_params),
            step=0,
        )
        self._target_critic_params = copy.deepcopy(critic_params)

        temp_params = self._temp_def.init(temp_key)["params"]
        self._temp = self._temp.replace(
            params=temp_params,
            opt_state=_make_dreamer_opt(self._temp_lr, **self._opt_kwargs).init(temp_params),
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


# ===========================================================================
# TwoHot distributional critic — mirrors DreamerV3 value head (symexp_twohot)
# ===========================================================================

def _symlog(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def _symexp(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sign(x) * jnp.expm1(jnp.abs(x))


def _make_twohot_bins(num_bins: int) -> jnp.ndarray:
    """Compute symexp-spaced bins identical to DreamerV3 symexp_twohot."""
    if num_bins % 2 == 1:
        half = jnp.linspace(-20.0, 0.0, (num_bins - 1) // 2 + 1, dtype=jnp.float32)
        half = _symexp(half)
        bins = jnp.concatenate([half, -half[:-1][::-1]], axis=0)
    else:
        half = jnp.linspace(-20.0, 0.0, num_bins // 2, dtype=jnp.float32)
        half = _symexp(half)
        bins = jnp.concatenate([half, -half[::-1]], axis=0)
    return bins


def _twohot_pred(logits: jnp.ndarray, bins: jnp.ndarray) -> jnp.ndarray:
    """Expected value from TwoHot logits.  Mirrors TwoHot.pred() in outs.py.

    Uses a symmetric summation to cancel floating-point bias at init (when
    logits are 0 and probabilities are uniform).

    Args:
        logits: [..., num_bins]
        bins:   [num_bins]
    Returns:
        Expected value [...].
    """
    probs = jax.nn.softmax(logits, axis=-1)
    n = bins.shape[0]
    if n % 2 == 1:
        m = (n - 1) // 2
        p1, p2, p3 = probs[..., :m], probs[..., m:m + 1], probs[..., m + 1:]
        b1, b2, b3 = bins[:m], bins[m:m + 1], bins[m + 1:]
        return (p2 * b2).sum(-1) + ((p1 * b1)[..., ::-1] + (p3 * b3)).sum(-1)
    else:
        h = n // 2
        p1, p2 = probs[..., :h], probs[..., h:]
        b1, b2 = bins[:h], bins[h:]
        return ((p1 * b1)[..., ::-1] + (p2 * b2)).sum(-1)


def _twohot_loss(logits: jnp.ndarray, target: jnp.ndarray,
                 bins: jnp.ndarray) -> jnp.ndarray:
    """Cross-entropy loss with two-hot encoded target.  Mirrors TwoHot.loss().

    Args:
        logits: [..., num_bins]
        target: [...] scalar targets
        bins:   [num_bins]
    Returns:
        Per-element loss [...].
    """
    target = jax.lax.stop_gradient(target.astype(jnp.float32))
    nb = bins.shape[0]
    below = (bins <= target[..., None]).sum(-1).astype(jnp.int32) - 1
    above = nb - (bins > target[..., None]).sum(-1).astype(jnp.int32)
    below = jnp.clip(below, 0, nb - 1)
    above = jnp.clip(above, 0, nb - 1)
    equal = (below == above)
    dist_to_below = jnp.where(equal, 1.0, jnp.abs(bins[below] - target))
    dist_to_above = jnp.where(equal, 1.0, jnp.abs(bins[above] - target))
    total = dist_to_below + dist_to_above
    weight_below = dist_to_above / total
    weight_above = dist_to_below / total
    target_twohot = (
        jax.nn.one_hot(below, nb) * weight_below[..., None] +
        jax.nn.one_hot(above, nb) * weight_above[..., None])
    log_pred = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    return -(target_twohot * log_pred).sum(-1)


# ---------------------------------------------------------------------------
# TwoHot critic networks
# ---------------------------------------------------------------------------

class _StateActionTwoHotSiLU(nn.Module):
    """Single Q-function with TwoHot distributional output.

    (obs, act) → logits [..., num_bins]
    Mirrors DreamerV3 value head: 3-layer SiLU+RMSNorm MLP → Linear(num_bins).
    Output layer is zero-initialised (outscale=0.0 in DreamerV3), giving
    uniform logits at init so that pred() = 0.
    """
    hidden_dims: Sequence[int]
    num_bins: int = 255

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray,
                 training: bool = False) -> jnp.ndarray:
        inputs = {"states": observations, "actions": actions}
        x = _SiLUMLP(self.hidden_dims, activate_final=True)(inputs, training=training)
        logits = nn.Dense(
            self.num_bins,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name='logits',
        )(x)
        return logits


class StateActionTwoHotEnsembleSiLU(nn.Module):
    """Double Q-function ensemble with TwoHot distributional output.

    Identical vmap structure to StateActionEnsembleSiLU but each member
    outputs logits of shape [..., num_bins] instead of scalars.
    The full ensemble output shape is [num_qs, batch, num_bins].
    """
    hidden_dims: Sequence[int]
    num_qs: int = 2
    num_bins: int = 255

    @nn.compact
    def __call__(self, states, actions, training: bool = False):
        VmapCritic = nn.vmap(
            _StateActionTwoHotSiLU,
            variable_axes={"params": 0, "intermediates": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num_qs,
        )
        return VmapCritic(self.hidden_dims, self.num_bins)(states, actions, training)


# ---------------------------------------------------------------------------
# JIT-compiled update step for distributional SAC
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=(
    "backup_entropy", "critic_reduction", "num_bins"))
def _update_jit_dreamer_dist(
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
    num_bins: int,
) -> Tuple[PRNGKey, TrainState, TrainState, Params, TrainState, Dict]:
    """SAC update with TwoHot distributional critic (mirrors DreamerV3 value).

    Critic loss  : TwoHot cross-entropy  (not MSE)
    Actor Q-value: _twohot_pred() expected value from TwoHot distribution
    Everything else (temperature, Bellman target formula, soft update) is
    identical to the standard SAC / SACDreamerLearner update.
    """
    bins = _make_twohot_bins(num_bins)

    # ---- Bellman target ----
    rng, key = jax.random.split(rng)
    dist = actor.apply_fn({"params": actor.params}, batch["next_observations"])
    next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)

    target_critic = critic.replace(params=target_critic_params)
    next_q_logits = target_critic.apply_fn(
        {"params": target_critic.params},
        batch["next_observations"], next_actions)
    # next_q_logits: [num_qs, batch, num_bins]
    next_qs = _twohot_pred(next_q_logits, bins)  # [num_qs, batch]

    if critic_reduction == "min":
        next_q = next_qs.min(axis=0)
    elif critic_reduction == "mean":
        next_q = next_qs.mean(axis=0)
    else:
        next_q = next_qs.min(axis=0)  # default to min

    target_q = batch["rewards"] + discount * batch["masks"] * next_q
    if backup_entropy:
        target_q -= (
            discount * batch["masks"]
            * temp.apply_fn({"params": temp.params})
            * next_log_probs)

    # ---- Critic update: TwoHot cross-entropy ----
    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, Dict]:
        q_logits = critic.apply_fn(
            {"params": critic_params},
            batch["observations"], batch["actions"])
        # q_logits: [num_qs, batch, num_bins]
        loss = _twohot_loss(q_logits, target_q, bins)  # [num_qs, batch]
        critic_loss = loss.mean()
        return critic_loss, {
            "critic_loss": critic_loss,
            "q": _twohot_pred(q_logits, bins).mean(),
            "target_actor_entropy": -next_log_probs.mean(),
        }

    grads, critic_info = jax.grad(critic_loss_fn, has_aux=True)(critic.params)
    new_critic = critic.apply_gradients(grads=grads)
    new_target_critic_params = soft_target_update(
        new_critic.params, target_critic_params, tau)

    # ---- Actor update: use TwoHot pred() for Q values ----
    rng, key = jax.random.split(rng)

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, Dict]:
        dist = actor.apply_fn({"params": actor_params}, batch["observations"])
        actions, log_probs = dist.sample_and_log_prob(seed=key)
        q_logits = new_critic.apply_fn(
            {"params": new_critic.params},
            batch["observations"], actions)
        # Use TwoHot pred() — expected Q, not raw logits
        qs = _twohot_pred(q_logits, bins)   # [num_qs, batch]
        q = qs.mean(axis=0)
        loss = (log_probs * temp.apply_fn({"params": temp.params}) - q).mean()
        return loss, {"actor_loss": loss, "entropy": -log_probs.mean()}

    grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    new_actor = actor.apply_gradients(grads=grads)

    new_temp, alpha_info = update_temperature(
        temp, actor_info["entropy"], target_entropy)

    return (
        rng, new_actor, new_critic, new_target_critic_params, new_temp,
        {**critic_info, **actor_info, **alpha_info})


# ---------------------------------------------------------------------------
# SACDreamerDistLearner
# ---------------------------------------------------------------------------

class SACDreamerDistLearner(Agent):
    """SAC with DreamerV3-aligned networks **and** TwoHot distributional critic.

    Architecture differences vs SACDreamerLearner:
      - Critic  : outputs logits for ``num_bins`` bins (symexp-spaced ±20)
                  instead of a scalar Q value.
      - Critic loss : TwoHot cross-entropy (mirrors DreamerV3 value loss).
      - Actor Q     : _twohot_pred() expected value, not raw critic output.

    Everything else (temperature, target networks, ReDo, reset_agent, …)
    is identical to SACDreamerLearner.
    """

    def __init__(
        self,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (64, 64, 64),
        model_size: Optional[str] = None,
        num_bins: int = 255,
        discount: float = 0.99,
        tau: float = 0.005,
        target_entropy: Optional[float] = None,
        backup_entropy: bool = True,
        critic_reduction: str = "min",
        init_temperature: float = 1.0,
        vd_mode: str = "disabled",
        redo: Optional[Dict] = None,
        opt: Optional[Dict] = None,
    ):
        action_dim = action_space.shape[-1]

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
        self.num_bins = num_bins

        observations = observation_space.sample()
        actions = action_space.sample()

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        if np.all(action_space.low == -1) and np.all(action_space.high == 1):
            low = high = None
        else:
            low = action_space.low
            high = action_space.high

        # ---- optimizer hyperparams (mirrors dreamerv3/configs.yaml opt block) ----
        opt = opt or {}
        opt_kwargs = dict(
            agc      = opt.get('agc',      0.3),
            eps      = opt.get('eps',      1e-20),
            beta1    = opt.get('beta1',    0.9),
            beta2    = opt.get('beta2',    0.999),
            momentum = opt.get('momentum', True),
            nesterov = opt.get('nesterov', False),
            wd       = opt.get('wd',       0.0),
            wdregex  = opt.get('wdregex',  r'/kernel$'),
            schedule = opt.get('schedule', 'const'),
            warmup   = opt.get('warmup',   1000),
            anneal   = opt.get('anneal',   0),
        )
        self._opt_kwargs = opt_kwargs

        # Actor: same NormalTanhPolicySiLU (mean+std, tanh-squashed)
        actor_def = NormalTanhPolicySiLU(hidden_dims, action_dim, low=low, high=high)
        actor_params = actor_def.init(actor_key, observations)["params"]
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=_make_dreamer_opt(actor_lr, **opt_kwargs),
        )

        # Critic: TwoHot distributional (logits → num_bins)
        critic_def = StateActionTwoHotEnsembleSiLU(
            hidden_dims, num_qs=2, num_bins=num_bins)
        critic_params = critic_def.init(
            {"params": critic_key},
            observations, actions,
        )["params"]
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=_make_dreamer_opt(critic_lr, **opt_kwargs),
        )
        target_critic_params = copy.deepcopy(critic_params)

        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)["params"]
        temp = TrainState.create(
            apply_fn=temp_def.apply,
            params=temp_params,
            tx=_make_dreamer_opt(temp_lr, **opt_kwargs),
        )

        self._actor = actor
        self._critic = critic
        self._target_critic_params = target_critic_params
        self._temp = temp
        self._rng = rng
        self._vd_mode = vd_mode

        # Stash for reset_agent()
        self._actor_def = actor_def
        self._critic_def = critic_def
        self._temp_def = temp_def
        self._actor_lr = actor_lr
        self._critic_lr = critic_lr
        self._temp_lr = temp_lr
        self._init_temperature = init_temperature
        self._observations_sample = observations
        self._actions_sample = actions

        # ReDo
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
    # Reset
    # ------------------------------------------------------------------

    def reset_agent(self) -> None:
        self._rng, actor_key, critic_key, temp_key = jax.random.split(self._rng, 4)

        actor_params = self._actor_def.init(
            actor_key, self._observations_sample)["params"]
        self._actor = self._actor.replace(
            params=actor_params,
            opt_state=_make_dreamer_opt(self._actor_lr, **self._opt_kwargs).init(actor_params),
            step=0,
        )

        critic_params = self._critic_def.init(
            {"params": critic_key},
            self._observations_sample, self._actions_sample,
        )["params"]
        self._critic = self._critic.replace(
            params=critic_params,
            opt_state=_make_dreamer_opt(self._critic_lr, **self._opt_kwargs).init(critic_params),
            step=0,
        )
        self._target_critic_params = copy.deepcopy(critic_params)

        temp_params = self._temp_def.init(temp_key)["params"]
        self._temp = self._temp.replace(
            params=temp_params,
            opt_state=_make_dreamer_opt(self._temp_lr, **self._opt_kwargs).init(temp_params),
            step=0,
        )
        for obj in (self._actor_redo, self._critic_redo,
                    self._actor_grad_redo, self._critic_grad_redo):
            if obj is not None:
                obj._step = 0

    # ------------------------------------------------------------------
    # Inference (reuse JIT helpers from SACDreamerLearner)
    # ------------------------------------------------------------------

    def eval_actions(self, observations: np.ndarray) -> np.ndarray:
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
        ) = _update_jit_dreamer_dist(
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
            self.num_bins,
        )
        self._rng = new_rng
        self._actor = new_actor
        self._critic = new_critic
        self._target_critic_params = new_target_critic_params
        self._temp = new_temp

        # ReDo analysis
        redo_due = (
            (self._actor_redo is not None and self._actor_redo.should_run()) or
            (self._critic_redo is not None and self._critic_redo.should_run()) or
            (self._actor_grad_redo is not None and self._actor_grad_redo.should_run()) or
            (self._critic_grad_redo is not None and self._critic_grad_redo.should_run())
        )
        if redo_due:
            obs     = np.asarray(batch['observations'])
            actions = np.asarray(batch['actions'])
            info.update(self._apply_act_redo_dist(obs, actions))
            info.update(self._apply_grad_redo_dist(batch))
        else:
            for obj in (self._actor_redo, self._critic_redo,
                        self._actor_grad_redo, self._critic_grad_redo):
                if obj is not None:
                    obj._step += 1

        return info

    def collect_noised_act_stats(self, obs: np.ndarray,
                                 actions: np.ndarray) -> Dict:
        return {}

    # ------------------------------------------------------------------
    # Internal ReDo helpers
    # ------------------------------------------------------------------

    def _collect_actor_acts_dist(self, obs: np.ndarray) -> Dict:
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

    def _collect_critic_acts_dist(self, obs: np.ndarray,
                                   actions: np.ndarray) -> Dict:
        _, state = self._critic.apply_fn(
            {'params': self._critic.params},
            obs, actions,
            mutable=['intermediates'],
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

    def _apply_act_redo_dist(self, obs: np.ndarray,
                              actions: np.ndarray) -> Dict:
        info = {}
        if self._actor_redo is not None and self._actor_redo.should_run():
            self._rng, key = jax.random.split(self._rng)
            acts = self._collect_actor_acts_dist(obs)
            new_p, mets = self._actor_redo.step(self._actor.params, acts, key)
            self._actor = self._actor.replace(params=new_p)
            info.update(mets)
        elif self._actor_redo is not None:
            self._actor_redo._step += 1

        if self._critic_redo is not None and self._critic_redo.should_run():
            self._rng, key = jax.random.split(self._rng)
            acts = self._collect_critic_acts_dist(obs, actions)
            new_p, mets = self._critic_redo.step(self._critic.params, acts, key)
            self._critic = self._critic.replace(params=new_p)
            info.update(mets)
        elif self._critic_redo is not None:
            self._critic_redo._step += 1

        return info

    def _apply_grad_redo_dist(self, batch: FrozenDict) -> Dict:
        info = {}
        if self._actor_grad_redo is not None and self._actor_grad_redo.should_run():
            self._rng, key = jax.random.split(self._rng)
            grads = _compute_actor_grads_dist(
                self._actor, self._critic, self._temp, batch, self.num_bins)
            new_p, _, mets = self._actor_grad_redo.step(
                self._actor.params, grads, key)
            self._actor = self._actor.replace(params=new_p)
            info.update(mets)
        elif self._actor_grad_redo is not None:
            self._actor_grad_redo._step += 1

        if self._critic_grad_redo is not None and self._critic_grad_redo.should_run():
            self._rng, key = jax.random.split(self._rng)
            grads = _compute_critic_grads_dist(
                self._actor, self._critic, self._target_critic_params,
                self._temp, batch, self.discount, self.backup_entropy,
                self.critic_reduction, self.num_bins)
            new_p, _, mets = self._critic_grad_redo.step(
                self._critic.params, grads, key)
            self._critic = self._critic.replace(params=new_p)
            info.update(mets)
        elif self._critic_grad_redo is not None:
            self._critic_grad_redo._step += 1

        return info


def _compute_actor_grads_dist(actor, critic, temp, batch, num_bins):
    """Compute actor gradients using TwoHot pred() for Q values."""
    bins = _make_twohot_bins(num_bins)
    rng = jax.random.PRNGKey(0)

    def loss_fn(actor_params):
        dist = actor.apply_fn({'params': actor_params}, batch['observations'])
        actions, log_probs = dist.sample_and_log_prob(seed=rng)
        q_logits = critic.apply_fn(
            {'params': critic.params}, batch['observations'], actions)
        qs = _twohot_pred(q_logits, bins)
        q = qs.mean(axis=0)
        return (log_probs * temp.apply_fn({'params': temp.params}) - q).mean(), {}

    grads, _ = jax.grad(loss_fn, has_aux=True)(actor.params)
    return grads


def _compute_critic_grads_dist(actor, critic, target_params, temp, batch,
                                discount, backup_entropy, critic_reduction,
                                num_bins):
    """Compute critic gradients using TwoHot cross-entropy loss."""
    bins = _make_twohot_bins(num_bins)
    rng = jax.random.PRNGKey(0)
    target_critic = critic.replace(params=target_params)
    dist = actor.apply_fn({'params': actor.params}, batch['next_observations'])
    next_actions, next_log_probs = dist.sample_and_log_prob(seed=rng)
    next_q_logits = target_critic.apply_fn(
        {'params': target_params}, batch['next_observations'], next_actions)
    next_qs = _twohot_pred(next_q_logits, bins)
    next_q = (next_qs.min(axis=0) if critic_reduction == 'min'
               else next_qs.mean(axis=0))
    target_q = batch['rewards'] + discount * batch['masks'] * next_q
    if backup_entropy:
        target_q -= (discount * batch['masks'] *
                     temp.apply_fn({'params': temp.params}) * next_log_probs)

    def critic_loss_fn(critic_params):
        q_logits = critic.apply_fn(
            {'params': critic_params}, batch['observations'], batch['actions'])
        return _twohot_loss(q_logits, target_q, bins).mean(), {}

    grads, _ = jax.grad(critic_loss_fn, has_aux=True)(critic.params)
    return grads
