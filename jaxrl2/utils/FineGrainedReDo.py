"""JAX implementation of Fine-Grained ReDo (Reset of Dormant units).

Operates on ninjax parameter dicts inside JIT-compiled training loops.
Two variants:
  FGReDo           – activation-based; uses LAYER_CALLBACK to collect
                     activations, then conditionally reinitialises dormant
                     neurons every `frequency` steps.
  FGGradientReDo   – gradient-based; called from Optimizer.__call__ after
                     nj.grad, returns modified (params, grads) so that
                     dormant neurons are reset before the optimizer update.
"""
import math
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import ninjax as nj

f32 = jnp.float32
# Fixed tau list for multi-threshold dormancy logging
_TAU_LIST = (0.05, 0.1, 0.2, 0.4)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _reinit(key, shape, fan_in, dtype):
    """LeCun normal initialisation (JAX)."""
    std = math.sqrt(1.0 / max(fan_in, 1)) / 0.87962566103423978
    return (jax.random.truncated_normal(key, -2.0, 2.0, shape) * std).astype(dtype)


def _fan_in(kernel):
    """fan_in from kernel shape: Linear [in,out] or Conv [kH,kW,in,out]."""
    return math.prod(kernel.shape[:-1])


def _expand_mask(mask, ndim):
    """Broadcast [out] boolean mask to [1,…,1,out] matching kernel ndim."""
    return mask.reshape((1,) * (ndim - 1) + mask.shape)


def _neuron_score(activation):
    """Per-output-neuron mean absolute activation (last axis = neurons)."""
    return jnp.abs(f32(activation)).mean(axis=tuple(range(activation.ndim - 1)))


def _dormancy_mask(norm_score, tau, mode, percentage):
    """Return boolean dormancy mask given normalised scores."""
    if 'threshold' in mode:
        return norm_score <= tau
    k = max(1, int(norm_score.shape[0] * percentage))
    thresh = jnp.sort(norm_score)[min(k - 1, norm_score.shape[0] - 1)]
    return norm_score <= thresh


def _effective_rank(sv: jnp.ndarray) -> jnp.ndarray:
    """Effective rank via Shannon entropy of normalised singular values.
    Ref: https://ieeexplore.ieee.org/document/7098875/
    0 * log(0) is defined as 0.
    """
    norm_sv = jnp.abs(sv) / (jnp.sum(jnp.abs(sv)) + 1e-8)
    safe_log = jnp.where(norm_sv > 0,
                         jnp.log(jnp.where(norm_sv > 0, norm_sv, 1.0)), 0.0)
    entropy = -jnp.sum(norm_sv * safe_log)
    return f32(jnp.exp(entropy))


def _stable_rank(sv: jnp.ndarray, threshold: float = 0.99) -> jnp.ndarray:
    """Stable rank: fewest singular values whose cumulative fraction >= threshold."""
    norm_sv = jnp.abs(sv) / (jnp.sum(jnp.abs(sv)) + 1e-8)
    cumsum = jnp.cumsum(norm_sv)          # descending sv → ascending cumsum
    return f32(jnp.sum(cumsum < threshold) + 1)


# ---------------------------------------------------------------------------
# FGReDo – activation-based
# ---------------------------------------------------------------------------

class FGReDo(nj.Module):
    """Activation-based ReDo for JAX/ninjax.

    Collects layer activations via LAYER_CALLBACK during the training forward
    pass, identifies dormant neurons (low mean absolute activation), and
    conditionally reinitialises their weights every ``frequency`` steps.

    Usage in Agent.train():
      1. Set ``nn.LAYER_CALLBACK`` before calling ``self.opt(…)``.
      2. Call ``mets.update(self.act_redo.step(activations))`` afterwards.

    The reset is implemented with ``jnp.where`` so it is JIT-compatible: the
    reinitialised values are computed every step but only *selected* at reset
    intervals, adding negligible overhead.
    """

    tau: float = 0.0
    mode: str = 'threshold'
    frequency: int = 1000
    log_item: str = 'disabled'
    reset_start: int = 0
    reset_end: int = 0            # 0 = no upper limit
    skip_last_layer: bool = False  # skip the last-executed layer (like PyTorch version)
    rank_threshold: float = 0.99  # cumulative fraction for srank

    def __init__(self):
        self._step = nj.Variable(jnp.zeros, (), jnp.int32, name='step')

    def step(self, activations: Dict) -> Dict:
        """Analyse post-activation values and optionally reset dormant neurons.

        ``activations`` is a {layer_path: activation_tensor} dict collected by
        LAYER_CALLBACK.  Keys are in forward-pass execution order (Python dict
        preserves insertion order).  The last entry mirrors the PyTorch version's
        behaviour of leaving the final layer untouched.
        """
        self._step.write(self._step.read() + 1)
        if self.log_item == 'disabled':
            return {}

        cur = self._step.read()
        in_range = jnp.ones((), bool)
        if self.reset_end > 0:
            in_range = (cur >= self.reset_start) & (cur <= self.reset_end)
        elif self.reset_start > 0:
            in_range = cur >= self.reset_start
        # Gate the ENTIRE analysis on frequency; reset is additionally gated
        # by whether 'reset' appears in log_item (Python static check).
        should_analyze = (cur % self.frequency == 0) & in_range

        ctx = nj.context()
        metrics = {}

        # Mirror PyTorch: skip the last linear/conv layer in execution order.
        paths = list(activations.keys())
        if self.skip_last_layer and len(paths) > 1:
            paths = paths[:-1]

        need_erank = 'erank' in self.log_item
        need_srank = 'srank' in self.log_item

        for path in paths:
            act = activations[path]
            kkey = path + '/kernel'
            if kkey not in ctx:
                continue

            kernel = ctx[kkey]
            score = _neuron_score(act)
            norm_score = score / (score.mean() + 1e-9)

            lname = path.replace('/', '_')
            # Gate ALL dormancy metrics: NaN when not at a frequency step
            # (NaN propagates through Agg and can be filtered at log time).
            for t in _TAU_LIST:
                metrics[f'{self.name}/Dormant_{t}/{lname}'] = jnp.where(
                    should_analyze, f32(norm_score <= t).mean() * 100, jnp.nan)
            metrics[f'{self.name}/Act_Mean/{lname}'] = jnp.where(
                should_analyze, score.mean(), jnp.nan)

            # SVD-based rank metrics – gated on should_analyze.
            if need_erank or need_srank:
                act_2d = f32(act).reshape(-1, act.shape[-1])
                k = min(act_2d.shape[0], act_2d.shape[1])
                sv = jax.lax.cond(
                    should_analyze,
                    lambda a: jnp.linalg.svd(a, compute_uv=False),
                    lambda a: jnp.zeros(k, f32),
                    act_2d)
                if need_erank:
                    metrics[f'{self.name}/erank/{lname}'] = jnp.where(
                        should_analyze, _effective_rank(sv), jnp.nan)
                if need_srank:
                    metrics[f'{self.name}/srank/{lname}'] = jnp.where(
                        should_analyze, _stable_rank(sv, self.rank_threshold), jnp.nan)

            if 'reset' not in self.log_item:
                continue

            mask = _dormancy_mask(norm_score, self.tau, self.mode, self.tau)
            new_k = _reinit(nj.seed(), kernel.shape, _fan_in(kernel), kernel.dtype)
            ctx[kkey] = jnp.where(
                should_analyze & _expand_mask(mask, kernel.ndim), new_k, kernel)

            bkey = path + '/bias'
            if bkey in ctx:
                b = ctx[bkey]
                ctx[bkey] = jnp.where(should_analyze & mask, jnp.zeros_like(b), b)

        return metrics


# ---------------------------------------------------------------------------
# FGGradientReDo – gradient-based
# ---------------------------------------------------------------------------

class FGGradientReDo(nj.Module):
    """Gradient-based ReDo for JAX/ninjax.

    Analyses gradient magnitudes immediately after ``nj.grad`` to identify
    neurons with low gradient activity, reinitialises their weights, and
    zeros the corresponding gradients **before** the optimiser applies its
    update – matching the PyTorch calling convention.

    Called from ``Optimizer.__call__`` via the ``gradient_redo`` keyword:

        redo_metrics, params, grads = gradient_redo.step(params, grads)

    Returns ``(metrics, new_params, new_grads)``.
    """

    tau: float = 0.0
    mode: str = 'threshold'
    frequency: int = 1000
    log_item: str = 'disabled'
    reset_start: int = 0
    reset_end: int = 0   # 0 = no upper limit

    def __init__(self):
        self._step = nj.Variable(jnp.zeros, (), jnp.int32, name='step')

    def step(
        self, params: Dict, grads: Dict
    ) -> Tuple[Dict, Dict, Dict]:
        """Analyse grads; return ``(metrics, new_params, new_grads)``."""
        self._step.write(self._step.read() + 1)
        if self.log_item == 'disabled':
            return {}, params, grads

        cur = self._step.read()
        in_range = jnp.ones((), bool)
        if self.reset_end > 0:
            in_range = (cur >= self.reset_start) & (cur <= self.reset_end)
        elif self.reset_start > 0:
            in_range = cur >= self.reset_start
        should_analyze = (cur % self.frequency == 0) & in_range

        new_p, new_g = dict(params), dict(grads)
        metrics = {}

        for key, grad in grads.items():
            if not key.endswith('/kernel') or key not in params:
                continue

            kernel = params[key]
            if grad.ndim == 2:        # Linear  [in, out]
                score = jnp.abs(f32(grad)).mean(axis=0)
            elif grad.ndim == 4:      # Conv2D  [kH, kW, in, out]
                score = jnp.abs(f32(grad)).mean(axis=(0, 1, 2))
            else:
                continue

            norm_score = score / (score.mean() + 1e-9)
            lname = key[:-len('/kernel')].replace('/', '_')
            # Gate ALL grad dormancy metrics: NaN when not at a frequency step.
            for t in _TAU_LIST:
                metrics[f'{self.name}/GradDormant_{t}/{lname}'] = jnp.where(
                    should_analyze, f32(norm_score <= t).mean() * 100, jnp.nan)
            metrics[f'{self.name}/Grad_Mean/{lname}'] = jnp.where(
                should_analyze, score.mean(), jnp.nan)

            if 'reset' not in self.log_item:
                continue

            mask = _dormancy_mask(norm_score, self.tau, self.mode, self.tau)
            new_k = _reinit(nj.seed(), kernel.shape, _fan_in(kernel), kernel.dtype)
            mask_k = _expand_mask(mask, kernel.ndim)
            reset = should_analyze & mask_k
            new_p[key] = jnp.where(reset, new_k, kernel)
            new_g[key] = jnp.where(reset, jnp.zeros_like(grad), grad)

            bkey = key[:-len('kernel')] + 'bias'
            if bkey in params:
                b = params[bkey]
                new_p[bkey] = jnp.where(
                    should_analyze & mask, jnp.zeros_like(b), b)
                if bkey in grads:
                    new_g[bkey] = jnp.where(
                        should_analyze & mask,
                        jnp.zeros_like(grads[bkey]), grads[bkey])

        return metrics, new_p, new_g
