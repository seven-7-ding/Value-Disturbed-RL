"""Activation-based and gradient-based ReDo for SAC (Flax/JAX).

Mirrors FineGrainedReDo.py but operates on plain Flax parameter trees
(nested FrozenDicts / dicts) instead of ninjax context dicts.

Two classes:
  SACReDo         – activation-based; called after each training step with
                    layer activations collected via Flax's ``sow`` mechanism.
  SACGradientReDo – gradient-based; called after jax.grad with (params, grads),
                    returns (new_params, new_grads, metrics).
"""
import math
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import unfreeze

f32 = jnp.float32
_TAU_LIST = (0.05, 0.1, 0.2, 0.4)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _lecun_normal(key, shape, fan_in, dtype):
    std = math.sqrt(1.0 / max(fan_in, 1)) / 0.87962566103423978
    return (jax.random.truncated_normal(key, -2.0, 2.0, shape) * std).astype(dtype)


def _fan_in(kernel):
    return math.prod(kernel.shape[:-1])


def _neuron_score(act):
    """Mean |activation| over all non-output axes."""
    return jnp.abs(f32(act)).mean(axis=tuple(range(act.ndim - 1)))


def _dormancy_mask(norm_score, tau: float, mode: str):
    if 'threshold' in mode:
        return norm_score <= tau
    # percentage mode: dormant if in lowest tau-fraction
    k = max(1, int(norm_score.shape[0] * tau))
    thresh = jnp.sort(norm_score)[min(k - 1, norm_score.shape[0] - 1)]
    return norm_score <= thresh


def _effective_rank(sv: jnp.ndarray) -> f32:
    p = jnp.abs(sv) / (jnp.sum(jnp.abs(sv)) + 1e-8)
    h = -jnp.sum(p * jnp.where(p > 0, jnp.log(jnp.where(p > 0, p, 1.0)), 0.0))
    return f32(jnp.exp(h))


def _stable_rank(sv: jnp.ndarray, threshold: float = 0.99) -> f32:
    p = jnp.abs(sv) / (jnp.sum(jnp.abs(sv)) + 1e-8)
    return f32(jnp.sum(jnp.cumsum(p) < threshold) + 1)


# ---------------------------------------------------------------------------
# Nested-dict helpers (Flax param trees use '/' as path separator here)
# ---------------------------------------------------------------------------

def _flatten(tree, prefix: str = '') -> Dict[str, Any]:
    """Flatten a nested dict to {'/'.join(path): leaf}."""
    out = {}
    if not (isinstance(tree, dict) or hasattr(tree, 'items')):
        return {prefix: tree} if prefix else {}
    for k, v in tree.items():
        path = f'{prefix}/{k}' if prefix else k
        if isinstance(v, dict) or hasattr(v, 'items'):
            out.update(_flatten(v, path))
        else:
            out[path] = v
    return out


def _get(tree, path: str) -> Optional[Any]:
    """Get leaf at slash-separated path, or None if missing."""
    node = tree
    for p in path.split('/'):
        if not (isinstance(node, dict) or hasattr(node, 'items')):
            return None
        node = node.get(p)
        if node is None:
            return None
    return node


def _set(tree, path: str, value: Any) -> dict:
    """Return a new nested plain-dict with leaf at path replaced by value."""
    parts = path.split('/', 1)
    new = dict(tree)
    if len(parts) == 1:
        new[parts[0]] = value
    else:
        child = dict(new.get(parts[0], {}))
        new[parts[0]] = _set(child, parts[1], value)
    return new


def _unfreeze(params) -> dict:
    """Convert FrozenDict (or plain dict) to a mutable plain dict."""
    if hasattr(params, '_dict'):          # FrozenDict
        return unfreeze(params)
    if isinstance(params, dict):
        return {k: _unfreeze(v) if isinstance(v, dict) or hasattr(v, '_dict') else v
                for k, v in params.items()}
    return params


# ---------------------------------------------------------------------------
# SACReDo – activation-based
# ---------------------------------------------------------------------------

class SACReDo:
    """Activation-based ReDo for SAC.

    Expects ``activations`` to be a flat dict produced by Flax's ``sow``
    mechanism inside MLP:

        {'MLP_0/act_0': act_0, 'MLP_0/act_1': act_1, ...}

    The ``name`` argument (e.g. 'actor' or 'critic') is prepended to every
    metric key so logs from different networks are easy to distinguish.
    """

    def __init__(
        self,
        name: str = 'net',
        tau: float = 0.0,
        mode: str = 'threshold',
        frequency: int = 1000,
        log_item: str = 'disabled',
        skip_last_layer: bool = False,
        rank_threshold: float = 0.99,
        reset_start: int = 0,
        reset_end: int = 0,
    ):
        self.name = name
        self.tau = tau
        self.mode = mode
        self.frequency = frequency
        self.log_item = log_item
        self.skip_last_layer = skip_last_layer
        self.rank_threshold = rank_threshold
        self.reset_start = reset_start
        self.reset_end = reset_end
        self._step = 0

    def should_run(self) -> bool:
        """Return True if calling step() right now will do real work."""
        if self.log_item == 'disabled':
            return False
        cur = self._step
        in_range = (self.reset_end == 0 or cur <= self.reset_end) and \
                   (self.reset_start == 0 or cur >= self.reset_start)
        return in_range and cur > 0 and (cur % self.frequency == 0)

    def step(self, params, activations: Dict, key) -> Tuple[Any, Dict]:
        """Analyse activations; return ``(new_params, metrics)``."""
        if self.log_item == 'disabled':
            self._step += 1
            return params, {}

        cur = self._step
        self._step += 1
        in_range = (self.reset_end == 0 or cur <= self.reset_end) and \
                   (self.reset_start == 0 or cur >= self.reset_start)
        if cur == 0 or cur % self.frequency != 0 or not in_range:
            return params, {}

        paths = list(activations.keys())
        if self.skip_last_layer and len(paths) > 1:
            paths = paths[:-1]

        need_erank = 'erank' in self.log_item
        need_srank = 'srank' in self.log_item
        do_reset   = 'reset' in self.log_item

        metrics = {}
        flat_p = _unfreeze(params)

        for path in paths:
            act = activations[path]
            # sow key is '{module}/layer_{i}_act' or '{module}/std_{i}_fc0_act'.
            # Strip the '_act' suffix to get the Dense layer path, then look up kernel.
            if not path.endswith('_act'):
                continue
            is_noised = path.endswith('_noised_act')
            if is_noised:
                layer_path = path[:-len('_noised_act')]
            else:
                layer_path = path[:-4]   # remove '_act'
            kernel = _get(flat_p, layer_path + '/kernel')
            if kernel is None:
                continue

            score      = _neuron_score(act)
            norm_score = score / (score.mean() + 1e-9)
            lname = path.rsplit('/', 1)[-1]
            # noised activations go into a separate 'redo_noised' group
            group = 'redo_noised' if is_noised else 'redo'

            pfx = self.name
            for t in _TAU_LIST:
                metrics[f'{pfx}/{group}/Dormant_{t}/{lname}'] = float(
                    f32(norm_score <= t).mean() * 100)
            metrics[f'{pfx}/{group}/Act_Mean/{lname}'] = float(score.mean())

            if need_erank or need_srank:
                act_2d = f32(act).reshape(-1, act.shape[-1])
                sv = jnp.linalg.svd(act_2d, compute_uv=False)
                if need_erank:
                    metrics[f'{pfx}/{group}/erank/{lname}'] = float(_effective_rank(sv))
                if need_srank:
                    metrics[f'{pfx}/{group}/srank/{lname}'] = float(
                        _stable_rank(sv, self.rank_threshold))

            if not do_reset:
                continue

            mask = _dormancy_mask(norm_score, self.tau, self.mode)
            key, subkey = jax.random.split(key)
            new_k   = _lecun_normal(subkey, kernel.shape, _fan_in(kernel), kernel.dtype)
            mask_k  = mask.reshape((1,) * (kernel.ndim - 1) + mask.shape)
            flat_p  = _set(flat_p, layer_path + '/kernel', jnp.where(mask_k, new_k, kernel))

            bias = _get(flat_p, layer_path + '/bias')
            if bias is not None:
                flat_p = _set(flat_p, layer_path + '/bias',
                              jnp.where(mask, jnp.zeros_like(bias), bias))

        return flat_p, metrics


# ---------------------------------------------------------------------------
# SACGradientReDo – gradient-based
# ---------------------------------------------------------------------------

class SACGradientReDo:
    """Gradient-based ReDo for SAC.

    Called after ``jax.grad`` with ``(params, grads)``.
    Identifies neurons with low gradient magnitude, reinitialises their
    weights, and zeroes the corresponding gradients before the optimiser step.

    Returns ``(new_params, new_grads, metrics)``.
    """

    def __init__(
        self,
        name: str = 'net',
        tau: float = 0.0,
        mode: str = 'threshold',
        frequency: int = 1000,
        log_item: str = 'disabled',
        reset_start: int = 0,
        reset_end: int = 0,
    ):
        self.name = name
        self.tau = tau
        self.mode = mode
        self.frequency = frequency
        self.log_item = log_item
        self.reset_start = reset_start
        self.reset_end = reset_end
        self._step = 0

    def should_run(self) -> bool:
        """Return True if calling step() right now will do real work."""
        if self.log_item == 'disabled':
            return False
        cur = self._step
        in_range = (self.reset_end == 0 or cur <= self.reset_end) and \
                   (self.reset_start == 0 or cur >= self.reset_start)
        return in_range and cur > 0 and (cur % self.frequency == 0)

    def step(self, params, grads, key) -> Tuple[Any, Any, Dict]:
        """Analyse grads; return ``(new_params, new_grads, metrics)``."""
        if self.log_item == 'disabled':
            self._step += 1
            return params, grads, {}

        cur = self._step
        self._step += 1
        in_range = (self.reset_end == 0 or cur <= self.reset_end) and \
                   (self.reset_start == 0 or cur >= self.reset_start)
        if cur == 0 or cur % self.frequency != 0 or not in_range:
            return params, grads, {}

        do_reset = 'reset' in self.log_item
        metrics  = {}
        flat_p   = _unfreeze(params)
        flat_g   = _unfreeze(grads)

        for path, grad in _flatten(flat_g).items():
            if not path.endswith('/kernel'):
                continue
            kernel = _get(flat_p, path)
            if kernel is None:
                continue
            if grad.ndim < 2:
                continue

            base_lname = path[:-len('/kernel')].rsplit('/', 1)[-1]
            pfx = self.name

            # ndim==2: non-vmapped Linear [in, out]  → 1 member
            # ndim==3: vmapped Linear [num_qs, in, out] → num_qs members
            is_vmapped  = grad.ndim >= 3
            num_members = grad.shape[0] if is_vmapped else 1

            new_k_slices = []
            new_g_slices = []
            masks        = []   # one mask per member, or None if not resetting

            for qi in range(num_members):
                g_i = grad[qi]   if is_vmapped else grad    # (in, out)
                k_i = kernel[qi] if is_vmapped else kernel

                score      = jnp.abs(f32(g_i)).mean(axis=tuple(range(g_i.ndim - 1)))
                norm_score = score / (score.mean() + 1e-9)
                member_pfx = f'{pfx}_{qi}' if is_vmapped else pfx

                for t in _TAU_LIST:
                    metrics[f'{member_pfx}/grad_redo/GradDormant_{t}/{base_lname}'] = float(
                        f32(norm_score <= t).mean() * 100)
                metrics[f'{member_pfx}/grad_redo/Grad_Mean/{base_lname}'] = float(score.mean())

                if do_reset:
                    mask   = _dormancy_mask(norm_score, self.tau, self.mode)
                    key, subkey = jax.random.split(key)
                    new_k  = _lecun_normal(subkey, k_i.shape, _fan_in(k_i), k_i.dtype)
                    mask_k = mask.reshape((1,) * (k_i.ndim - 1) + mask.shape)
                    new_k_slices.append(jnp.where(mask_k, new_k, k_i))
                    new_g_slices.append(jnp.where(mask_k, jnp.zeros_like(g_i), g_i))
                    masks.append(mask)
                else:
                    new_k_slices.append(k_i)
                    new_g_slices.append(g_i)
                    masks.append(None)

            if do_reset:
                if is_vmapped:
                    flat_p = _set(flat_p, path, jnp.stack(new_k_slices, axis=0))
                    flat_g = _set(flat_g, path, jnp.stack(new_g_slices, axis=0))
                else:
                    flat_p = _set(flat_p, path, new_k_slices[0])
                    flat_g = _set(flat_g, path, new_g_slices[0])

                bpath = path[:-len('kernel')] + 'bias'
                bias  = _get(flat_p, bpath)
                if bias is not None:
                    bg = _get(flat_g, bpath)
                    if is_vmapped:
                        new_b  = jnp.stack([jnp.where(m, jnp.zeros_like(bias[qi]), bias[qi])
                                            for qi, m in enumerate(masks)], axis=0)
                        flat_p = _set(flat_p, bpath, new_b)
                        if bg is not None:
                            new_bg = jnp.stack([jnp.where(m, jnp.zeros_like(bg[qi]), bg[qi])
                                                for qi, m in enumerate(masks)], axis=0)
                            flat_g = _set(flat_g, bpath, new_bg)
                    else:
                        mask = masks[0]
                        flat_p = _set(flat_p, bpath, jnp.where(mask, jnp.zeros_like(bias), bias))
                        if bg is not None:
                            flat_g = _set(flat_g, bpath, jnp.where(mask, jnp.zeros_like(bg), bg))

        return flat_p, flat_g, metrics
