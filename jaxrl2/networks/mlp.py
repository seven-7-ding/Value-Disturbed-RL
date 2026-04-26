from math import dist
from typing import Callable, Optional, Sequence, Union

import flax.linen as nn
import jax.numpy as jnp
import jax
from flax.core.frozen_dict import FrozenDict

from jaxrl2.networks.constants import default_init


def _flatten_dict(x: Union[FrozenDict, jnp.ndarray]) -> jnp.ndarray:
    if hasattr(x, "values"):
        return jnp.concatenate([_flatten_dict(v) for k, v in sorted(x.items())], -1)
    else:
        return x


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    scale_final: Optional[float] = None
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        x = _flatten_dict(x)

        for i, size in enumerate(self.hidden_dims):
            if i + 1 == len(self.hidden_dims) and self.scale_final is not None:
                x = nn.Dense(size, kernel_init=default_init(self.scale_final))(x)
            else:
                x = nn.Dense(size, kernel_init=default_init())(x)

            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training
                    )
        return x

class RIMLP(nn.Module):
    """
    Randomness-injected MLP. Injects noise at the designated layer (default: the output before the last linear layer).
    """
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    scale_final: Optional[float] = None
    dropout_rate: Optional[float] = None
    # TDDO: Enable injecting noise at arbitrary layers.
    noise_layer: Optional[str] = "first" # "first", "last", or None
    relative_noise_scale: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        x = _flatten_dict(x)

        for i, size in enumerate(self.hidden_dims):
            if i + 1 == len(self.hidden_dims) and self.scale_final is not None:
                x = nn.Dense(size, kernel_init=default_init(self.scale_final))(x)
            else:
                x = nn.Dense(size, kernel_init=default_init())(x)

            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training
                    )
            if self.noise_layer is not None and (
                (self.noise_layer == "first" and i == 0) or
                (self.noise_layer == "last" and i + 2 == len(self.hidden_dims))
            ):
                noise = self.relative_noise_scale * jnp.abs(x) * jnp.clip(jax.random.normal(self.make_rng("noise"), x.shape), -1.0, 1.0)
                x = x + noise
        return x

class RAMLP(nn.Module):
    """
    Randomness-added MLP. Adds noise to the output of the designated layer (default: the output before the last linear layer).
    """
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    scale_final: Optional[float] = None
    dropout_rate: Optional[float] = None
    # TDDO: Enable adding noise at arbitrary layers.
    noise_layer: Optional[str] = "first" # "first", "last", or None
    dist_type: str = "gaussian"

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        x = _flatten_dict(x)

        for i, size in enumerate(self.hidden_dims):
            if i + 1 == len(self.hidden_dims) and self.scale_final is not None:
                x = nn.Dense(size, kernel_init=default_init(self.scale_final))(x)
            else:
                x = nn.Dense(size, kernel_init=default_init())(x)

            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training
                    )
            if self.noise_layer is not None and (
                (self.noise_layer == "first" and i == 0) or
                (self.noise_layer == "last" and i + 2 == len(self.hidden_dims))
            ):
                if self.dist_type == "gaussian":
                    _std = nn.Dense(size, kernel_init=default_init())(x)
                    _std = self.activations(_std)
                    _std = nn.Dense(size, kernel_init=default_init())(_std)
                    _std = jnp.clip(_std, 1e-6)
                    x = x + _std * jnp.clip(jax.random.normal(self.make_rng("noise"), x.shape), -1.0, 1.0)
                else:
                    raise ValueError(f"Invalid dist_type: {self.dist_type}")
        return x