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
                x = nn.Dense(size, kernel_init=default_init(self.scale_final), name=f'layer_{i}')(x)
            else:
                x = nn.Dense(size, kernel_init=default_init(), name=f'layer_{i}')(x)

            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training
                    )
                if self.is_mutable_collection('intermediates'):
                    self.sow('intermediates', f'layer_{i}_act', x)
                if self.is_mutable_collection('intermediates'):
                    self.sow('intermediates', f'layer_{i}_noised_act', x)
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
                x = nn.Dense(size, kernel_init=default_init(self.scale_final), name=f'layer_{i}')(x)
            else:
                x = nn.Dense(size, kernel_init=default_init(), name=f'layer_{i}')(x)

            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training
                    )
                if self.is_mutable_collection('intermediates'):
                    self.sow('intermediates', f'layer_{i}_act', x)
            if self.noise_layer is not None and (
                ("first" in self.noise_layer and i == 0) or
                ("last" in self.noise_layer and i + 2 == len(self.hidden_dims))
            ):
                noise = self.relative_noise_scale * jax.lax.stop_gradient(jnp.abs(x)) * jnp.clip(jax.random.normal(self.make_rng("noise"), x.shape), -1.0, 1.0)
                # noise = self.relative_noise_scale * jnp.abs(x) * jnp.clip(jax.random.normal(self.make_rng("noise"), x.shape), -1.0, 1.0)
                x = x + noise
                if self.is_mutable_collection('intermediates'):
                    self.sow('intermediates', f'layer_{i}_noised_act', x)
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
                x = nn.Dense(size, kernel_init=default_init(self.scale_final), name=f'layer_{i}')(x)
            else:
                x = nn.Dense(size, kernel_init=default_init(), name=f'layer_{i}')(x)

            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training
                    )
                if self.is_mutable_collection('intermediates'):
                    self.sow('intermediates', f'layer_{i}_act', x)
            if self.noise_layer is not None and (
                ("first" in self.noise_layer and i == 0) or
                ("last" in self.noise_layer and i + 2 == len(self.hidden_dims))
            ):
                if self.dist_type == "gaussian":
                    _std = nn.Dense(size, kernel_init=default_init(), name=f'std_{i}_fc0')(x)
                    _std = self.activations(_std)
                    if self.is_mutable_collection('intermediates'):
                        self.sow('intermediates', f'std_{i}_fc0_act', _std)
                    _std = nn.Dense(size, kernel_init=default_init(), name=f'std_{i}_fc1')(_std)
                    if self.is_mutable_collection('intermediates'):
                        self.sow('intermediates', f'std_{i}_fc1_act', _std)
                    x = x + _std * jnp.clip(jax.random.normal(self.make_rng("noise"), x.shape), -1.0, 1.0)
                    if self.is_mutable_collection('intermediates'):
                        self.sow('intermediates', f'layer_{i}_noised_act', x)
                else:
                    raise ValueError(f"Invalid dist_type: {self.dist_type}")
        return x

class RNMLP(nn.Module):
    """
    Regularization-Normalized MLP. Applies LayerNorm at the designated layer(s).
    norm_layer supports "first", "last", or "first_last" (both).
    Uses string-contains matching, so "_first_last" also works.
    """
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    scale_final: Optional[float] = None
    dropout_rate: Optional[float] = None
    norm_layer: Optional[str] = "first"  # "first", "last", "first_last", or None

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        x = _flatten_dict(x)

        for i, size in enumerate(self.hidden_dims):
            if i + 1 == len(self.hidden_dims) and self.scale_final is not None:
                x = nn.Dense(size, kernel_init=default_init(self.scale_final), name=f'layer_{i}')(x)
            else:
                x = nn.Dense(size, kernel_init=default_init(), name=f'layer_{i}')(x)

            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training
                    )
                if self.is_mutable_collection('intermediates'):
                    self.sow('intermediates', f'layer_{i}_act', x)
            if self.norm_layer is not None and (
                ("first" in self.norm_layer and i == 0) or
                ("last" in self.norm_layer and i + 2 == len(self.hidden_dims))
            ):
                x = nn.LayerNorm(name=f'layer_{i}_norm')(x)
                if self.is_mutable_collection('intermediates'):
                    self.sow('intermediates', f'layer_{i}_noised_act', x)
        return x


def build_mlp(vd_mode: str, hidden_dims: Sequence[int], activations=nn.relu, dropout_rate=None, activate_final=False, ) -> nn.Module:
    """
    Factory: 根据 vd_mode 字符串构建对应的 MLP 模块实例。

    vd_mode 格式示例：
      "disabled"              -> MLP（标准）
      "RI_first"              -> RIMLP，在第一层注入噪声
      "RI_last"               -> RIMLP，在倒数第二层注入噪声
      "RI_first_last"         -> RIMLP，在两处均注入噪声
      "RA_first_gaussian"     -> RAMLP，在第一层添加学习到的高斯噪声
      "RA_last_gaussian"      -> RAMLP，在倒数第二层添加学习到的高斯噪声
      "RN_first"              -> RNMLP，在第一层施加 LayerNorm
      "RN_last"               -> RNMLP，在倒数第二层施加 LayerNorm
      "RN_first_last"         -> RNMLP，在两处均施加 LayerNorm
    """
    # 提取层位置信息（利用字符串包含检测，与各 MLP 内部逻辑保持一致）
    layer_spec = ""
    if "first" in vd_mode:
        layer_spec += "_first"
    if "last" in vd_mode:
        layer_spec += "_last"

    if "disabled" in vd_mode:
        return MLP(hidden_dims, activations=activations, dropout_rate=dropout_rate, activate_final=activate_final)
    elif "RI" in vd_mode:
        return RIMLP(hidden_dims, activations=activations, noise_layer=layer_spec, dropout_rate=dropout_rate, activate_final=activate_final)
    elif "RA" in vd_mode:
        dist_type = "gaussian" if "gaussian" in vd_mode else "gaussian"
        return RAMLP(hidden_dims, activations=activations, noise_layer=layer_spec, dist_type=dist_type, dropout_rate=dropout_rate, activate_final=activate_final)
    elif "RN" in vd_mode:
        return RNMLP(hidden_dims, activations=activations, norm_layer=layer_spec, dropout_rate=dropout_rate, activate_final=activate_final)
    else:
        raise ValueError(f"Invalid vd_mode: {vd_mode}")
