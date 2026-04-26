from typing import Callable, Sequence

import flax.linen as nn
import jax.numpy as jnp

from jaxrl2.networks.mlp import MLP, RIMLP, RAMLP


class StateActionValue(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    vd_mode: str = "disabled"

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray, training: bool = False
    ) -> jnp.ndarray:
        inputs = {"states": observations, "actions": actions}
        
        if "disabled" in self.vd_mode:
            critic = MLP((*self.hidden_dims, 1), activations=self.activations)(
                inputs, training=training
            )
        elif "RI" in self.vd_mode:
            if "first" in self.vd_mode:
                noise_layer = "first"
            elif "last" in self.vd_mode:
                noise_layer = "last"
            else:
                raise ValueError(f"Invalid vd_mode: {self.vd_mode}")
            critic = RIMLP((*self.hidden_dims, 1), activations=self.activations, noise_layer=noise_layer)(
                inputs, training=training
            )
        elif "RA" in self.vd_mode:
            if "first" in self.vd_mode:
                noise_layer = "first"
            elif "last" in self.vd_mode:
                noise_layer = "last"
            else:
                raise ValueError(f"Invalid vd_mode: {self.vd_mode}")
            
            if "gaussian" in self.vd_mode:
                dist_type = "gaussian"
            else:
                raise ValueError(f"Invalid vd_mode: {self.vd_mode}")
            critic = RAMLP((*self.hidden_dims, 1), activations=self.activations, noise_layer=noise_layer, dist_type=dist_type)(
                inputs, training=training
            )
        return jnp.squeeze(critic, -1)
