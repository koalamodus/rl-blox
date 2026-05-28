from collections.abc import Callable

import chex
import jax
import jax.numpy as jnp
from flax import nnx


class MLP(nnx.Module):
    n_outputs: int
    activation: Callable[[jnp.ndarray], jnp.ndarray]
    hidden_layers: nnx.List[nnx.Linear]
    output_layer: nnx.Linear

    def __init__(
        self,
        n_features: int,
        n_outputs: int,
        hidden_nodes: list[int],
        activation: str,
        rngs: nnx.Rngs,
    ):
        chex.assert_scalar_positive(n_features)
        chex.assert_scalar_positive(n_outputs)

        self.n_outputs = n_outputs
        self.activation = getattr(jax.nn, activation)

        layers = nnx.List()
        n_in = n_features

        for n_out in hidden_nodes:
            layers.append(nnx.Linear(n_in, n_out, rngs=rngs))
            n_in = n_out

        self.hidden_layers = layers
        self.output_layer = nnx.Linear(n_in, n_outputs, rngs=rngs)

    # def __call__(self, x: jnp.ndarray, steer_vec_1: jnp.ndarray = jnp.zeros(32), steer_vec_2: jnp.ndarray = jnp.zeros(32)) -> jnp.ndarray:
    #     activations = []

    #     for layer in self.hidden_layers:
    #         x = self.activation(layer(x))
    #         x = x + steer_vec
    #         activations.append(x)

    #     self.sow(nnx.Intermediate, "hidden", activations)

    #     return self.output_layer(x)

    def __call__(
        self,
        x: jnp.ndarray,
        steer_vecs: jnp.ndarray | None = None,
    ) -> jnp.ndarray:

        if steer_vecs is None:
            steer_vecs = [
                jnp.zeros((self.hidden_layers[i].kernel.shape[-1],)) for i in range(len(self.hidden_layers))
            ]

        activations = []

        for i, layer in enumerate(self.hidden_layers):
            x = self.activation(layer(x))
            x = x + steer_vecs[i]
            activations.append(x)

        self.sow(nnx.Intermediate, "hidden", activations)

        return self.output_layer(x)