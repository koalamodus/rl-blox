from flax import nnx
import jax
import jax.numpy as jnp

# note: some version has bias in SAE
class SAE(nnx.Module):
    latent_dim: int
    input_dim: int

    def __init__(self, latent_dim: int, input_dim: int, rngs: nnx.Rngs):
        self.encoder = nnx.Linear(input_dim, latent_dim, rngs=rngs)
        self.decoder = nnx.Linear(latent_dim, input_dim, rngs=rngs)

    def __call__(self, x):
        z = nnx.relu(self.encoder(x))
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z


# note: some version uses normalized L1-loss or devides L1-norm wtih number of SAE input
def sparsity_penalty(z, sparsity_weight=0.1):
    # return sparsity_weight * jnp.abs(z)
    return sparsity_weight * jnp.sum(jnp.abs(z))

def total_loss(model, x, sparsity_weight=0.1):
    x_reconstructed, z = model(x)

    # note: some version uses normalized mse instead of L2-norm
    reconstruction_loss = optax.squared_error(predictions=x_reconstructed, targets=x)

    sparsity_loss = sparsity_penalty(z, sparsity_weight)
    # print(f"Loss: {reconstruction_loss} + {sparsity_loss}")
    return reconstruction_loss + sparsity_loss

# Initialize model and optimizer
import optax
from functools import partial


rng = jax.random.PRNGKey(0)
model = SAE(latent_dim=64, input_dim=128, rngs=nnx.Rngs(rng))
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
# initialise optimiser
optimizer = nnx.Optimizer(
    model, optax.adam(learning_rate=0.003), wrt=nnx.Param
)


# Visualize the model
nnx.display(model)


# # Define training step
# def loss_fn(model):
#     return total_loss(model, x, sparsity_weight)

# @nnx.jit
# def train_step(model, optimizer, x, sparsity_weight=0.1):
#     loss, grads = nnx.value_and_grad(loss_fn)(model)
#     optimizer.update(model, grads)
#     return loss

def train_step_with_loss(
    loss, optimizer: nnx.Optimizer, sae: nnx.Module, *args, **kwargs
) -> tuple[float, float]:
    grad_fn = nnx.value_and_grad(loss, argnums=0, has_aux=True)
    value, grad = grad_fn(sae, *args, **kwargs)
    optimizer.update(sae, grad)
    return value

train_step = partial(train_step_with_loss, total_loss)
train_step = partial(nnx.jit)(train_step)

# def train_step(model, optimizer, x, sparsity_weight=0.1):
#     def loss_fn(model):
#         # Accessing parameters from the model's state
#         # params = nnx.state(model, nnx.Param)
#         # print(f"Parameters: {params}")
#         return total_loss(model, x, sparsity_weight=sparsity_weight)

#     loss, grads = nnx.value_and_grad(loss_fn)(model)
#     # print(f"grads: {len(grads)}")
#     # print(f"model: {model}")
#     optimizer.update(model, grads)
#     return loss

# Training loop

# Assuming `activations` is a JAX array of shape [num_samples, 128]
# activations = jnp.ones(128)  # Replace with actual activations
batch_size = 32
activations = jnp.ones((batch_size, 128))  # Example input data

# for epoch in range(100):
#     loss = train_step(model, optimizer, activations)
#     print(f'Epoch {epoch+1}, Loss: {loss}')

#     loss = train_step(model, optimizer, activations)

gamma: float = 0.99
for epoch in range(100): 
    loss = train_step(
        # optimizer, model, activations, gamma
        optimizer, model, activations
    )


# # Replace layer in MLP with SAE
# class MLPwithSAE(nnx.Module):
#     def __init__(self, sae_encoder: nnx.Module, rngs: nnx.Rngs):
#         self.sae_encoder = sae_encoder
#         self.linear = nnx.Linear(64, 10, rngs=rngs)

#     def __call__(self, x):
#         z = self.sae_encoder(x)
#         return self.linear(z)


# class MLPWithSAE(nnx.Module):
#     def __init__(self, layers: list, sae_layer_idx: int, sae: nnx.Module, rngs: nnx.Rngs):
#         self.layers = layers
#         self.sae_layer_idx = sae_layer_idx
#         self.sae = sae
#         self.rngs = rngs

#     def __call__(self, x):
#         for i, layer in enumerate(self.layers):
#             x = layer(x)
#             if i == self.sae_layer_idx:
#                 x, _ = self.sae(x)
#         return x