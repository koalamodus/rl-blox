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

# # note: some version uses normalized L1-loss or devides L1-norm wtih number of SAE input
# def sparsity_penalty(z, sparsity_weight=0.1):
#     # return sparsity_weight * jnp.abs(z)
#     return sparsity_weight * jnp.sum(jnp.abs(z))

# def total_loss(model, x, sparsity_weight=0.1):
#     x_reconstructed, z = model(x)

#     # note: some version uses normalized mse instead of L2-norm
#     reconstruction_loss = optax.squared_error(predictions=x_reconstructed, targets=x)

#     sparsity_loss = sparsity_penalty(z, sparsity_weight)
#     # print(f"Loss: {reconstruction_loss} + {sparsity_loss}")
#     return reconstruction_loss + sparsity_loss


# from functools import partial

# # Define training step
# def loss_fn(model):
#     return total_loss(model, x, sparsity_weight)

# @nnx.jit
# def train_step(model, optimizer, x, sparsity_weight=0.1):
#     loss, grads = nnx.value_and_grad(loss_fn)(model)
#     optimizer.update(model, grads)
#     return loss

# def train_step_with_loss(
#     loss, optimizer: nnx.Optimizer, sae: nnx.Module, *args, **kwargs
# ) -> tuple[float, float]:
#     grad_fn = nnx.value_and_grad(loss, argnums=0, has_aux=True)
#     value, grad = grad_fn(sae, *args, **kwargs)
#     optimizer.update(sae, grad)
#     return value

# train_step = partial(train_step_with_loss, total_loss)
# train_step = partial(nnx.jit)(train_step)

# def train_step(model, optimizer, x_batch):
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



# Initialize model and optimizer
import optax
learning_rate = 1e-3
sparsity_coeff = 1e-3

latent_dim=4
input_dim=32

rng = jax.random.PRNGKey(0)
model = SAE(latent_dim, input_dim, rngs=nnx.Rngs(rng))
optimizer = nnx.Optimizer(model, optax.adam(learning_rate), wrt=nnx.Param)

# Training step
@nnx.jit
def train_step(model, opt, x_batch):
    def loss_fn(m):
        x_hat, z = m(x_batch)
        recon_loss = jnp.mean((x_hat - x_batch) ** 2)
        sparsity_loss = sparsity_coeff * jnp.mean(jnp.abs(z))
        return recon_loss + sparsity_loss
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    opt.update(model, grads)
    return loss

def create_dummy_database(num_samples: int,
                          input_dim: int,
                          rng: jax.Array):
    """
    Creates a dummy database of shape (num_samples, input_dim) where:
      - base_array is integers from 0 to input_dim-1 (as float32)
      - each sample i multiplies base_array by a random multiplier m_i in [0.5, 2.0)
    
    Args:
      num_samples : number of samples (rows) to create
      input_dim   : dimension of each sample (length of base array)
      rng         : JAX PRNGKey
    
    Returns:
      data_array : jnp.ndarray of shape (num_samples, input_dim), dtype float32
    """
    base_array = jnp.arange(input_dim, dtype=jnp.float32)
    # split key for m_values generation
    rng, subkey = jax.random.split(rng)
    m_values = jax.random.uniform(subkey,
                                  shape=(num_samples,),
                                  minval=0.5,
                                  maxval=2.0,
                                  dtype=jnp.float32)  # sample multipliers
    # broadcast multiplication
    data_array = base_array[None, :] * m_values[:, None]
    return data_array



num_samples = int(1e5)
db = create_dummy_database(num_samples, input_dim, rng)


# Split
train_frac = 0.8
n_train    = int(num_samples * train_frac)
train_data = db[:n_train]
val_data   = db[n_train:]

num_epochs = 5
batch_size = 128

# Simple loop over batches
for epoch in range(num_epochs):
    for start in range(0, train_data.shape[0], batch_size):
        end = start + batch_size
        if end > train_data.shape[0]:
            break  # or handle last smaller batch if you allow it
        batch = train_data[start:end]

        # Now batch is your input **and** target for autoencoder
        loss = train_step(model, optimizer, batch)
    print(f"Epoch {epoch}, last loss = {loss:.4f}")


# Eval step
@nnx.jit
def eval_step(model,x_batch):
    def loss_fn(m):
        x_hat, z = m(x_batch)
        recon_loss = jnp.mean((x_hat - x_batch) ** 2)
        sparsity_loss = sparsity_coeff * jnp.mean(jnp.abs(z))
        return recon_loss + sparsity_loss
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    return loss

# Compute validation loss (without update)
val_loss = 0.0
val_batches = 0
for start in range(0, val_data.shape[0], batch_size):
    end = start + batch_size
    if end > val_data.shape[0]:
        break
    batch = val_data[start:end]
    val_loss += eval_step(model, batch)  # define eval_step similarly to train_step but no update
    val_batches += 1
val_loss /= val_batches
print(f"Validation loss: {val_loss:.4f}")

import matplotlib.pyplot as plt


state = nnx.state(model)
print(state)

encoder_w = state['encoder']['kernel'].value
encoder_b = state['encoder']['bias'].value
decoder_w = state['decoder']['kernel'].value
decoder_b = state['decoder']['bias'].value


def visualize_encoder(encoder_w, encoder_b):
    # Reshape bias as row
    encoder_b_row = encoder_b[jnp.newaxis, :]  # shape (1, latent)
    # Create empty row (NaNs) for separator
    empty_row = jnp.full((1, encoder_w.shape[1]), jnp.nan)

    # Stack bias, empty row, and weights
    combined_matrix = jnp.vstack([encoder_b_row, empty_row, encoder_w])

    n_input, n_latent = encoder_w.shape
    n_total_rows = combined_matrix.shape[0]

    # Shared color scale (ignore NaNs)
    vmin = jnp.nanmin(combined_matrix)
    vmax = jnp.nanmax(combined_matrix)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot combined matrix; NaNs appear as white
    im = ax.imshow(combined_matrix, cmap='viridis', aspect='equal', vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Weight / Bias value')

    # X-axis: latent neuron index
    ax.set_xticks(jnp.arange(n_latent))
    ax.set_xticklabels([str(i) for i in range(n_latent)])
    ax.set_xlabel("Latent neuron index")

    # Y-axis labels
    y_labels = ["Bias", ""] + [f"{i}" for i in range(n_input)]
    ax.set_yticks(jnp.arange(n_total_rows))
    ax.set_yticklabels(y_labels)
    ax.set_ylabel("Input neuron / Bias")

    ax.set_title("Encoder bias (top) and weights (bottom)")

    plt.tight_layout()
    plt.show()

def visualize_decoder(decoder_w, decoder_b):
    # Reshape bias as row
    decoder_b_row = decoder_b[jnp.newaxis, :]  # shape (1, output)
    # Create empty row (NaNs) for separator
    empty_row = jnp.full((1, decoder_w.shape[1]), jnp.nan)

    # Stack bias, empty row, and weights
    combined_matrix = jnp.vstack([decoder_b_row, empty_row, decoder_w])

    n_output, n_latent = decoder_w.shape
    n_total_rows = combined_matrix.shape[0]

    # Shared color scale (ignore NaNs)
    vmin = jnp.nanmin(combined_matrix)
    vmax = jnp.nanmax(combined_matrix)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot combined matrix; NaNs appear as white
    im = ax.imshow(combined_matrix, cmap='viridis', aspect='equal', vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Weight / Bias value')

    # X-axis: latent neuron index (input to decoder)
    ax.set_xticks(jnp.arange(n_latent))
    ax.set_xticklabels([str(i) for i in range(n_latent)])
    ax.set_xlabel("Latent neuron index")

    # Y-axis labels
    y_labels = ["Bias", ""] + [f"Output {i}" for i in range(n_output)]
    ax.set_yticks(jnp.arange(n_total_rows))
    ax.set_yticklabels(y_labels)
    ax.set_ylabel("Output neuron / Bias")

    ax.set_title("Decoder bias (top) and weights (bottom)")

    plt.tight_layout()
    plt.show()


visualize_encoder(encoder_w, encoder_b)
visualize_decoder(decoder_w, decoder_b)


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