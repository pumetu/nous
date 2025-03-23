import jax
import jax.numpy as jnp
from nous.nn import pytree_dataclass, ArraySpec
import nous.nn as nn
from dataclasses import dataclass
import optax

context_length = 8
batch_size = 256


@dataclass
class Config:
    vocab_size: int


@pytree_dataclass
class BigramLM(nn.Module):
    w_embed: jax.Array | ArraySpec

    @classmethod
    def allocate(cls, config):
        return cls(
            w_embed=ArraySpec(
                shape=(config.vocab_size, config.vocab_size),
                initializer=jax.nn.initializers.uniform(),
            )
        )

    def __call__(self, x, y):
        logits = self.w_embed[x]
        targets = jax.nn.one_hot(y, num_classes=vocab_size)
        loss = optax.softmax_cross_entropy(logits, targets)
        return loss


def get_batch(train: bool, key):
    data = train_data if train else val_data
    idx = jax.random.randint(key, (batch_size,), 0, len(data) - context_length)
    x = jnp.stack([data[i : i + context_length] for i in idx])
    y = jnp.stack([data[i + 1 : i + context_length + 1] for i in idx])
    return x, y


# Load data
text = open("data/shakespeare.txt", "r").read()
characters = sorted(list(set(text)))
vocab_size = len(characters)

# Tokenize
char_to_int = {ch: i for i, ch in enumerate(characters)}
int_to_char = {i: ch for i, ch in enumerate(characters)}
encode = lambda x: [char_to_int[c] for c in x]
decode = lambda x: "".join([int_to_char[i] for i in x])

dataset = jnp.array(encode(text), dtype=jnp.int32)
n = int(0.9 * len(dataset))
train_data, val_data = dataset[:n], dataset[n:]
key = jax.random.PRNGKey(42)

config = Config(vocab_size=vocab_size)
model = BigramLM.initialize(key, config)
optimizer = optax.adamw(learning_rate=1e-3)
state = optimizer.init(model)

for i in range(5000):
    key, dkey = jax.random.split(key)
    xb, yb = get_batch(True, dkey)
    loss, grad = jax.value_and_grad(lambda m: jax.vmap(m)(xb, yb).mean())(model)
    updates, state = optimizer.update(grad, state, model)
    model = optax.apply_updates(model, updates)
    if i % 500 == 0:
        print(loss)
