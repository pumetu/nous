import jax
import jax.numpy as jnp

from functools import partial
from dataclasses import dataclass, fields, is_dataclass
from typing import Callable


def pytree_dataclass(cls, meta_fields: tuple = ()):
    assert not is_dataclass(cls), (
        f"{cls} is already a dataclass, please check {cls} again to avoid double decoration"
    )
    cls = dataclass(cls)
    all_fields = tuple(f.name for f in fields(cls) if f.init)
    data_fields = tuple(f for f in all_fields if f not in meta_fields)
    return jax.tree_util.register_dataclass(
        cls, data_fields=data_fields, meta_fields=meta_fields
    )


@partial(pytree_dataclass, meta_fields=("shape", "initializer"))
class ArraySpec:
    shape: tuple[int, ...]
    dtype: jnp.dtype = jnp.float32
    initializer: Callable | None = None


is_type = lambda x, cls: (type(x).__name__ == cls.__name__) and (
    type(x).__module__ == cls.__module__
)
is_param = lambda x: is_type(x, ArraySpec)


class Module:
    @classmethod
    def allocate(cls, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def initialize(cls, key: jax.random.PRNGKey, config):
        spec = cls.allocate(config)
        keys = iter(jax.random.split(key, len(jax.tree.leaves(spec, is_leaf=is_param))))

        def _initialize():
            return jax.tree.map(
                lambda x: x.initializer(next(keys), x.shape, x.dtype),
                spec,
                is_leaf=is_param,
            )

        return _initialize()


@pytree_dataclass
class Linear(Module):
    weight: jax.Array | ArraySpec
    bias: jax.Array | ArraySpec | None

    @classmethod
    def allocate(cls, shape: tuple[int, ...], dtype: jnp.dtype, bias: bool = True):
        _init = jax.nn.initializers.he_uniform()
        return cls(
            weight=ArraySpec(shape, dtype, _init),
            bias=ArraySpec(shape, dtype, _init) if bias else None,
        )

    def __call__(self, x):
        x = x @ self.weight
        return x + self.bias if self.bias is not None else x
