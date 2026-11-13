# SPDX-License-Identifier: GPL-3.0-only
"""Original Adam state and arithmetic used by the native trainer."""
from typing import NamedTuple
import jax
import jax.numpy as j
class Adam(NamedTuple):
    step: jax.Array
    m: object
    v: object


def adam_empty(p):
    return Adam(j.asarray(0, j.int32), jax.tree.map(j.zeros_like, p), jax.tree.map(j.zeros_like, p))


def moment_products(prior, gradient):
    return (jax.tree.map(lambda m: j.float32(.9)*m, prior),
            jax.tree.map(lambda g: j.float32(.1)*g, gradient))


def adam_finish(p, state, gradient, left, right):
    step = state.step + j.int32(1)
    m = jax.tree.map(lambda a,b: a+b, left, right)
    v = jax.tree.map(lambda x,g: j.float32(.999)*x+j.float32(.001)*j.square(g), state.v, gradient)
    correction1 = 1-j.float32(.9)**step
    correction2 = 1-j.float32(.999)**step
    p = jax.tree.map(lambda x,a,b: x-j.float32(.0003)*(a/correction1)/
        (j.sqrt(b/correction2)+j.float32(1e-8)), p, m, v)
    return p, Adam(step, m, v)
