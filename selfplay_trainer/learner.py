# SPDX-License-Identifier: GPL-3.0-only
"""Original GAE, row packing and PPO objective; no battle simulator dependencies."""
import pickle
import time
import jax
import jax.numpy as j
import numpy as np
from . import objective
from . import optimizer as glu
from .optimizer import Adam
from .model import SMALL
@jax.jit
def stack(rows, outs):
    return (jax.tree.map(lambda *a: j.stack(a), *rows), jax.tree.map(lambda *a: j.stack(a), *outs))


@jax.jit
def gae(rows, outs, bootstrap, lam):
    """Undiscounted GAE per (lane, side) over acting rows; terminal resets carry."""
    value, eligible = rows['old_value'], outs['eligible']
    fin, reward = outs['finished'], outs['reward']

    def back(carry, x):
        v_next, a_next, r_pend = carry
        v, acc, f, r = x
        f2 = f[:, None]
        r_pend = j.where(f2, r, r_pend)
        v_next = j.where(f2, 0., v_next)
        a_next = j.where(f2, 0., a_next)
        adv = r_pend + v_next - v + lam * a_next
        adv = j.where(acc, adv, 0.)
        carry = (j.where(acc, v, v_next), j.where(acc, adv, a_next), j.where(acc, 0., r_pend))
        return carry, adv

    zero = j.zeros_like(bootstrap)
    _, adv = jax.lax.scan(back, (bootstrap, zero, zero), (value, eligible, fin, reward), reverse=True)
    return adv, adv + value


def _flat(x):
    return x.reshape((-1,) + x.shape[3:])


@jax.jit
def flatten(rows, eligible, adv, ret):
    data = {k: _flat(rows[k]) for k in ('categorical', 'numeric', 'legal', 'action', 'old_logp', 'old_value')}
    data.update(eligible=_flat(eligible), advantage=_flat(adv), returns=_flat(ret))
    return data


@jax.jit
def permutation(eligible_flat, key):
    u = jax.random.uniform(key, eligible_flat.shape)
    return j.argsort(j.where(eligible_flat, u, 2.)).astype(j.int32)


def _update(p, adam, data, perm, k, minibatch, mixed, config=SMALL):
    idx = jax.lax.dynamic_slice(perm, (k * minibatch,), (minibatch,))
    b = {key: data[key][idx] for key in data}
    b['categorical'] = b['categorical'].astype(j.int32)
    (total, aux), g = jax.value_and_grad(lambda q: objective.loss(q, b, mixed=mixed, config=config), has_aux=True)(p)
    left, right = glu.moment_products(adam.m, g)
    p2, adam2 = glu.adam_finish(p, adam, g, left, right)
    gnorm = j.sqrt(sum(j.sum(j.square(x)) for x in jax.tree.leaves(g)))
    m = dict(total=total, policy=aux['policy'], value=aux['value'], entropy=aux['entropy'], KL=aux['KL'],
             clip_fraction=aux['clip_fraction'], eligible=aux['eligible'], grad_norm=gnorm,
             finite=j.isfinite(total) & j.isfinite(gnorm) & aux['aux_finite'])
    return p2, adam2, m


def save_checkpoint(path, p, adam, meta):
    host = jax.device_get(dict(parameters=p, adam_step=adam.step, adam_m=adam.m, adam_v=adam.v))
    raw = pickle.dumps(dict(meta=meta, **host), protocol=5)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        f.write(raw); f.flush()
        import os; os.fsync(f.fileno())
    tmp.replace(path)
    return len(raw)


def load_checkpoint(path):
    with open(path, 'rb') as f:
        d = pickle.load(f)
    from .model import checkpoint_config
    checkpoint_config(d)
    if not all(np.isfinite(x).all() for x in jax.tree.leaves((d['parameters'],d['adam_m'],d['adam_v']))):
        raise ValueError('Nonfinite checkpoint')
    p = jax.tree.map(j.asarray, d['parameters'])
    adam = Adam(j.asarray(d['adam_step']), jax.tree.map(j.asarray, d['adam_m']), jax.tree.map(j.asarray, d['adam_v']))
    return p, adam, d['meta']


def make_update(config=SMALL):
    from functools import partial
    return jax.jit(partial(_update, config=config), static_argnames=('minibatch','mixed'), donate_argnums=(0,1))

update = make_update()
