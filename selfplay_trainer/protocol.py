# SPDX-License-Identifier: GPL-3.0-only
"""Binary actor protocol and batched policy inference."""
from functools import lru_cache, partial
from pathlib import Path
import shutil
import jax
import jax.numpy as j
import numpy as np
from . import model
ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which('node') or 'node'
TOK, CAT, NUM, ACT = 97, 8, 32, 20
MAGIC = 0x314e414a
def layout(G):
    pad = lambda n: (n + 3) & ~3
    off, size = {}, 0
    for k, n in (('finished', G), ('winner', G), ('turns', 2 * G), ('rejected', 2 * G), ('actor', 2 * G), ('mask', 2 * G * ACT),
                 ('cat', 2 * G * TOK * CAT * 2), ('num', 2 * G * TOK * NUM * 2)):
        off[k] = size; size += pad(n)
    return off, size


def _policy(p, cat, num, mask, actor, key, mixed, config=model.SMALL):
    B = cat.shape[0]
    legal = mask.astype(bool)
    legal = legal | ((~actor)[..., None] & (j.arange(ACT) == 0))       # keep non-actor rows well formed
    logits, value, aux = model.forward(p, cat.reshape(-1, TOK, CAT).astype(j.int32),
                                       num.reshape(-1, TOK, NUM), legal.reshape(-1, ACT), mixed, config)
    logits = logits.reshape(B, 2, ACT); value = value.reshape(B, 2)
    action = jax.random.categorical(key, logits, axis=-1).astype(j.int32)
    logp = j.take_along_axis(jax.nn.log_softmax(logits, -1), action[..., None], -1)[..., 0]
    action = j.where(actor, action, -1)
    return action, logp, value, legal


def to_device(frame):
    return (j.asarray(frame['cat']), j.asarray(frame['num'].view(j.bfloat16)), j.asarray(frame['mask']),
            j.asarray(frame['actor'].astype(bool)))


def outcome(frame):
    """Per-slot terminal flags and per-side rewards for the step that produced this frame."""
    fin = frame['finished'] > 0
    win = frame['winner'].astype(np.int32)
    reward = np.zeros((len(fin), 2), np.float32)
    natural = frame['finished'] == 1
    for side in (0, 1):
        reward[:, side] = np.where(natural & (win == side), 1., np.where(natural & (win == 1 - side), -1., 0.))
    return fin, reward, frame['rejected'].astype(bool)


@lru_cache(maxsize=16)
def make_policy(config=model.SMALL):
    return jax.jit(partial(_policy, config=config), static_argnames=('mixed',))

policy = make_policy()
