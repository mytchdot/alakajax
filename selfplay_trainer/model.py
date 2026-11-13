# SPDX-License-Identifier: GPL-3.0-only
"""Size-configurable native policy; the legacy 128/4/4 math and RNG order are preserved.

Only width, layer count and attention heads vary. The encoder, vocabulary, packets,
49 packed tokens, exact-erf GELU and BF16_BF16_F32 execution stay unchanged.
Configuration is checkpoint metadata, never an optimizable parameter leaf.
"""
from dataclasses import asdict, dataclass
from functools import partial

import jax
import jax.numpy as j
import numpy as np

TOKENS = 49


@dataclass(frozen=True)
class ModelConfig:
    width: int = 128
    layers: int = 4
    heads: int = 4

    def __post_init__(self):
        if any(type(x) is not int or x <= 0 for x in (self.width, self.layers, self.heads)):
            raise ValueError('width, layers and heads must be positive integers')
        if self.width % self.heads:
            raise ValueError('width must be divisible by heads')

    def to_dict(self):
        return asdict(self)


SMALL = ModelConfig()
BIG = ModelConfig(384, 6, 12)


def initialize(vocabulary_size, seed=15501, config=SMALL):
    rng = np.random.Generator(np.random.PCG64(seed))
    w = config.width

    def weight(a, b, scale=1.):
        return j.asarray((rng.standard_normal((a, b)) * (scale / a**.5)).astype(np.float32))

    p = dict(embedding=weight(vocabulary_size, w, 2.),
             numeric=weight(32, w), pack=weight(5*w, w),
             position=weight(TOKENS, w, .1), final_norm=j.ones(w, j.float32),
             policy=weight(w, 1, .1), value=weight(w, 1, .1),
             opponent=weight(w, 20, .1))
    p['layers'] = tuple(dict(norm1=j.ones(w, j.float32), norm2=j.ones(w, j.float32),
        qkv=weight(w, 3*w), out=weight(w, w), up=weight(w, 4*w),
        down=weight(4*w, w)) for _ in range(config.layers))
    return p


def tensor_shapes(vocabulary_size, config):
    w = config.width
    shapes = dict(embedding=(vocabulary_size, w), numeric=(32, w), pack=(5*w, w),
                  position=(TOKENS, w), final_norm=(w,), policy=(w, 1), value=(w, 1), opponent=(w, 20))
    layer = dict(norm1=(w,), norm2=(w,), qkv=(w, 3*w), out=(w, w), up=(w, 4*w), down=(4*w, w))
    shapes.update({f'layers.{i}.{k}': shape for i in range(config.layers) for k, shape in layer.items()})
    return shapes


def named_tensors(p):
    for name in ('embedding', 'numeric', 'pack', 'position', 'final_norm', 'policy', 'value', 'opponent'):
        yield name, p[name]
    for i, layer in enumerate(p['layers']):
        for name in ('norm1', 'norm2', 'qkv', 'out', 'up', 'down'):
            yield f'layers.{i}.{name}', layer[name]


def validate_parameters(p, config):
    expected = tensor_shapes(int(p['embedding'].shape[0]), config)
    actual = dict(named_tensors(p))
    if actual.keys() != expected.keys() or len(p['layers']) != config.layers:
        raise ValueError('model configuration does not match checkpoint layers')
    for name, shape in expected.items():
        if actual[name].shape != shape:
            raise ValueError(f'{name}: expected {shape}, received {actual[name].shape}')


def checkpoint_config(d):
    declared = d['meta'].get('model_config')
    config = ModelConfig(**declared) if declared is not None else SMALL
    validate_parameters(d['parameters'], config)
    return config


def dot(x, w, mixed=True):
    if mixed:
        return j.matmul(x.astype(j.bfloat16), w.astype(j.bfloat16),
            precision=jax.lax.DotAlgorithmPreset.BF16_BF16_F32,
            preferred_element_type=j.float32)
    return j.matmul(x, w, precision=jax.lax.Precision.HIGHEST)


def norm(x, scale):
    return x * jax.lax.rsqrt(j.mean(j.square(x), axis=-1, keepdims=True) + j.float32(1e-5)) * scale


def forward(p, categorical, numeric, legal, mixed=True, config=SMALL):
    batch, width, heads = categorical.shape[0], config.width, config.heads
    x = p['embedding'][categorical].sum(axis=-2) / j.float32(8**.5)
    x = x + dot(numeric, p['numeric'], mixed)
    mons = x[:, 1:61].reshape(batch, 12, 5*width)
    x = j.concatenate((x[:, :1], dot(mons, p['pack'], mixed), x[:, 61:]), axis=1)
    x = x + p['position']
    for layer in p['layers']:
        z = norm(x, layer['norm1'])
        q, k, v = j.split(dot(z, layer['qkv'], mixed), 3, axis=-1)
        q, k, v = [y.reshape(batch, TOKENS, heads, width//heads).transpose(0, 2, 1, 3) for y in (q, k, v)]
        scores = dot(q, k.swapaxes(-2, -1), mixed) * j.float32((width//heads)**-.5)
        a = jax.nn.softmax(scores, axis=-1)
        z = dot(a, v, mixed).transpose(0, 2, 1, 3).reshape(batch, TOKENS, width)
        x = x + dot(z, layer['out'], mixed)
        z = jax.nn.gelu(dot(norm(x, layer['norm2']), layer['up'], mixed), approximate=False)
        x = x + dot(z, layer['down'], mixed)
    x = norm(x, p['final_norm'])
    logits = dot(x[:, -20:], p['policy'], mixed)[..., 0]
    logits = j.where(legal, logits, j.float32(-1e9))
    return logits, dot(x[:, 0], p['value'], mixed)[:, 0], dot(x[:, 0], p['opponent'], mixed)


compiled = jax.jit(forward, static_argnames=('mixed', 'config'))


def configured_forward(config):
    return partial(forward, config=config)
