# SPDX-License-Identifier: GPL-3.0-only
"""Same PPO objective; explicit versioned model backward only."""
import jax
import jax.numpy as j
from .model import forward, SMALL

from .branches import branches

def loss(p,mixed_batch,mixed=True,config=SMALL):
    b=mixed_batch
    logits,value,aux=forward(p,b['categorical'],b['numeric'],b['legal'],mixed,config)
    logps=jax.nn.log_softmax(logits,-1)
    logp=j.take_along_axis(logps,b['action'][:,None],-1)[:,0]
    weights=b['eligible'].astype(j.float32);n=j.maximum(weights.sum(),1)
    a=b['advantage'];mean=(a*weights).sum()/n
    scale=jax.lax.rsqrt(((a-mean)**2*weights).sum()/n+j.float32(1e-8));a=(a-mean)*scale
    ratio=j.exp(logp-b['old_logp'])
    policy=-(j.minimum(ratio*a,j.clip(ratio,.8,1.2)*a)*weights).sum()/n
    critic=(((value-b['returns'])**2)*weights).sum()/n
    entropy=-(j.exp(logps)*logps).sum(-1)
    entropy=(entropy*weights).sum()/n
    # Auxiliary head is executed and finite-checked, but has no fabricated target.
    total=policy+j.float32(.5)*critic-j.float32(.01)*entropy
    return total,dict(policy=policy,value=critic,entropy=entropy,
        KL=((b['old_logp']-logp)*weights).sum()/n,
        clip_fraction=((j.abs(ratio-1)>.2)*weights).sum()/n,
        aux_finite=j.isfinite(aux).all(),eligible=weights.sum(),
        train_logits=jax.lax.stop_gradient(logits),
        train_values=jax.lax.stop_gradient(value),
        train_aux=jax.lax.stop_gradient(aux),
        source_logp=jax.lax.stop_gradient(logp),
        source_ratio=jax.lax.stop_gradient(ratio),
        source_advantage=jax.lax.stop_gradient(a),
        source_branch=jax.lax.stop_gradient(branches(a,ratio)))
