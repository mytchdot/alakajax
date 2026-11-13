# SPDX-License-Identifier: GPL-3.0-only
"""Portable parameter-only export; tensor offsets count little-endian FP32 elements."""
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
from .model import SMALL, ModelConfig, checkpoint_config, named_tensors, validate_parameters

VOCABULARY_SHA256 = '0aba5bda3a8dbd243b9f0f0f51689b11cb727a9f44a6807a76a2b95d397c8fc7'


def export(checkpoint, output, schema='v2'):
    with open(checkpoint,'rb') as f:d=pickle.load(f)
    config=checkpoint_config(d)
    if schema=='v1' and config!=SMALL:raise ValueError('v1 requires the 128/4/4 model')
    entries=[];blobs=[];offset=0
    for name,value in named_tensors(d['parameters']):
        x=np.ascontiguousarray(np.asarray(value,dtype='<f4'))
        if not np.isfinite(x).all():raise ValueError('nonfinite tensor '+name)
        entries.append(dict(name=name,shape=list(x.shape),offset=offset));blobs.append(x.tobytes());offset+=x.size
    raw=b''.join(blobs)
    manifest=dict(schema='psai-native-compact49-v1' if schema=='v1' else 'psai-native-transformer-v2',
                  model_config=config.to_dict(),architecture='compact49 transformer; exact-erf GELU; BF16 numeric inputs',
                  vocabulary_sha256=d['meta'].get('vocabulary_sha256',VOCABULARY_SHA256),
                  vocabulary_size=int(d['parameters']['embedding'].shape[0]),weights_sha256=hashlib.sha256(raw).hexdigest(),
                  format=d['meta'].get('format','gen9randombattle'),tensors=entries)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/'weights.bin').write_bytes(raw);(output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def load_export(directory):
    directory=Path(directory);m=json.loads((directory/'manifest.json').read_text());raw=(directory/'weights.bin').read_bytes()
    if m['schema'] not in ('psai-native-compact49-v1','psai-native-transformer-v2'):raise ValueError('unsupported export schema')
    if hashlib.sha256(raw).hexdigest()!=m['weights_sha256']:raise ValueError('weights hash mismatch')
    p={};layers={};next_offset=0
    for t in m['tensors']:
        if t['offset']!=next_offset:raise ValueError('noncontiguous tensor offsets')
        n=int(np.prod(t['shape']));x=np.frombuffer(raw,dtype='<f4',count=n,offset=t['offset']*4).reshape(t['shape']).copy()
        if not np.isfinite(x).all():raise ValueError('nonfinite export')
        next_offset+=n;name=t['name']
        if name.startswith('layers.'):
            _,i,k=name.split('.');layers.setdefault(int(i),{})[k]=x
        else:p[name]=x
    if next_offset*4!=len(raw):raise ValueError('unexpected trailing weights')
    p['layers']=tuple(layers[i] for i in range(len(layers)))
    validate_parameters(p,ModelConfig(**m.get('model_config',SMALL.to_dict())))
    return p,m
