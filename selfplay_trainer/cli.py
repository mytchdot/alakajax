# SPDX-License-Identifier: GPL-3.0-only
"""Stable train/eval/export/status commands; device selection precedes JAX import."""
import argparse
import json
import os
from pathlib import Path
import pickle
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def metadata():
    import shutil
    raw = subprocess.check_output([shutil.which('node') or 'node','--import','tsx',str(ROOT/'src/metadata.ts')], cwd=ROOT,
                                 env=dict(os.environ, TSX_DISABLE_CACHE='1'),text=True)
    return json.loads(raw)


def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',choices=('cpu','gpu','auto'),default='auto')
    sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train',help='native self-play and asynchronous PPO')
    t.add_argument('--out',required=True)
    t.add_argument('--format',choices=('gen9randombattle','gen9ou'),default='gen9randombattle')
    for name in ('teams','eval-teams','featured','init','baseline','resume'):
        t.add_argument('--'+name)
    t.add_argument('--featured-share',type=float,default=.25)
    for name,default in [('workers',10),('slots',128),('horizon',16),('minibatch',2048),('epochs',2),('max-segments',10**9),('warmup-segments',2),('seed',1337),('eval-games',0),('eval-workers',4),('eval-slots',32)]:
        t.add_argument('--'+name,type=int,default=default)
    for name,default in [('minutes',60.),('lam',.95),('checkpoint-every-s',900.),('snapshot-every-s',21600.),('eval-every-s',14400.)]:
        t.add_argument('--'+name,type=float,default=default)
    for name in ('width','layers','heads'):
        t.add_argument('--'+name,type=int)
    t.add_argument('--precision',choices=('bf16','fp32'),default=None)
    t.add_argument('--deterministic-teams',action='store_true',help='seed Random Battles team generation as well as battle RNG')
    t.add_argument('--eval-at-start',action='store_true')
    t.add_argument('--eval-opponents',default='heuristic,random,policy')
    e=sub.add_parser('eval',help='evaluate against baselines or an earlier checkpoint')
    e.add_argument('--checkpoint',required=True);e.add_argument('--opponent',choices=('random','heuristic','policy'),default='heuristic')
    e.add_argument('--other');e.add_argument('--teams');e.add_argument('--format',choices=('gen9randombattle','gen9ou'))
    e.add_argument('--games',type=int,default=500);e.add_argument('--workers',type=int,default=4);e.add_argument('--slots',type=int,default=32)
    e.add_argument('--seed',type=int,default=7331);e.add_argument('--timeout-s',type=float,default=1800.);e.add_argument('--output')
    e.add_argument('--deterministic-teams',action='store_true')
    e.add_argument('--precision',choices=('bf16','fp32'),default=None)
    x=sub.add_parser('export',help='portable FP32 weights and dimension-tagged manifest')
    x.add_argument('--checkpoint',required=True);x.add_argument('--out',required=True);x.add_argument('--schema',choices=('v1','v2'),default='v2')
    st=sub.add_parser('status');st.add_argument('--out',required=True)
    return p


def main(argv=None):
    parser=build_parser();a=parser.parse_args(argv)
    # Preserve the source launcher's numeric and allocator defaults without its device identity.
    for key,value in {'JAX_ENABLE_X64':'true','JAX_DEFAULT_MATMUL_PRECISION':'highest','NVIDIA_TF32_OVERRIDE':'0',
                      'XLA_PYTHON_CLIENT_PREALLOCATE':'false','XLA_PYTHON_CLIENT_MEM_FRACTION':'0.85',
                      'OMP_NUM_THREADS':'2','TSX_DISABLE_CACHE':'1'}.items():
        os.environ.setdefault(key,value)
    os.environ.setdefault('XLA_FLAGS','--xla_gpu_autotune_level=4 --xla_backend_optimization_level=3 --xla_gpu_enable_fast_min_max=false')
    if a.device != 'auto': os.environ['JAX_PLATFORMS']='cpu' if a.device=='cpu' else 'cuda'
    if a.command=='status':
        out=Path(a.out)
        for name in ('supervisor.json','summary.json'):
            path=out/name
            if path.exists():print(path.name, path.read_text())
        path=out/'segments.jsonl'
        if path.exists():
            with path.open('rb') as f:
                f.seek(max(0,path.stat().st_size-16384));lines=f.read().splitlines()
            if lines: print('latest_segment',lines[-1].decode())
        return
    import jax
    from . import model, driver
    from . import export as portable
    if a.command=='export':
        print(json.dumps(portable.export(a.checkpoint,a.out,a.schema)));return
    if a.command=='eval':
        with open(a.checkpoint,'rb') as f:d=pickle.load(f)
        config=model.checkpoint_config(d);fmt=a.format or d['meta'].get('format','gen9randombattle')
        if a.games<1 or a.workers<1 or not 1<=a.slots<=512:parser.error('positive games/workers and slots in [1,512] required')
        if fmt!='gen9randombattle' and not a.teams:parser.error('OU evaluation needs --teams')
        if a.opponent=='policy' and not a.other:parser.error('policy evaluation needs --other')
        p_b=driver.load_params(a.other) if a.other else None
        other_config=driver.checkpoint_model_config(a.other) if a.other else config
        mixed=(a.precision=='bf16') if a.precision else jax.devices()[0].platform!='cpu'
        result=driver.evaluate(jax.tree.map(jax.numpy.asarray,d['parameters']),a.opponent,a.games,seed=a.seed,teams=a.teams,fmt=fmt,
                               W=a.workers,G=a.slots,p_b=p_b,timeout_s=a.timeout_s,config=config,opponent_config=other_config,mixed=mixed,deterministic_teams=a.deterministic_teams)
        result['requested_games']=a.games;result['complete']=result['games']>=a.games
        if a.output:
            path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result))
        if not result['complete']:raise SystemExit(2)
        return
    for name in ('workers','slots','horizon','minibatch','epochs','max_segments'):
        if getattr(a,name)<1:parser.error(name+' must be positive')
    if a.slots>512:parser.error('slots must not exceed 512')
    if not 0<=a.lam<=1 or not 0<=a.featured_share<=1:parser.error('lam and featured-share must be in [0,1]')
    if a.minutes<=0:parser.error('minutes must be positive')
    if a.format=='gen9ou' and (not a.teams or (a.eval_games and not a.eval_teams)):parser.error('OU needs --teams and --eval-teams when evaluation is enabled')
    for key in ('teams','eval_teams','featured','init','resume','baseline'):
        path=getattr(a,key)
        if path:
            if not Path(path).is_file():parser.error(key+' file does not exist')
            setattr(a,key,str(Path(path).resolve()))
    declared=None
    for path in (a.resume,a.init if not a.resume else None):
        if path:declared=driver.checkpoint_model_config(path)
    defaults=declared or model.SMALL
    a.model_config=model.ModelConfig(a.width or defaults.width,a.layers or defaults.layers,a.heads or defaults.heads)
    if declared and a.model_config!=declared:parser.error('requested dimensions differ from checkpoint')
    v=metadata();a.vocabulary_size=v['size'];a.vocabulary_sha256=v['sha256']
    for path in (a.resume,a.init,a.baseline):
        if path:
            with open(path,'rb') as f:d=pickle.load(f)
            model.checkpoint_config(d)
            if d['parameters']['embedding'].shape[0]!=a.vocabulary_size:parser.error('checkpoint vocabulary size differs')
            if d['meta'].get('vocabulary_sha256',a.vocabulary_sha256)!=a.vocabulary_sha256:parser.error('checkpoint vocabulary hash differs')
    a.mixed=(a.precision=='bf16') if a.precision else jax.devices()[0].platform!='cpu'
    a.out=str(Path(a.out).resolve())
    del a.command,a.device
    driver.train(a)
