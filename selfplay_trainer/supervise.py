# SPDX-License-Identifier: GPL-3.0-only
"""Start, stop and inspect an owned training run with bounded automatic resume."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]


def resolve_training_paths(args, cwd):
    """Keep file arguments relative to the caller when the child changes directory."""
    options={'--teams','--eval-teams','--featured','--init','--baseline'}
    result=list(args)
    for i,argument in enumerate(args):
        name,separator,value=argument.partition('=')
        if name not in options:
            continue
        if separator:
            result[i]=name+'='+str((cwd/value).resolve())
        elif i+1<len(args):
            result[i+1]=str((cwd/args[i+1]).resolve())
    return result


def write_state(out,**values):
    tmp=out/'supervisor.tmp';tmp.write_text(json.dumps(values,indent=2)+'\n');tmp.replace(out/'supervisor.json')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('start','run','stop','status'))
    p.add_argument('--out',required=True);p.add_argument('--hours',type=float,default=24.)
    p.add_argument('--max-restarts',type=int,default=3);p.add_argument('--token',default=None)
    argv=list(sys.argv[1:] if argv is None else argv)
    at=argv.index('--') if '--' in argv else len(argv)
    training_args=argv[at+1:];a=p.parse_args(argv[:at]);a.training_args=training_args;out=Path(a.out).resolve();state=out/'supervisor.json'
    if a.action=='status':
        print(state.read_text() if state.exists() else json.dumps({'status':'not started'}));return
    out.mkdir(parents=True,exist_ok=True)
    if a.action=='stop':
        (out/'STOP').write_text('stop requested\n');print('STOP requested');return
    if a.hours<=0 or a.max_restarts<0:p.error('positive hours and nonnegative max-restarts required')
    args=a.training_args
    if args[:1]==['--']:args=args[1:]
    if 'train' not in args:p.error('supply -- [--device cpu] train TRAIN_OPTIONS')
    if any(x.split('=',1)[0] in ('--out','--resume') for x in args):p.error('supervisor supplies --out and --resume')
    args=resolve_training_paths(args,Path.cwd())
    # An OS advisory lock, held across the complete run, prevents duplicate supervisors.
    if a.action=='start':
        if (out/'STOP').exists():p.error('remove this run\'s STOP file before resuming')
        token=uuid.uuid4().hex
        cmd=[sys.executable,'-m','selfplay_trainer.supervise','run','--out',str(out),'--hours',str(a.hours),'--max-restarts',str(a.max_restarts),'--token',token,'--',*args]
        with (out/'supervisor.log').open('ab') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=log,start_new_session=True)
        # run owns all state writes; wait for admission or its explicit failure.
        for _ in range(100):
            if state.exists() and json.loads(state.read_text()).get('token')==token:
                print(state.read_text());return
            if child.poll() is not None:raise SystemExit('supervisor did not start; see supervisor.log')
            time.sleep(.05)
        raise SystemExit('supervisor admission timed out; inspect supervisor.log')
    import fcntl
    with (out/'supervisor.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('this output already has a supervisor')
        previous=json.loads(state.read_text()) if state.exists() else {}
        # Explicitly starting again resumes the original deadline; no hidden reset.
        deadline=previous.get('deadline_epoch',time.time()+a.hours*3600)
        stop=False
        def signal_stop(*_):
            nonlocal stop
            stop=True
        signal.signal(signal.SIGTERM,signal_stop);signal.signal(signal.SIGINT,signal_stop)
        restarts=0;child=None;exitcode=None
        common=dict(token=a.token,pid=os.getpid(),deadline_epoch=deadline)
        try:
            while not stop and not (out/'STOP').exists() and time.time()<deadline:
                ckpt=out/'checkpoints/latest.pkl'
                cmd=[sys.executable,'-m','selfplay_trainer',*args,'--out',str(out),'--minutes',str(max(.001,(deadline-time.time())/60))]
                if ckpt.exists():cmd+=['--resume',str(ckpt)]
                with (out/'train.log').open('ab') as log:
                    child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=log,start_new_session=True)
                write_state(out,**common,status='running',child_pid=child.pid,restarts=restarts)
                while child.poll() is None:
                    if stop or (out/'STOP').exists() or time.time()>=deadline:
                        child.terminate()
                        try:child.wait(timeout=180)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid,signal.SIGKILL);child.wait()
                        break
                    time.sleep(1)
                exitcode=child.returncode
                if exitcode==0 or stop or (out/'STOP').exists():break
                restarts+=1
                if restarts>a.max_restarts:break
                time.sleep(1)
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                try:child.wait(timeout=180)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            write_state(out,**common,status='stopped' if exitcode in (0,None) or stop or (out/'STOP').exists() else 'failed',exit_code=exitcode,restarts=restarts)
        if exitcode not in (0,None) and not stop and not (out/'STOP').exists():raise SystemExit(exitcode)

if __name__=='__main__':main()
