# SPDX-License-Identifier: GPL-3.0-only
"""CPU integration of native actors, collection, PPO, checkpoint, eval and export."""
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest

os.environ['JAX_PLATFORMS']='cpu'
os.environ['JAX_ENABLE_X64']='true'
os.environ['JAX_DEFAULT_MATMUL_PRECISION']='highest'
os.environ['TSX_DISABLE_CACHE']='1'
import jax
import numpy as np
from selfplay_trainer import model, learner, export
from selfplay_trainer.optimizer import adam_empty
from selfplay_trainer.driver import Workers

ROOT=Path(__file__).resolve().parents[1]


class Workflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='trainer-tests-')
        cls.out=Path(cls.temp.name)
        cls.cli('train','--out',str(cls.out/'run'),'--width','16','--layers','1','--heads','1',*cls.small())
        cls.first=pickle.loads((cls.out/'run/checkpoints/latest.pkl').read_bytes())

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    @staticmethod
    def small():return ['--workers','1','--slots','2','--horizon','4','--minibatch','8','--epochs','1','--max-segments','3','--minutes','5','--warmup-segments','0']

    @staticmethod
    def cli(*args,expect=0):
        result=subprocess.run([sys.executable,'-m','selfplay_trainer','--device','cpu',*args],cwd=ROOT,text=True,capture_output=True,timeout=180)
        if result.returncode!=expect:raise AssertionError(result.stdout+'\n'+result.stderr)
        return result.stdout

    def test_training_and_resume(self):
        summary=json.loads((self.out/'run/summary.json').read_text())
        self.assertTrue(summary['parameters_changed']);self.assertGreater(summary['adam_step_final'],0);self.assertIsNone(summary['error'])
        metrics=[json.loads(x) for x in (self.out/'run/learner.jsonl').read_text().splitlines()]
        self.assertTrue(all(x['finite'] and np.isfinite(x['loss']) for x in metrics))
        self.assertEqual(summary['totals']['steps'],summary['adam_step_final'])
        self.assertGreater(summary['totals']['trained_unique_rows'],0)
        for leaf in jax.tree.leaves(self.first['parameters']):self.assertTrue(np.isfinite(leaf).all())
        original=model.initialize(3412,1337,model.ModelConfig(16,1,1))
        self.assertTrue(any(not np.array_equal(a,b) for a,b in zip(jax.tree.leaves(original),jax.tree.leaves(self.first['parameters']))))
        self.cli('train','--out',str(self.out/'resumed'),'--resume',str(self.out/'run/checkpoints/latest.pkl'),*self.small())
        resumed=pickle.loads((self.out/'resumed/checkpoints/latest.pkl').read_bytes())
        self.assertGreater(int(resumed['adam_step']),int(self.first['adam_step']))
        self.assertGreater(resumed['meta']['segment'],self.first['meta']['segment'])
        self.assertTrue(any(not np.array_equal(a,b) for a,b in zip(jax.tree.leaves(resumed['parameters']),jax.tree.leaves(self.first['parameters']))))
        # Nonzero moments/step survive an exact load/save with no optimizer action.
        p,adam,meta=learner.load_checkpoint(self.out/'run/checkpoints/latest.pkl')
        learner.save_checkpoint(self.out/'copy.pkl',p,adam,meta)
        copy=pickle.loads((self.out/'copy.pkl').read_bytes())
        for key in ('parameters','adam_m','adam_v','adam_step'):
            for a,b in zip(jax.tree.leaves(copy[key]),jax.tree.leaves(self.first[key])):np.testing.assert_array_equal(a,b)

    def test_eval_baselines_and_head_to_head(self):
        checkpoint=str(self.out/'run/checkpoints/latest.pkl')
        for opponent in ('random','heuristic','policy'):
            extra=['--other',checkpoint] if opponent=='policy' else []
            stdout=self.cli('eval','--checkpoint',checkpoint,'--opponent',opponent,'--games','2','--workers','1','--slots','2',*extra)
            result=json.loads(stdout.splitlines()[-1]);self.assertTrue(result['complete']);self.assertGreaterEqual(result['games'],2)
            self.assertEqual(result['capped'],0);self.assertEqual(result['abandoned'],0)
            self.assertEqual(result['games'],result['wins']+result['losses']+result['ties'])

    def test_head_to_head_different_model_dimensions(self):
        config=model.ModelConfig(24,1,3)
        p=model.initialize(3412,23,config)
        learner.save_checkpoint(self.out/'other.pkl',p,adam_empty(p),{'format':'gen9randombattle','model_config':config.to_dict()})
        stdout=self.cli('eval','--checkpoint',str(self.out/'run/checkpoints/latest.pkl'),'--opponent','policy',
                        '--other',str(self.out/'other.pkl'),'--games','2','--workers','1','--slots','2','--deterministic-teams')
        result=json.loads(stdout.splitlines()[-1]);self.assertTrue(result['complete']);self.assertEqual(result['capped'],0)

    def test_odd_slot_evaluation_scores_the_actor_side(self):
        # Both workers have one local slot, so both policies play p1. The fixture
        # gives p1 two natural wins; global lane parity would report a false loss.
        from unittest.mock import patch
        from selfplay_trainer import driver
        class FixtureWorkers:
            def __init__(self,*args,**kwargs):self.reads=0
            def read(self):
                self.reads+=1
                return {'finished':np.full(2,self.reads>1,np.int8),
                        'winner':np.zeros(2,np.int8),'turns':np.ones(2,np.int16)}
            def write(self,actions):pass
            def close(self):pass
        def fixture_policy(*args,**kwargs):return (np.zeros((2,2),np.int32),)
        with patch.object(driver,'Workers',FixtureWorkers), \
             patch.object(driver,'make_policy',return_value=fixture_policy), \
             patch.object(driver,'to_device',return_value=(None,None,None,None)):
            result=driver.evaluate(None,'random',2,seed=17,W=2,G=1,mixed=False)
        self.assertEqual((result['wins'],result['losses'],result['score']),(2,0,1.))

    def test_export_round_trip(self):
        checkpoint=self.out/'run/checkpoints/latest.pkl'
        manifest=export.export(checkpoint,self.out/'export')
        params,loaded=export.load_export(self.out/'export');self.assertEqual(manifest,loaded)
        for a,b in zip(jax.tree.leaves(params),jax.tree.leaves(self.first['parameters'])):np.testing.assert_array_equal(a,b)
        cat=np.zeros((2,97,8),np.int32);num=np.zeros((2,97,32),np.float32);mask=np.ones((2,20),bool)
        config=model.checkpoint_config(self.first)
        a=model.forward(self.first['parameters'],cat,num,mask,False,config)
        b=model.forward(params,cat,num,mask,False,config)
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
        raw=self.out/'export/weights.bin';data=bytearray(raw.read_bytes());data[0]^=1;raw.write_bytes(data)
        with self.assertRaisesRegex(ValueError,'hash mismatch'):export.load_export(self.out/'export')

    def test_v1_export_compatibility(self):
        p=model.initialize(3412,7);adam=adam_empty(p)
        learner.save_checkpoint(self.out/'small.pkl',p,adam,{'format':'gen9randombattle'})
        export.export(self.out/'small.pkl',self.out/'v1','v1');q,m=export.load_export(self.out/'v1')
        self.assertEqual(m['schema'],'psai-native-compact49-v1')
        for a,b in zip(jax.tree.leaves(p),jax.tree.leaves(q)):np.testing.assert_array_equal(a,b)

    def test_resume_rejects_dimension_mismatch_before_output(self):
        result=subprocess.run([sys.executable,'-m','selfplay_trainer','--device','cpu','train','--out',str(self.out/'bad'),
                               '--resume',str(self.out/'run/checkpoints/latest.pkl'),'--width','32'],cwd=ROOT,text=True,capture_output=True,timeout=30)
        self.assertNotEqual(result.returncode,0);self.assertIn('dimensions differ',result.stderr)
        self.assertFalse((self.out/'bad').exists())

    def test_actor_determinism_and_legal_frames(self):
        a=Workers(1,2,919,deterministic_teams=True);b=Workers(1,2,919,deterministic_teams=True)
        try:
            for _ in range(80):
                fa=a.read();fb=b.read()
                for key in fa:np.testing.assert_array_equal(fa[key],fb[key],err_msg=key)
                active=fa['actor'].astype(bool)
                self.assertTrue(fa['mask'][active].any(-1).all())
                choices=np.where(active,fa['mask'].argmax(-1),-1)
                a.write(choices);b.write(choices)
        finally:a.close();b.close()

    def test_ou_training_and_periodic_eval(self):
        team='Pikachu\nAbility: Static\nLevel: 100\nEVs: 4 Def / 252 SpA / 252 Spe\n- Thunderbolt\n- Quick Attack\n'
        (self.out/'team.txt').write_text(team)
        subprocess.run(['node','--import','tsx','src/teams.ts','--input',str(self.out/'team.txt'),'--out',str(self.out/'pool.txt')],cwd=ROOT,check=True,capture_output=True,timeout=30)
        self.cli('train','--format','gen9ou','--out',str(self.out/'ou'),'--teams',str(self.out/'pool.txt'),'--eval-teams',str(self.out/'pool.txt'),
                 '--width','16','--layers','1','--heads','1',*self.small(),'--eval-games','2','--eval-at-start','--eval-opponents','random,heuristic,policy','--eval-workers','1','--eval-slots','2')
        s=json.loads((self.out/'ou/summary.json').read_text());self.assertTrue(s['parameters_changed']);self.assertGreater(s['totals']['games'],0)
        records=[json.loads(line) for line in (self.out/'ou/evals.jsonl').read_text().splitlines()]
        self.assertEqual(len(records),3);self.assertTrue(all(r['games']>=2 and r['capped']==r['abandoned']==0 for r in records))

    def test_supervisor_start_resume_and_stop(self):
        import time
        out=self.out/'supervised'
        def invoke(action,*extra):
            return subprocess.run([sys.executable,'-m','selfplay_trainer.supervise',action,'--out',str(out),*extra],cwd=ROOT,text=True,capture_output=True,timeout=20)
        args=['--hours','.1','--','--device','cpu','train','--width','16','--layers','1','--heads','1',*self.small()]
        first=invoke('start',*args);self.assertEqual(first.returncode,0,first.stdout+first.stderr)
        def stopped():
            for _ in range(90):
                state=json.loads((out/'supervisor.json').read_text())
                if state['status']!='running':return state
                time.sleep(.2)
            self.fail('supervisor did not stop')
        state=stopped();self.assertEqual(state['exit_code'],0)
        initial=pickle.loads((out/'checkpoints/latest.pkl').read_bytes())
        second=invoke('start',*args);self.assertEqual(second.returncode,0,second.stdout+second.stderr)
        resumed=stopped();self.assertEqual(resumed['deadline_epoch'],state['deadline_epoch'])
        final=pickle.loads((out/'checkpoints/latest.pkl').read_bytes());self.assertGreater(int(final['adam_step']),int(initial['adam_step']))
        # Start a long owned child, then exercise its STOP boundary.
        longargs=list(args)
        at=longargs.index('--max-segments');longargs[at+1]='1000000'
        self.assertEqual(invoke('start',*longargs).returncode,0)
        self.assertEqual(invoke('stop').returncode,0)
        self.assertEqual(stopped()['status'],'stopped')

    def test_failure_does_not_hang_or_publish(self):
        env=dict(os.environ,JAX_NATIVE_TEST_LEARNER_FAIL='1')
        result=subprocess.run([sys.executable,'-m','selfplay_trainer','--device','cpu','train','--out',str(self.out/'failure'),
                               '--width','16','--layers','1','--heads','1',*self.small()],cwd=ROOT,env=env,text=True,capture_output=True,timeout=30)
        self.assertEqual(result.returncode,3,result.stdout+result.stderr)
        self.assertFalse((self.out/'failure/checkpoints/latest.pkl').exists())

    def test_supervisor_relative_inputs_from_parent_directory(self):
        import time
        pool=self.out/'relative-pool.txt'
        team='Pikachu\nAbility: Static\nLevel: 100\nEVs: 4 Def / 252 SpA / 252 Spe\n- Thunderbolt\n- Quick Attack\n'
        (self.out/'relative-team.txt').write_text(team)
        subprocess.run(['node','--import','tsx','src/teams.ts','--input',str(self.out/'relative-team.txt'),'--out',str(pool)],
                       cwd=ROOT,check=True,capture_output=True,timeout=30)
        args=[sys.executable,'-m','selfplay_trainer.supervise','start','--out','relative-run','--hours','.1','--',
              '--device','cpu','train','--format','gen9ou','--teams','relative-pool.txt','--eval-teams=relative-pool.txt',
              '--init=run/checkpoints/latest.pkl','--baseline','run/checkpoints/latest.pkl',
              *self.small(),'--eval-games','2','--eval-at-start','--eval-opponents','policy','--eval-workers','1','--eval-slots','2']
        result=subprocess.run(args,cwd=self.out,text=True,capture_output=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        out=self.out/'relative-run'
        try:
            for _ in range(900):
                state=json.loads((out/'supervisor.json').read_text())
                if state['status']!='running':break
                time.sleep(.2)
            self.assertEqual(state['status'],'stopped',(out/'train.log').read_text())
            self.assertEqual(state['exit_code'],0,(out/'train.log').read_text())
            summary=json.loads((out/'summary.json').read_text())
            self.assertTrue(summary['parameters_changed'])
            self.assertEqual(summary['args']['teams'],str(pool))
            self.assertEqual(summary['args']['init'],str(self.out/'run/checkpoints/latest.pkl'))
            self.assertTrue((out/'checkpoints/latest.pkl').is_file())
            self.assertFalse((ROOT/'relative-run').exists())
        finally:
            # Terminate only the supervisor created by this test, on failure too.
            (out/'STOP').write_text('test cleanup\n')


if __name__=='__main__':unittest.main()
