// SPDX-License-Identifier: GPL-3.0-only
import test from 'node:test';
import assert from 'node:assert/strict';
import { ViewV21, ViewV22, vocabulary, actionIndex, legalActions } from '../src/index.ts';
import { trajectory } from './trajectory.ts';

test('vocabulary and action mapping remain stable',()=>{
 assert.equal(vocabulary.sha256,'0aba5bda3a8dbd243b9f0f0f51689b11cb727a9f44a6807a76a2b95d397c8fc7');
 assert.equal(vocabulary.names.length,3412);
 const req:any={teamPreview:true,side:{id:'p1',pokemon:[{details:'Pikachu',condition:'100/100'},{details:'Ditto',condition:'100/100'}]}};
 assert.deepEqual(legalActions(req).map(actionIndex),[14,15]);assert.equal(legalActions({...req,wait:true}).length,0);
});

test('v21 fixed native trajectory matches the reference digest',()=>{
 const result=trajectory(ViewV21,'gen9randombattle');
 assert.equal(result.sha256,'496f8efa4856babd1d70045c49c82dac6a2c47fa5c45efbb2713273b224de4b0');assert.equal(result.skipped,0);
});

test('v22 fixed native OU trajectory matches the reference digest',()=>{
 const result=trajectory(ViewV22,'gen9ou');
 assert.equal(result.sha256,'9d92fee04f51a189cec5b47120dd72f7cfabc0f2cc41d26e2c4dbb625ae1d2d4');assert.equal(result.skipped,0);
});

test('v21 and v22 agree on a fixed Random Battles trajectory',()=>{
 const a=trajectory(ViewV21,'gen9randombattle',41,1),b=trajectory(ViewV22,'gen9randombattle',41,1);
 assert.equal(a.sha256,b.sha256);assert.equal(a.decisions,b.decisions);
});
