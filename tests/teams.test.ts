// SPDX-License-Identifier: GPL-3.0-only
import test from 'node:test';
import assert from 'node:assert/strict';
import { validateAndPack, splitPool } from '../src/teams.ts';

test('supplied synthetic team validates, packs and deduplicates',()=>{
 const text='Pikachu\nAbility: Static\nLevel: 100\nEVs: 4 Def / 252 SpA / 252 Spe\n- Thunderbolt\n- Quick Attack\n';
 const packed=validateAndPack(text);assert.equal(packed.length,1);
 assert.deepEqual(validateAndPack(packed[0]+'\n'+packed[0]),packed);
 assert.throws(()=>validateAndPack('Unknown species\n- Unknown move'));
 const split=splitPool(['a','b','c','d'],.5,'17');
 assert.equal(split.evaluation.length,2);assert.equal(split.train.length,2);
 assert.deepEqual(splitPool(['d','c','b','a'],.5,'17'),split);
 assert.ok(split.train.every(t=>!split.evaluation.includes(t)));
});
