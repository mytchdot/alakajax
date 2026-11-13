// SPDX-License-Identifier: GPL-3.0-only
/** Synthetic seeded battles, with each View receiving only its own channel. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import Showdown from 'pokemon-showdown';
import BattleModule from 'pokemon-showdown/dist/sim/battle.js';
const { Battle, Teams } = Showdown as any;
const { extractChannelMessages } = BattleModule as any;
export function trajectory(View: any, format: string, seed = 17, games = 2) {
  const hash=createHash('sha256');let decisions=0,skipped=0;const logs:any[]=[];
  for(let g=0;g<games;g++) {
    const views:any={p1:new View('p1'),p2:new View('p2')};const pending=new Set<string>();let ended=false;
    const feed=(side:string,lines:string[])=>{
      logs.push({game:g,side,lines});
      for(const line of lines){
        try {views[side].add([line]);}catch(error){skipped++;hash.update(String(error));}
        if(line.startsWith('|request|')){const r=JSON.parse(line.slice(9));if(r.wait)pending.delete(side);else if(r.side)pending.add(side);}
        if(line.startsWith('|win|')||line==='|tie')ended=true;
      }
    };
    const battle=new Battle({formatid:format,seed:[seed+g,223,341,419].join(','),send:(type:string,raw:any)=>{
      const text=Array.isArray(raw)?raw.join('\n'):raw;
      if(type==='update'){const m=extractChannelMessages(text,[1,2]);feed('p1',m[1]);feed('p2',m[2]);}
      if(type==='sideupdate'){const i=text.indexOf('\n');feed(text.slice(0,i),text.slice(i+1).split('\n'));}
    }});
    // Synthetic one-Pokemon teams are built at test time; no team pool is shipped.
    const team=Teams.pack([{species:'Pikachu',name:'Pikachu',ability:'Static',moves:['Thunderbolt','Quick Attack'],nature:'Timid',level:100}]);
    for(const side of ['p1','p2'])battle.setPlayer(side,{name:side,...(format==='gen9ou'?{team}:{seed:[seed+g,223,341,side==='p1'?420:421].join(',')})});
    battle.sendUpdates();
    try {
      for(let phase=0;phase<3500&&!ended;phase++){
        const choices:any[]=[];
        for(const side of ['p1','p2'])if(pending.has(side)){
          const view=views[side],encoded=view.encode();hash.update(JSON.stringify(encoded));decisions++;
          assert.equal(encoded.categorical.length,97);assert.equal(encoded.numeric.length,97);assert.equal(encoded.mask.length,20);
          const actions=view.actions();const action=actions[(phase*17+g*3)%actions.length];choices.push([side,action.id]);
        }
        pending.clear();assert.ok(choices.length,'a live game must request an action');
        for(const [side,choice] of choices)battle.choose(side,choice);battle.sendUpdates();
      }
      assert.ok(ended,'the battle must finish naturally');
    } finally {battle.destroy();}
  }
  return {sha256:hash.digest('hex'),games,decisions,skipped,logs};
}
