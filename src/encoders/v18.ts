// SPDX-License-Identifier: GPL-3.0-only
/** X self type-holder attribution; all previous observer versions stay frozen. */
import { View as Previous, vocabulary, actionIndex } from './v17.ts';
export { vocabulary, actionIndex };
const key=(ident: string)=>ident.replace(/^(p[12])[a-z]: /,'$1: ');
export class View extends Previous {
  override add(lines: string[]) {
    for (const line of lines) {
      const f=line.split('|'),tag=f[1];
      const ability=f.find(x=>x.startsWith('[from] ability: '))?.slice(16);
      if(tag==='-start' && f[3]==='typechange' && ['Protean','Libero'].includes(ability||'')) {
        const own=f[2]?.startsWith(this.side)??false;
        const holder=this.opponent.find(p=>p.ident && key(p.ident)===key(f[2]));
        (own?this.ownEffects:holder?.effects)?.add('typechange');
        if(holder)holder.ability=ability;
        this.events.push({tag,entity:0,own,amount:0});this.events=this.events.slice(-16);this.lines.push(line);
      } else if(tag==='-ability' && ability==='Trace') {
        const own=f[2]?.startsWith(this.side)??false;
        const holder=this.opponent.find(p=>p.ident && key(p.ident)===key(f[2]));
        if(holder)holder.ability=f[3];
        this.events.push({tag,entity:0,own,amount:0});this.events=this.events.slice(-16);this.lines.push(line);
      } else super.add([line]);
    }
  }
}
