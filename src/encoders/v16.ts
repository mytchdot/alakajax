// SPDX-License-Identifier: GPL-3.0-only
/** U public-holder attribution; original/v14 and qualified v15 stay frozen. */
import { View as Previous, vocabulary, actionIndex } from './v15.ts';
export { vocabulary, actionIndex };
const key = (ident: string) => ident.replace(/^(p[12])[a-z]: /, '$1: ');
export class View extends Previous {
  override add(lines: string[]) {
    for (const line of lines) {
      const f=line.split('|'), tag=f[1];
      const attributedStart=tag==='-start' && ((f[3]==='confusion' && f.includes('[from] ability: Poison Puppeteer')) || (f[3]==='Attract' && f.includes('[from] ability: Cute Charm')));
      if (attributedStart) {
        const source=f.find(x=>x.startsWith('[of] '))?.slice(5);
        if (!source) throw Error('Missing reactive ability holder');
        const target=this.opponent.find(p => p.ident && key(p.ident)===key(f[2]));
        const own=f[2]?.startsWith(this.side) ?? false;
        (own?this.ownEffects:target?.effects)?.add(f[3].toLowerCase());
        const holder=this.opponent.find(p=>p.ident && key(p.ident)===key(source));
        if (holder) holder.ability=f[3]==='Attract'?'Cute Charm':'Poison Puppeteer';
        this.events.push({tag,entity:0,own,amount:Number(f[4]) || 0});
        this.events=this.events.slice(-16);this.lines.push(line);
      } else {
        super.add([line]);
        if ((tag==='-activate' && f[3]==='ability: Synchronize') || (tag==='-block' && f[3]==='ability: Flower Veil')) {
          const holder=this.opponent.find(p => p.ident && key(p.ident)===key(f[2]));
          if (holder) holder.ability=f[3].slice(9);
        }
      }
    }
  }
}
