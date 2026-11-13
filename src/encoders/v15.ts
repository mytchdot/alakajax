// SPDX-License-Identifier: GPL-3.0-only
/** Explicit current-own-form extension. v14/OBS1 remains byte-for-byte frozen.
 * Only a player's private request and extracted public channel enter this view.
 * Persistent request details stay intact; current form affects encoding only.
 */
import { View as Previous, vocabulary, actionIndex } from './v2.ts';
export { vocabulary, actionIndex };
const key = (ident: string) => ident.replace(/^(p[12])[a-z]: /, '$1: ');
export class View extends Previous {
  ownCurrentForm = new Map<string, string>();
  ownActiveIdent = '';
  override add(lines: string[]) {
    for (const line of lines) {
      const f=line.split('|'), tag=f[1];
      const own=f[2]?.startsWith(this.side) ?? false;
      if (own && ['switch','drag'].includes(tag)) {
        if (this.ownActiveIdent) this.ownCurrentForm.delete(this.ownActiveIdent);
        this.ownActiveIdent=key(f[2]);
        this.ownCurrentForm.set(this.ownActiveIdent,f[3].split(',')[0]);
      }
      if (own && ['detailschange','-formechange','replace'].includes(tag))
        this.ownCurrentForm.set(key(f[2]),f[3].split(',')[0]);
      if (own && tag==='faint') this.ownCurrentForm.delete(key(f[2]));
      if (['detailschange','-formechange'].includes(tag)) {
        // v14 deliberately rejects new attribution effects. Handle this new
        // family here without rewriting a protocol line or altering v14 bytes.
        const other=this.side==='p1'?'p2':'p1';
        const target=f[2]?.startsWith(other) ? this.opponent.find(p => p.ident && key(p.ident)===key(f[2])) : undefined;
        if (target) target.species=f[3].split(',')[0];
        for (const suffix of f.slice(4)) if (suffix.startsWith('[from] ability: ')) {
          const ability=suffix.slice(16);
          if (!['Hunger Switch','Shields Down'].includes(ability)) throw Error('Unsupported form attribution: '+ability);
          if (target) target.ability=ability;
        }
        this.lines.push(line);
      } else if (tag==='-damage' && f.includes('[from] ability: Gulp Missile')) {
        const holder=f.find(x => x.startsWith('[of] '))?.slice(5);
        if (!holder) throw Error('Missing Gulp Missile holder');
        const victim=this.opponent.find(p => p.ident && key(p.ident)===key(f[2]));
        if (victim) victim.condition=f[3];
        const source=this.opponent.find(p => p.ident && key(p.ident)===key(holder));
        if (source) source.ability='Gulp Missile';
        this.events.push({tag,entity:0,own,amount:Number(f[4]) || 0});
        this.events=this.events.slice(-16);this.lines.push(line);
      } else {
        super.add([line]);
        if (!own && tag==='-activate' && ['ability: Disguise','ability: Ice Face','ability: Zero to Hero','ability: Gulp Missile'].includes(f[3])) {
          const target=this.opponent.find(p => p.ident && key(p.ident)===key(f[2]));
          if (target) target.ability=f[3].slice(9);
        }
      }
    }
  }
  override encode() {
    const request=this.request;
    if (!request?.side) return super.encode();
    const projected=structuredClone(request);
    for (const p of projected.side!.pokemon) {
      const current=this.ownCurrentForm.get(key(p.ident));
      if (current) p.details=[current,...p.details.split(',').slice(1)].join(',');
    }
    this.request=projected;
    try {return super.encode();}
    finally {this.request=request;}
  }
}
