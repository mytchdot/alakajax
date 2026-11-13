// SPDX-License-Identifier: GPL-3.0-only
/** Virtual-copy knowledge uses only this player's extracted stream and request.
 * -transform has no token in the protected vocabulary; its causal protocol line
 * is retained, while species/move/ability/boost knowledge updates explicitly.
 */
import { View as Previous, vocabulary, actionIndex } from './v18.ts';
export { vocabulary, actionIndex };
const key=(ident:string)=>ident.replace(/^(p[12])[a-z]: /,'$1: ');
import Showdown from 'pokemon-showdown';
const { Dex } = Showdown;
const id=(s:string)=>s.toLowerCase().replace(/[^a-z0-9]/g,'');
export class View extends Previous {
  copyOriginal=new Map<string,{species:string,moves:string[],ability?:string}>();
  copyLinks=new Map<string,{target:string,own:boolean,species:string,pending:boolean}>();
  override add(lines:string[]) {
    for(const line of lines) {
      const f=line.split('|'),tag=f[1],own=f[2]?.startsWith(this.side)??false;
      const source=key(f[2]||'');
      if(tag==='-transform') {
        const targetKey=key(f[3]);
        const target=this.opponent.find(p=>p.ident && key(p.ident)===targetKey);
        const holder=this.opponent.find(p=>p.ident && key(p.ident)===source);
        const mine=this.request?.side?.pokemon.find(p=>key(p.ident)===targetKey);
        const species=own?target?.species:(this.ownCurrentForm.get(targetKey)||mine?.details.split(',')[0]);
        if(!species)throw Error('Copy target has no independently disclosed current species');
        this.copyLinks.set(source,{target:targetKey,own,species,pending:true});
        if(own) {
          this.ownCurrentForm.set(source,species);this.ownBoosts={...(target?.boosts||{})};
        } else {
          if(!holder)throw Error('Copy holder has no public identity');
          this.copyOriginal.set(source,{species:holder.species,moves:[...holder.moves],ability:f.includes('[from] ability: Imposter')?'Imposter':holder.ability});
          holder.species=species;holder.boosts={...this.ownBoosts};holder.moves=[];holder.ability=undefined;
          if(mine){holder.moves=mine.moves.map(m=>Dex.moves.get(m).name);holder.ability=Dex.abilities.get(mine.ability).name;}
        }
        this.lines.push(line);continue;
      }
      if((tag==='switch'||tag==='drag') && !own) {
        // A live outgoing copier is cleared before installing another public mon.
        for(const p of this.opponent.filter(p=>p.active && p.ident))this.clearCopy(key(p.ident!));
      }
      if(tag==='faint')this.clearCopy(source);
      if((tag==='switch'||tag==='drag') && own) {
        if(this.ownActiveIdent)this.copyLinks.delete(this.ownActiveIdent);
      }
      super.add([line]);
      if(tag==='request' && this.request?.side) {
        for(const [copier,link] of this.copyLinks) {
          if(!link.pending)continue;
          const member=this.request.side.pokemon.find(p=>key(p.ident)===(link.own?copier:link.target));
          const opponent=this.opponent.find(p=>p.ident && key(p.ident)===(link.own?link.target:copier));
          if(!member || !opponent)continue;
          // Both directions disclose copied moves through this actor's OWN
          // request. Never inspect the other player's request or hidden roster.
          opponent.moves=member.moves.map(m=>Dex.moves.get(m).name);
          opponent.ability=Dex.abilities.get(member.ability).name;
          link.pending=false;
        }
      }
    }
  }
  clearCopy(source:string) {
    const original=this.copyOriginal.get(source);
    const holder=this.opponent.find(p=>p.ident && key(p.ident)===source);
    if(original&&holder){holder.species=original.species;holder.moves=[...original.moves];holder.ability=original.ability;}
    this.copyOriginal.delete(source);this.copyLinks.delete(source);
  }
}
