// SPDX-License-Identifier: GPL-3.0-only
/** v21: v20 appearance semantics plus external Dancer ownership attribution. */
/** Public appearances are uncertain between a switch and an Illusion reveal.
 * Opponent numeric[row][4], previously an always-zero private-stat slot, is now
 * explicit appearance uncertainty. Never use hidden roster IDs to resolve it.
 * Current-entry evidence is reattributed only by a real public replace event.
 */
import {View as Previous,vocabulary,actionIndex} from './v19.ts';
export {vocabulary,actionIndex};
const key=(ident:string)=>ident.replace(/^(p[12])[a-z]: /,'$1: ');
const id=(s:string)=>s.toLowerCase().replace(/[^a-z0-9]/g,'');
type Mon=Previous['opponent'][number];
export class View extends Previous {
  appearancePrior: {alias:string,mon:Mon|null,uncertain:boolean}|undefined=undefined;
  appearanceUncertain=new Map<string,boolean>();
  episodeItemTouched=false;
  ownAlias='';
  ownSource='';
  override add(lines:string[]) {
    // A fresh OWN request is allowed evidence for our actual chosen entrant.
    // It never resolves an opposing disguise or supplies opponent metadata.
    const fresh=lines.filter(x=>x.startsWith('|request|')).map(x=>JSON.parse(x.slice(9))).findLast(r=>r.side);
    const ownActive=fresh?.side?.pokemon.find((p:any)=>p.active);
    for(const raw of lines) {
      let line=raw;const f=raw.split('|'),tag=f[1];
      const own=f[2]?.startsWith(this.side)??false;
      const theirs=f[2]?.startsWith(this.side==='p1'?'p2':'p1')??false;
      if(tag==='move'&&f[3]==='recharge') {
        if(f.length!==6||f[4]!==''||f[5]!=='[still]')throw Error('Unsupported synthetic recharge history');
        this.consumedStart.clear();this.events.push({tag,entity:0,own,amount:0});this.events=this.events.slice(-16);this.lines.push(raw);
        continue; // a queue action is not an owned Dex move or moveslot
      }
      const borrowed=theirs&&tag==='move'&&raw.includes('[from] ability: Dancer');
      const priorOwned=borrowed?this.opponent.find(p=>p.ident&&key(p.ident)===key(f[2]))?.moves.slice():undefined;
      if(own && (tag==='switch'||tag==='drag')) {
        this.ownAlias=key(f[2]);this.ownSource=ownActive?key(ownActive.ident):'';
      }
      if(own && this.ownSource && key(f[2])===this.ownAlias) {
        const g=[...f];g[2]=f[2].replace(/: .+$/,': '+this.ownSource.split(': ').slice(1).join(': '));line=g.join('|');
      }
      if(theirs && (tag==='switch'||tag==='drag')) {
        for(const p of this.opponent.filter(p=>p.active&&p.ident))this.clearCopy(key(p.ident!));
        const alias=key(f[2]),old=this.opponent.find(p=>p.ident&&key(p.ident)===alias);
        this.appearancePrior={alias,mon:old?structuredClone({...old,active:false,boosts:{},effects:new Set<string>()}):null,uncertain:this.appearanceUncertain.get(alias)??true};
        this.episodeItemTouched=false;
        super.add([line]);
        const current=this.opponent.find(p=>p.ident&&key(p.ident)===alias);
        if(!current)throw Error('Missing public appearance');
        // This could be a different real holder using the same visible name.
        // Keep old evidence separately until an actual reveal resolves ownership.
        current.moves=[];current.ability=undefined;current.item=undefined;current.tera=/, tera:([^,]+)/.exec(f[3])?.[1];
        this.appearanceUncertain.set(alias,true);
        continue;
      }
      if(theirs && tag==='replace') {
        const current=this.opponent.find(p=>p.active&&p.ident);
        if(!current)throw Error('Illusion reveal has no active public appearance');
        const alias=key(current.ident!),source=key(f[2]),episode=structuredClone(current),prior=this.appearancePrior;
        if(alias!==source) {
          if(prior?.alias===alias && prior.mon) {
            Object.assign(current,structuredClone(prior.mon));current.active=false;
            this.appearanceUncertain.set(alias,prior.uncertain);
          } else {
            this.opponent=this.opponent.filter(p=>p!==current);this.appearanceUncertain.delete(alias);
          }
        }
        let holder=this.opponent.find(p=>p.ident&&key(p.ident)===source);
        const previous=holder?structuredClone(holder):undefined;
        if(!holder) {holder=episode;this.opponent.push(holder);}
        Object.assign(holder,episode,{ident:f[2],species:f[3].split(',')[0],active:true});
        holder.moves=[...new Set([...(previous?.moves||[]),...episode.moves])].slice(0,4);
        if(!this.episodeItemTouched && previous?.item!==undefined)holder.item=previous.item;
        if(episode.tera===undefined && previous?.tera!==undefined)holder.tera=previous.tera;
        super.add([line]);this.appearanceUncertain.set(source,false);this.appearancePrior=undefined;
        continue;
      }
      if(theirs && (tag==='-item'||tag==='-enditem'))this.episodeItemTouched=true;
      if(tag==='move'&&raw.includes('[from] ability: Dancer'))line=line.replace('|[from] ability: Dancer','');
      super.add([line]);
      if(borrowed) {
        const holder=this.opponent.find(p=>p.ident&&key(p.ident)===key(f[2]));
        if(holder)holder.moves=priorOwned||[];
      }
      if(theirs&&tag==='-activate'&&f[3]==='ability: Dancer') {
        const holder=this.opponent.find(p=>p.ident&&key(p.ident)===key(f[2]));
        if(holder)holder.ability='Dancer';
      }
      if(theirs && tag==='-end' && f[3]==='Illusion') {
        const holder=this.opponent.find(p=>p.ident&&key(p.ident)===key(f[2]));
        if(!holder)throw Error('Illusion End has no public holder');holder.ability='Illusion';
      }
      if(tag==='request' && this.request?.side) {
        const active=this.request.side.pokemon.find(p=>p.active);
        if(active && id(active.baseAbility||'')==='illusion') {
          // Admitted original Illusion holders are the two static Zoroark
          // identities. Their own request is true even while public switch is fake.
          const actual=key(active.ident);
          this.ownSource=actual;this.ownActiveIdent=actual;
          this.ownCurrentForm.set(actual,active.details.split(',')[0]);
        }
      }
      // Keep actual protocol history; rewritten own aliases affect tracking only.
      if(line!==raw && this.lines.at(-1)===line)this.lines[this.lines.length-1]=raw;
    }
  }
  override encode() {
    const result=super.encode();
    for(let i=0;i<this.opponent.length;i++) {
      const p=this.opponent[i];result.numeric[31+i*5][4]=Number(p.ident?this.appearanceUncertain.get(key(p.ident))??true:true);
    }
    return result;
  }
}
