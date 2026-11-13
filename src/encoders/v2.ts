// SPDX-License-Identifier: GPL-3.0-only
/** Custom self-play observations. This module accepts private streams, never Battle objects. */
import { createHash } from 'node:crypto';
import type { Action, BattleRequest } from '../types.ts';
import Showdown from 'pokemon-showdown';
const { Dex } = Showdown;
import { legalActions, readRequest } from '../legal.ts';
import { toID } from '../types.ts';

export const schema = 'psai-selfplay-observation-v1';
export const ACTIONS = 20;
export const EVENTS = 16;
export const TOKENS = 1 + 12 * 5 + EVENTS + ACTIONS;
export const NUMERIC = 32;
const kinds = ['field', 'own', 'opponent', 'move', 'event', 'action'];
const tags = ['switch', 'drag', 'replace', 'move', 'faint', '-damage', '-heal', '-boost', '-unboost', '-status', '-curestatus', '-weather', '-sidestart', '-sideend', '-terastallize', '-activate', '-item', '-enditem', '-ability', '-start', '-end'];
const names = ['<unknown>', ...[
  ...kinds.map(x => `kind:${x}`), ...tags.map(x => `event:${x}`),
  ...Dex.species.all().map(x => `species:${x.id}`), ...Dex.moves.all().map(x => `move:${x.id}`),
  ...Dex.items.all().map(x => `item:${x.id}`), ...Dex.abilities.all().map(x => `ability:${x.id}`),
  ...Dex.types.all().map(x => `type:${toID(x.name)}`),
  ...['brn', 'par', 'psn', 'tox', 'slp', 'frz', 'fnt'].map(x => `status:${x}`),
].filter((x, i, a) => a.indexOf(x) === i).sort()];
export const vocabulary = { schema, names, actions: ACTIONS, tokens: TOKENS, categorical: 8, numeric: NUMERIC,
  sha256: createHash('sha256').update(JSON.stringify(names)).digest('hex') };
const ids = new Map(names.map((n, i) => [n, i]));
const id = (kind: string, name = '') => ids.get(`${kind}:${toID(name)}`) ?? 0;
const kind = (name: string) => ids.get(`kind:${name}`)!;
export const actionIndex = (a: Action) => a.kind === 'move' ? a.slot - 1 + (a.tera ? 4 : 0) : a.kind === 'switch' ? 7 + a.slot : 13 + a.slot;
export interface Encoded { categorical: number[][]; numeric: number[][]; mask: boolean[] }
interface Known { species: string; ident?: string; condition: string; moves: string[]; item?: string; ability?: string; tera?: string; active?: boolean; level?: number; boosts: Record<string, number>; effects: Set<string> }
const sameIdent = (a?: string, b?: string) => !!a && !!b && a.replace(/^(p[12])[a-z]: /, '$1: ') === b.replace(/^(p[12])[a-z]: /, '$1: ');
const empty = (species = ''): Known => ({ species, condition: '', moves: [], boosts: {}, effects: new Set() });
export class View {
  lines: string[] = [];
  request?: BattleRequest;
  initial?: BattleRequest;
  opponent: Known[] = [];
  ownBoosts: Record<string, number> = {};
  ownEffects = new Set<string>(); fields = new Set<string>();
  weather = ''; turn = 0;
  consumedStart = new Map<string, string>();
  hazards: Record<string, Record<string, number>> = { p1: {}, p2: {} };
  events: { tag: string; entity: number; own: boolean; amount: number }[] = [];
  constructor(readonly side: 'p1' | 'p2') {}
  add(lines: string[]) {
    for (const line of lines) {
      this.lines.push(line);
      const request = readRequest(line);
      if (request) { this.request = request; if (request.side && !this.initial) this.initial = structuredClone(request); continue; }
      const f = line.split('|'), tag = f[1], own = f[2]?.startsWith(this.side) ?? false;
      const theirs = f[2]?.startsWith(this.side === 'p1' ? 'p2' : 'p1');
      if (['move','switch','drag','turn','upkeep'].includes(tag)) this.consumedStart.clear();
      let target = theirs ? this.opponent.find(p => sameIdent(p.ident, f[2])) : undefined;
      let entity = 0;
      if (tag === 'turn') this.turn = Number(f[2]);
      if (tag === 'poke' && theirs && this.opponent.length < 6) this.opponent.push(empty(f[3].split(',')[0]));
      if (['switch', 'drag', 'replace'].includes(tag)) {
        if (own && tag !== 'replace') { this.ownBoosts = {}; this.ownEffects.clear(); }
        if (theirs) {
          const species = f[3].split(',')[0];
          target ??= this.opponent.find(p => !p.ident && (p.species === species || Dex.species.get(p.species).baseSpecies === Dex.species.get(species).baseSpecies));
          if (!target && tag === 'replace') target = this.opponent.find(p => p.active);
          if (!target && this.opponent.length < 6) { target = empty(species); this.opponent.push(target); }
          if (target) { this.opponent.forEach(p => { if (p.active && tag !== 'replace') { p.boosts = {}; p.effects.clear(); } p.active = false; }); Object.assign(target, { species, ident: f[2], active: true, condition: f[4] || target.condition, boosts: tag === 'replace' ? target.boosts : {}, effects: tag === 'replace' ? target.effects : new Set(), level: Number(/L(\d+)/.exec(f[3])?.[1] ?? 100) }); }
        }
        entity = id('species', f[3].split(',')[0]);
      }
      if (tag === 'move') {
        entity = id('move', f[3]);
        if (target && !f.includes('[from] ability: Magic Bounce') && !target.moves.includes(f[3]) && target.moves.length < 4) target.moves.push(f[3]);
      }
      if (target && ['-damage', '-heal', '-sethp'].includes(tag)) target.condition = f[3];
      if (target && tag === 'faint') { target.condition = '0 fnt'; target.active = false; target.boosts = {}; target.effects.clear(); }
      if (own && tag === 'faint') { this.ownBoosts = {}; this.ownEffects.clear(); }
      if (target && tag === '-status') target.condition = target.condition.split(' ')[0] + ' ' + f[3];
      if (target && tag === '-curestatus') target.condition = target.condition.split(' ')[0];
      if (target && tag === '-item') {
        const transfer = f.slice(4).some(x => ['[from] move: Trick','[from] move: Switcheroo','[from] ability: Magician','[from] ability: Pickpocket'].includes(x));
        if (!(transfer && this.consumedStart.get(f[2]) === f[3])) target.item = f[3];
        this.consumedStart.delete(f[2]);
      }
      if (tag === '-enditem' && ['Booster Energy','White Herb'].includes(f[3])) this.consumedStart.set(f[2],f[3]);
      if (tag === '-item' && f.includes('[from] ability: Magician')) {
        const donor = f.find(x => x.startsWith('[of] '))?.slice(5);
        const previous = this.opponent.find(p => sameIdent(p.ident, donor));
        if (previous) previous.item = '';
      }
      if (target && tag === '-enditem') target.item = '';
      if (target && tag === '-ability') {
        if (f[3] === 'As One') {
          const compound = ({calyrexice: 'As One (Glastrier)', calyrexshadow: 'As One (Spectrier)'} as Record<string, string>)[toID(target.species)];
          if (!compound) throw Error('Ambiguous public As One identity');
          target.ability = compound;
        } else if (['As One (Glastrier)', 'As One (Spectrier)'].includes(target.ability || '') && ['Unnerve','Chilling Neigh','Grim Neigh'].includes(f[3])) {
          const component = target.ability === 'As One (Glastrier)' ? 'Chilling Neigh' : 'Grim Neigh';
          if (!['Unnerve', component].includes(f[3])) throw Error('Conflicting public As One component');
        } else target.ability = f[3];
      }
      if (target && tag === '-terastallize') target.tera = f[3];
      if (target && ['detailschange', '-formechange'].includes(tag)) target.species = f[3].split(',')[0];
      const effects = own ? this.ownEffects : target?.effects;
      if (effects && tag === '-start') effects.add(toID(f[3].replace('move: ', '')));
      if (effects && tag === '-end') effects.delete(toID(f[3].replace('move: ', '')));
      if (tag === '-fieldstart') this.fields.add(toID(f[2].replace('move: ', '')));
      if (tag === '-fieldend') this.fields.delete(toID(f[2].replace('move: ', '')));
      const boosts = own ? this.ownBoosts : target?.boosts;
      if (boosts && ['-boost', '-unboost', '-setboost'].includes(tag)) boosts[f[3]] = tag === '-setboost' ? Number(f[4]) : Math.max(-6, Math.min(6, (boosts[f[3]] ?? 0) + Number(f[4]) * (tag === '-unboost' ? -1 : 1)));
      if (tag === '-clearallboost') { this.ownBoosts = {}; this.opponent.forEach(p => { if (p.active) p.boosts = {}; }); }
      if (boosts && ['-clearboost', '-clearpositiveboost', '-clearnegativeboost'].includes(tag)) for (const stat of Object.keys(boosts)) if (tag === '-clearboost' || (tag === '-clearpositiveboost' ? boosts[stat] > 0 : boosts[stat] < 0)) boosts[stat] = 0;
      if (tag === '-weather') this.weather = f[2];
      if (['-sidestart', '-sideend'].includes(tag)) {
        const side = f[2].slice(0, 2), effect = toID(f[3].replace('move: ', ''));
        if (this.hazards[side]) this.hazards[side][effect] = tag === '-sideend' ? 0 : (this.hazards[side][effect] ?? 0) + 1;
      }
      // Experimental semantics v4: resolve the holder by event/effect, never
      // infer it from an undifferentiated [of]. No protocol lines are changed.
      const of = f.find(x => x.startsWith('[of] '))?.slice(5);
      for (const suffix of f.slice(3)) {
        const ability = suffix.startsWith('[from] ability: ') ? suffix.slice(16) : undefined;
        const item = suffix.startsWith('[from] item: ') ? suffix.slice(13) : undefined;
        if (!ability && !item) continue;
        let holderIdent: string | undefined;
        if (ability) {
          if (tag === '-heal' && ['Volt Absorb', 'Water Absorb', 'Earth Eater', 'Poison Heal', 'Cheek Pouch', 'Ice Body', 'Dry Skin'].includes(ability)) holderIdent = f[2];
          else if (tag === '-damage' && ['Liquid Ooze', 'Rough Skin', 'Aftermath', 'Bad Dreams'].includes(ability) && of) holderIdent = of;
          else if ((tag === '-damage' && ability === 'Dry Skin') || (tag === '-item' && ability === 'Harvest')) holderIdent = f[2];
          else if (tag === '-fieldstart' && ['Electric Surge', 'Grassy Surge', 'Psychic Surge', 'Hadron Engine', 'Seed Sower'].includes(ability) && of) holderIdent = of;
          else if (tag === '-weather' && ['Drizzle', 'Drought', 'Sand Stream', 'Snow Warning', 'Orichalcum Pulse'].includes(ability) && of) holderIdent = of;
          else if (tag === '-start' && f[3] === 'Disable' && ability === 'Cursed Body' && of) holderIdent = of;
          else if (tag === '-start' && f[3] === 'Charge' && ability === 'Electromorphosis') holderIdent = f[2];
          else if (tag === '-status' && ['Static', 'Flame Body', 'Effect Spore', 'Poison Touch', 'Toxic Chain'].includes(ability) && of) holderIdent = of;
          else if (tag === 'move' && ability === 'Magic Bounce') holderIdent = f[2];
          else if (tag === '-item' && ability === 'Frisk' && of) holderIdent = of;
          else if (tag === '-item' && ['Magician','Pickpocket'].includes(ability)) holderIdent = f[2];
          else if (tag === '-enditem' && ability === 'Pickpocket') continue;
          else if (['-immune', '-fail'].includes(tag) || (tag === '-curestatus' && ability === 'Natural Cure')) holderIdent = f[2];
          else throw Error('Unsupported public ability attribution: ' + tag + '/' + ability);
        } else {
          if (tag === '-heal' && item === 'Sitrus Berry') continue;
          if (tag === '-status' && ['Toxic Orb', 'Flame Orb'].includes(item!)) holderIdent = f[2];
          else if ((tag === '-heal' && item === 'Leftovers') || (tag === '-damage' && item === 'Life Orb')) holderIdent = f[2];
          else if (tag === '-damage' && item === 'Rocky Helmet' && of) holderIdent = of;
          // Consumed triggered-item boost attribution does not restore held item.
          else if (['-boost','-unboost'].includes(tag) && ['Weakness Policy', 'Throat Spray'].includes(item!)) continue;
          else throw Error('Unsupported public item attribution: ' + tag + '/' + item);
        }
        const holder = this.opponent.find(p => sameIdent(p.ident, holderIdent));
        if (holder && ability) holder.ability = ability;
        if (holder && item) holder.item = item;
      }
      // These direct public ability names are not [from] suffixes. They justify
      // knowledge of the named holder, including after switching or fainting.
      if (target && tag === '-end' && ['Protosynthesis', 'Quark Drive'].includes(f[3])) target.ability = f[3];
      if (target && tag === '-activate' && f[3] === 'Orichalcum Pulse' && f[4] === '[source]') target.ability = 'Orichalcum Pulse';
      if (target && ((tag === '-start' && ['ability: Flash Fire','ability: Slow Start'].includes(f[3])) ||
                     (tag === 'cant' && ['ability: Queenly Majesty','ability: Truant','ability: Damp'].includes(f[3])) ||
                     (tag === '-block' && f[3] === 'ability: Aroma Veil') ||
                     (tag === '-activate' && ['ability: Liquid Ooze', 'ability: Rough Skin', 'ability: Aftermath', 'ability: Toxic Debris', 'ability: Shed Skin', 'ability: Bad Dreams', 'ability: Cud Chew', 'ability: Hydration', 'ability: Orichalcum Pulse', 'ability: Protosynthesis', 'ability: Hadron Engine', 'ability: Quark Drive', 'ability: Insomnia', 'ability: Vital Spirit', 'ability: Immunity', 'ability: Limber', 'ability: Magma Armor', 'ability: Water Veil', 'ability: Water Bubble', 'ability: Own Tempo', 'ability: Oblivious', 'ability: Supreme Overlord', 'ability: Battle Bond'].includes(f[3])))) target.ability = f[3].slice(9);
      if (target && tag === '-activate' && f[3] === 'ability: Sticky Hold') target.ability = 'Sticky Hold';
      if (target && tag === '-activate' && f[3] === 'move: Poltergeist') target.item = f[4];
      if (tags.includes(tag)) { this.events.push({ tag, entity, own, amount: Number(f[4]) || 0 }); this.events = this.events.slice(-EVENTS); }
    }
  }
  actions() { return this.request ? legalActions(this.request) : []; }
  encode(): Encoded {
    if (!this.request?.side) throw new Error('No private request');
    const categorical = Array.from({ length: TOKENS }, () => Array<number>(8).fill(0));
    const numeric = Array.from({ length: TOKENS }, () => Array<number>(NUMERIC).fill(0));
    const mask = Array<boolean>(ACTIONS).fill(false);
    categorical[0][0] = kind('field');
    numeric[0][0] = this.turn / 250; numeric[0][1] = Number(Boolean(this.request.teamPreview));
    numeric[0][2] = Number(Boolean(this.request.forceSwitch?.some(Boolean)));
    numeric[0][3] = Number(Boolean(this.request.active?.[0]?.canTerastallize));
    const weathers = ['RainDance', 'SunnyDay', 'Sandstorm', 'Snowscape', 'Hail'];
    weathers.forEach((w, i) => numeric[0][4 + i] = Number(toID(this.weather) === toID(w)));
    for (const [s, side] of [this.side, this.side === 'p1' ? 'p2' : 'p1'].entries()) ['stealthrock', 'spikes', 'toxicspikes', 'stickyweb', 'reflect'].forEach((h, i) => numeric[0][9 + s * 5 + i] = (this.hazards[side][h] ?? 0) / 3);
    ['trickroom', 'gravity', 'electricterrain', 'grassyterrain', 'mistyterrain', 'psychicterrain'].forEach((effect, i) => numeric[0][19 + i] = Number(this.fields.has(effect)));
    for (const [s, side] of [this.side, this.side === 'p1' ? 'p2' : 'p1'].entries()) ['lightscreen', 'auroraveil', 'tailwind'].forEach((h, i) => numeric[0][25 + s * 3 + i] = Number(Boolean(this.hazards[side][h])));
    const own = this.request.side.pokemon;
    for (let i = 0; i < 12; i++) {
      const mine = i < 6, p = mine ? own[i] : this.opponent[i - 6];
      const row = 1 + i * 5;
      categorical[row][0] = kind(mine ? 'own' : 'opponent');
      if (!p) continue;
      const species = Dex.species.get(mine ? own[i].details.split(',')[0] : this.opponent[i - 6].species);
      const known = mine ? undefined : this.opponent[i - 6];
      const tera = mine ? own[i].terastallized || own[i].teraType : known?.tera;
      categorical[row] = [categorical[row][0], id('species', species.id), id('item', p.item), id('ability', mine ? own[i].ability || own[i].baseAbility : known?.ability), id('type', species.types[0]), id('type', species.types[1]), id('type', tera), id('status', p.condition.split(' ')[1])];
      const [hp, max] = p.condition.split(' ')[0].split('/').map(Number);
      const active = Boolean(p.active), boosts = mine && active ? this.ownBoosts : known?.boosts ?? {};
      numeric[row][0] = Number.isFinite(hp) ? hp / (max || hp || 1) : 0;
      numeric[row][1] = Number(p.condition.includes('fnt')); numeric[row][2] = Number(active); numeric[row][3] = Number(Boolean(species.exists));
      ['atk', 'def', 'spa', 'spd', 'spe'].forEach((stat, k) => { numeric[row][4 + k] = mine ? (own[i].stats[stat] ?? 0) / 500 : 0; numeric[row][9 + k] = (boosts[stat] ?? 0) / 6; });
      numeric[row][14] = Number(Boolean(mine ? own[i].terastallized : known?.tera));
      numeric[row][15] = Number(mine || known?.item !== undefined); numeric[row][16] = Number(mine || known?.ability !== undefined);
      numeric[row][17] = (mine ? Number(/L(\d+)/.exec(own[i].details)?.[1] ?? 100) : known?.level ?? 0) / 100;
      numeric[row][18] = (boosts.accuracy ?? 0) / 6; numeric[row][19] = (boosts.evasion ?? 0) / 6;
      const effects = mine && active ? this.ownEffects : known?.effects;
      ['taunt', 'encore', 'substitute', 'confusion', 'leechseed', 'disable', 'yawn', 'perish3', 'focusenergy', 'ingrain', 'aquaring', 'saltcure'].forEach((effect, j) => numeric[row][20 + j] = Number(Boolean(effects?.has(effect))));
      const moves = p.moves;
      moves.slice(0, 4).forEach((m, j) => {
        const move = Dex.moves.get(m), r = row + j + 1;
        categorical[r] = [kind('move'), id('move', move.id), id('type', move.type), 0, 0, 0, 0, 0];
        numeric[r][0] = move.basePower / 200; numeric[r][1] = move.accuracy === true ? 1 : move.accuracy / 100;
        numeric[r][2] = move.priority / 7; numeric[r][3] = Number(move.category === 'Physical'); numeric[r][4] = Number(move.category === 'Special'); numeric[r][5] = Number(move.category === 'Status');
        const requestMove = mine && active ? this.request!.active?.[0]?.moves.find(x => x.id === move.id) : undefined;
        numeric[r][6] = (requestMove?.pp ?? 0) / 64; numeric[r][7] = Number(Boolean(requestMove?.disabled)); numeric[r][8] = Number(mine);
      });
    }
    this.events.forEach((event, i) => {
      const r = 61 + i; categorical[r][0] = kind('event'); categorical[r][1] = ids.get(`event:${event.tag}`) ?? 0; categorical[r][2] = event.entity;
      numeric[r][0] = Number(event.own); numeric[r][1] = event.amount / 6;
    });
    for (const action of this.actions()) {
      const index = actionIndex(action), row = TOKENS - ACTIONS + index;
      if (index < 0 || index >= ACTIONS || mask[index]) throw new Error('Action encoding collision');
      mask[index] = true; categorical[row][0] = kind('action');
      categorical[row][1] = action.kind === 'move' ? id('move', action.move) : id('species', own[action.slot - 1].details.split(',')[0]);
      numeric[row][0] = Number(action.tera); numeric[row][1] = Number(action.kind === 'move'); numeric[row][2] = Number(action.kind === 'switch'); numeric[row][3] = Number(action.kind === 'team'); numeric[row][4] = action.slot / 6;
    }
    if (!mask.some(Boolean) || numeric.some(row => row.some(n => !Number.isFinite(n)))) throw new Error('Invalid encoded observation');
    return { categorical, numeric, mask };
  }
}
