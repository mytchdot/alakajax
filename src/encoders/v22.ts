// SPDX-License-Identifier: GPL-3.0-only
/** OU public-attribution and team-preview Illusion extension; v21 stays frozen.
 * Accepts only the player's protocol channel and private requests, never Battle.
 * Each line is processed separately, independent of add() call boundaries.
 */
import { View as Previous, vocabulary, actionIndex } from './v21.ts';
import Showdown from 'pokemon-showdown';
const { Dex } = Showdown;
export { vocabulary, actionIndex };

const key = (ident: string) => ident.replace(/^(p[12])[a-z]: /, '$1: ');
type Mon = Previous['opponent'][number];
const sameSpecies = (a: string, b: string) => a === b || Dex.species.get(a).baseSpecies === Dex.species.get(b).baseSpecies;
// Only the stat-changing berry family shares the -boost/-unboost protocol.
// Lansat's Focus Energy and other distinct berry effects are not inferred here.
const boostItems = new Set([
  'Electric Seed', 'Grassy Seed', 'Misty Seed', 'Psychic Seed',
  'Liechi Berry', 'Ganlon Berry', 'Salac Berry', 'Petaya Berry', 'Apicot Berry', 'Starf Berry',
  'Kee Berry', 'Maranga Berry', 'Mirror Herb', 'Room Service',
]);
const healingBerries = new Set(['Oran Berry', 'Figy Berry', 'Wiki Berry', 'Mago Berry', 'Aguav Berry', 'Iapapa Berry', 'Enigma Berry']);

export class View extends Previous {
  private hasPreview = false;
  private previewPrior: { mon: Mon; before: Mon } | undefined;

  override add(lines: string[]) {
    for (const raw of lines) {
      const f = raw.split('|'), tag = f[1];
      const theirs = f[2]?.startsWith(this.side === 'p1' ? 'p2' : 'p1') ?? false;
      if (tag === 'poke' && theirs) this.hasPreview = true;

      if (this.hasPreview && theirs && (tag === 'switch' || tag === 'drag')) {
        const alias = key(f[2]), species = f[3].split(',')[0];
        const named = this.opponent.find(p => p.ident && key(p.ident) === alias);
        const slot = this.opponent.find(p => !p.ident && sameSpecies(p.species, species));
        if (named && slot && !sameSpecies(named.species, species)) {
          // Multiple real members may share a nickname. This new public
          // appearance occupies its matching preview species slot; the old
          // name alone cannot identify it as the previously revealed member.
          if (named.active) this.clearCopy(alias);
          named.ident = undefined;
        }
        // v21 remembers prior identified appearances, but a preview slot has no
        // ident yet. Save that slot too, before the apparent entrant occupies it.
        const mon = this.opponent.find(p => p.ident && key(p.ident) === key(f[2]))
          || this.opponent.find(p => !p.ident && sameSpecies(p.species, f[3].split(',')[0]));
        this.previewPrior = mon ? {
          mon, before: structuredClone({ ...mon, active: false, boosts: {}, effects: new Set<string>() }),
        } : undefined;
      }

      if (this.hasPreview && theirs && tag === 'replace') {
        const current = this.opponent.find(p => p.active && p.ident);
        if (!current) throw Error('Illusion reveal has no active public appearance');
        const source = key(f[2]), species = f[3].split(',')[0];
        // Bind the revealed holder to its existing public preview slot. Do not
        // append a duplicate Zoroark or sacrifice the disguised species' slot.
        const holder = this.opponent.find(p => p !== current && p.ident && key(p.ident) === source)
          || this.opponent.find(p => p !== current && !p.ident && sameSpecies(p.species, species));
        if (holder) {
          const alias = key(current.ident!), episode = structuredClone(current), previous = structuredClone(holder);
          const prior = this.previewPrior?.mon === current ? this.previewPrior.before : this.appearancePrior?.mon;
          if (!prior) throw Error('Illusion preview appearance has no saved public slot');
          Object.assign(current, structuredClone(prior), { active: false });
          if (current.ident) this.appearanceUncertain.set(key(current.ident), this.appearancePrior?.uncertain ?? true);
          else this.appearanceUncertain.delete(alias);
          Object.assign(holder, episode, { ident: f[2], species, active: true });
          // A shared nickname must not let the base replace handler select an
          // inactive disguise slot instead of the active, revealed holder.
          for (const p of this.opponent) if (p !== holder && p.ident && key(p.ident) === source) p.ident = undefined;
          holder.moves = [...new Set([...previous.moves, ...episode.moves])].slice(0, 4);
          if (!this.episodeItemTouched && previous.item !== undefined) holder.item = previous.item;
          if (episode.tera === undefined && previous.tera !== undefined) holder.tera = previous.tera;
          // v21 now sees the active, correctly bound reveal holder. It retains
          // normal replace/event/form/uncertainty semantics and raw history.
          this.appearancePrior = undefined;
        }
        this.previewPrior = undefined;
      }

      let line = raw;
      for (const suffix of f.slice(3)) {
        const item = suffix.startsWith('[from] item: ') ? suffix.slice(13) : undefined;
        const ability = suffix.startsWith('[from] ability: ') ? suffix.slice(16) : undefined;
        let holderIdent: string | undefined;
        let handled = false;
        if (item) {
          if ((tag === '-heal' && ['Black Sludge', 'Shell Bell'].includes(item))
              || (tag === '-damage' && ['Black Sludge', 'Sticky Barb'].includes(item))
              || (tag === '-fail' && f[3] === 'unboost' && item === 'Clear Amulet')) {
            holderIdent = f[2]; handled = true;
          } else if ((['-boost', '-unboost'].includes(tag) && boostItems.has(item))
              || (tag === '-heal' && healingBerries.has(item))) {
            // The preceding -enditem already disclosed consumption. A berry's
            // effect can also be received via Bug Bite/Pluck: do not infer that
            // its recipient holds it, or resurrect it after consumption.
            handled = true;
          }
        }
        if (ability) {
          if ((tag === '-heal' && ability === 'Rain Dish')
              || (tag === '-damage' && ability === 'Solar Power')
              || (tag === '-start' && f[3] === 'Charge' && ability === 'Wind Power')) {
            holderIdent = f[2]; handled = true;
          } else if (tag === '-fieldstart' && ability === 'Misty Surge') {
            holderIdent = f.find(x => x.startsWith('[of] '))?.slice(5);
            if (!holderIdent) throw Error('Missing Misty Surge holder');
            handled = true;
          } else if ((tag === '-status' && ability === 'Poison Point')
              || (tag === '-weather' && ability === 'Sand Spit')) {
            holderIdent = f.find(x => x.startsWith('[of] '))?.slice(5);
            if (!holderIdent) throw Error(`Missing ${ability} holder`);
            handled = true;
          }
        }
        if (!handled) continue;
        const holder = holderIdent ? this.opponent.find(p => p.ident && key(p.ident) === key(holderIdent!)) : undefined;
        if (holder && item) { holder.item = item; this.episodeItemTouched = true; }
        if (holder && ability) holder.ability = ability;
        line = line.split('|').filter(x => x !== suffix).join('|');
      }
      super.add([line]);
      if (line !== raw && this.lines.at(-1) === line) this.lines[this.lines.length - 1] = raw;
    }
  }
}
