// SPDX-License-Identifier: GPL-3.0-only
// Extracted from the original trainer's shared protocol; action ordering is preserved.
import type { Action, BattleRequest } from './types.ts';
export function readRequest(line: string): BattleRequest | null {
  if (!line.startsWith('|request|')) return null;
  try { const req = JSON.parse(line.slice(9)); return req && typeof req === 'object' ? req : null; } catch { return null; }
}
export function legalActions(request: BattleRequest): Action[] {
  if (request.wait || !request.side?.pokemon) return [];
  const team = request.side.pokemon;
  const pokemonLabel = (i: number) => team[i].details.split(',')[0];
  if (request.teamPreview) return team.map((_, i) => ({ id: `team ${i + 1}${team.map((_, j) => j + 1).filter(j => j !== i + 1).join('')}`, label: `Lead ${pokemonLabel(i)}`, kind: 'team', slot: i + 1, tera: false }));
  const active = request.active?.[0];
  const forced = request.forceSwitch?.some(Boolean);
  const actions: Action[] = [];
  if (!forced && active) {
    active.moves.forEach((move, i) => {
      if (move.disabled || (move.pp <= 0 && move.id !== 'struggle')) return;
      actions.push({ id: `move ${i + 1}`, label: move.move, kind: 'move', slot: i + 1, tera: false, move: move.id });
      if (active.canTerastallize) actions.push({ id: `move ${i + 1} terastallize`, label: `${move.move} + Tera ${active.canTerastallize}`, kind: 'move', slot: i + 1, tera: true, move: move.id });
    });
  }
  if (forced || (active && !active.trapped)) team.forEach((pkmn, i) => {
    const fainted = pkmn.condition.endsWith('fnt') || pkmn.condition.startsWith('0 ');
    const reviving = team.some(p => p.reviving);
    if (pkmn.active || (reviving ? !fainted : fainted)) return;
    actions.push({ id: `switch ${i + 1}`, label: `${reviving ? 'Revive' : 'Switch to'} ${pokemonLabel(i)}`, kind: 'switch', slot: i + 1, tera: false });
  });
  return actions;
}
