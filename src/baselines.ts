// SPDX-License-Identifier: GPL-3.0-only
// Extracted from the original repository's random/heuristic baselines.
// Only the damage heuristic and uniform legal-action selection are included.
import Showdown from 'pokemon-showdown';
import { legalActions } from './legal.ts';
import type { Action, BattleRequest, Format } from './types.ts';
const { Dex } = Showdown;
const generationNumber = (format: string) => Number(/^gen(\d+)/.exec(format)?.[1] ?? 9);
export function randomAction(request: BattleRequest, random = Math.random): Action {
  const actions = legalActions(request);
  if (!actions.length) throw new Error('No legal actions');
  return actions[Math.floor(random() * actions.length)];
}
export function heuristicAction(request: BattleRequest, lines: string[], format: Format): Action {
  const dex = Dex.forGen(generationNumber(format));
  const opponent = request.side?.id === 'p1' ? 'p2' : 'p1';
  const lastSwitch = lines.findLast(l => l.startsWith(`|switch|${opponent}`) || l.startsWith(`|drag|${opponent}`) || l.startsWith(`|replace|${opponent}`));
  const opposingSpecies = dex.species.get(lastSwitch?.split('|')[3]?.split(',')[0] ?? '');
  const team = request.side!.pokemon;
  const active = team.find(p => p.active);
  const ownSpecies = dex.species.get(active?.details.split(',')[0] ?? '');
  const actions = legalActions(request);
  const scored = actions.map(action => {
    let score: number;
    if (action.kind === 'move') {
      const move = dex.moves.get(action.move!);
      const category = generationNumber(format) <= 3 && move.category !== 'Status' ? (['Normal', 'Fighting', 'Flying', 'Poison', 'Ground', 'Rock', 'Bug', 'Ghost', 'Steel'].includes(move.type) ? 'Physical' : 'Special') : move.category;
      const attack = active?.stats[category === 'Physical' ? 'atk' : 'spa'] ?? 100;
      const multiplier = opposingSpecies.exists ? dex.getImmunity(move.type, opposingSpecies.types) ? 2 ** dex.getEffectiveness(move.type, opposingSpecies.types) : 0 : 1;
      score = Number(move.basePower || 0) * (ownSpecies.types?.includes(move.type) ? 1.5 : 1) * multiplier * attack / 100 * (typeof move.accuracy === 'number' ? move.accuracy / 100 : 1);
      if (typeof move.damage === 'number') score = move.damage;
      if (move.damage === 'level') score = Number(/L(\d+)/.exec(active?.details ?? '')?.[1] ?? 100);
      if (action.tera) score -= 1; // Simple baseline conserves Tera, without claiming strategic evaluation.
    } else {
      const p = team[action.slot - 1];
      const species = dex.species.get(p.details.split(',')[0]);
      const [hp, max] = p.condition.split(' ')[0].split('/').map(Number);
      const resistance = opposingSpecies.types?.reduce((sum, type) => sum + (dex.getImmunity(type, species.types) ? 2 ** dex.getEffectiveness(type, species.types) : 0), 0) ?? 1;
      score = 1 + (hp / (max || hp || 1)) * 10 - resistance;
    }
    return { action, score };
  }).sort((a, b) => b.score - a.score || a.action.id.localeCompare(b.action.id));
  if (!scored.length) throw new Error('No legal baseline actions');
  return scored[0].action;
}
