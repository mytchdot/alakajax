// SPDX-License-Identifier: GPL-3.0-only
export interface RequestPokemon {
  ident: string; details: string; condition: string; active: boolean;
  stats: Record<string, number>; moves: string[]; baseAbility?: string;
  ability?: string; item: string; teraType?: string; terastallized?: string; reviving?: boolean;
}
export interface BattleRequest {
  rqid?: number; wait?: boolean; forceSwitch?: boolean[]; teamPreview?: boolean;
  side?: { id: 'p1' | 'p2'; name: string; pokemon: RequestPokemon[] };
  active?: { moves: { id: string; move: string; pp: number; disabled?: boolean; target?: string }[];
    trapped?: boolean; maybeTrapped?: boolean; canTerastallize?: string }[];
}
export interface Action { id: string; label: string; kind: 'move' | 'switch' | 'team'; slot: number; tera: boolean; move?: string }

export type Format = string;
export const toID = (text: string) => text.toLowerCase().replace(/[^a-z0-9]/g, '');
