// SPDX-License-Identifier: GPL-3.0-only
/** Validate user-provided Showdown exports or packed teams; never download implicitly. */
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';
import { createHash } from 'node:crypto';
import Showdown from 'pokemon-showdown';
const { Teams, TeamValidator } = Showdown;

export function validateAndPack(text: string, format = 'gen9ou'): string[] {
  const validator = new TeamValidator(format);
  const chunks = text.includes('|') && !text.includes('\nAbility:')
    ? text.split(/\r?\n/).filter(x => x.trim())
    : text.replace(/^===.*===\s*$/gm, '\n---\n').split(/\n\s*---\s*\n/).filter(x => x.trim());
  const out = new Set<string>();
  for (const [i, chunk] of chunks.entries()) {
    const team = chunk.includes('|') ? Teams.unpack(chunk.trim()) : Teams.import(chunk.trim());
    if (!team?.length) throw Error(`team ${i + 1}: cannot parse`);
    const problems = validator.validateTeam(team);
    if (problems?.length) throw Error(`team ${i + 1}: ${problems.join('; ')}`);
    // Do not redistribute user nicknames in normalized pools.
    for (const mon of team) mon.name = mon.species;
    out.add(Teams.pack(team));
  }
  if (!out.size) throw Error('input contains no teams');
  return [...out];
}

export function splitPool(teams: string[], fraction: number, seed: string) {
  if (!(fraction >= 0 && fraction < 1)) throw Error('heldout fraction must be in [0,1)');
  const ordered = [...teams].sort((a,b) => createHash('sha256').update(seed+a).digest('hex').localeCompare(createHash('sha256').update(seed+b).digest('hex')));
  const n = Math.floor(ordered.length * fraction);
  return { train: ordered.slice(n), evaluation: ordered.slice(0,n) };
}

if (process.argv[1] && import.meta.url === (await import('node:url')).pathToFileURL(process.argv[1]).href) {
  const args = process.argv.slice(2);
  const value = (flag: string, fallback?: string) => { const i=args.indexOf(flag); return i<0 ? fallback : args[i+1]; };
  const input=value('--input'), output=value('--out'), evalOut=value('--eval-out');
  if (!input || !output) throw Error('usage: npm run teams -- --input teams.txt --out train.txt [--eval-out heldout.txt --heldout-fraction 0.1 --seed 17 --format gen9ou]');
  const teams=validateAndPack(readFileSync(input,'utf8'),value('--format','gen9ou'));
  const split=splitPool(teams,evalOut ? Number(value('--heldout-fraction','0.1')) : 0,value('--seed','17')!);
  const write=(path:string, pool:string[]) => { mkdirSync(dirname(path),{recursive:true});writeFileSync(path,pool.join('\n')+'\n'); };
  if (evalOut && !split.evaluation.length) throw Error('held-out split is empty; supply more teams or a larger fraction');
  write(output,split.train); if(evalOut) write(evalOut,split.evaluation);
  console.log(JSON.stringify({unique:teams.length,train:split.train.length,evaluation:split.evaluation.length}));
}
