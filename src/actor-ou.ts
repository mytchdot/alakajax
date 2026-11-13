// SPDX-License-Identifier: GPL-3.0-only
/** OU actor: G synchronous Showdown games per process with teams drawn from a validated pool.
 *
 * Same binary stdio protocol as actor.ts (for Random Battles):
 *   worker -> driver: header int32[4] = [MAGIC, frame, G, bytes] then the fixed body
 *   driver -> worker: int8[G*2] chosen action index (0..19) or -1 for no decision.
 * Differences: the format is a fixed-team format (default gen9ou) whose teams come from a packed
 * pool file (one team per line, built by src/teams.ts); the encoder is v22
 * (OU attributions, team-preview Illusion; identical to v21 on Random Battles); team preview is a
 * normal decision (actions 14..19 choose the lead). Lines the encoder cannot read are skipped and
 * counted. A game whose observation cannot be encoded is abandoned
 * (finished code 3: terminal, no reward) instead of crashing the worker.
 * Usage: actor-ou.ts G SEED MODE --teams FILE [--format gen9ou] [--featured FILE --featured-share P]
 *   MODE: selfplay | heuristic | random (baseline plays one side, as in actor.ts).
 *   Featured teams (e.g. teams you want extra practice with) replace one random side's team with probability P
 *   in self-play only.
 */
import { readFileSync } from 'node:fs';
import Showdown from 'pokemon-showdown';
import BattleModule from 'pokemon-showdown/dist/sim/battle.js';
import { View, actionIndex } from './encoders/v22.ts';
import { heuristicAction, randomAction } from './baselines.ts';
const { Battle } = Showdown as any; const { extractChannelMessages } = BattleModule as any;
type Side = 'p1' | 'p2';
const SIDES: Side[] = ['p1', 'p2'];
const MAGIC = 0x314e414a; // 'JAN1'
const TOK = 97, CAT = 8, NUM = 32, ACT = 20;
const argv = process.argv.slice(2);
const option = (name: string) => { const i = argv.indexOf(name); return i < 0 ? undefined : argv.splice(i, 2)[1]; };
const TEAMS = option('--teams'), FORMAT = option('--format') ?? 'gen9ou';
const FEATURED = option('--featured'), FEATURED_SHARE = Number(option('--featured-share') ?? (FEATURED ? 0.25 : 0));
const G = Number(argv[0]); const SEED = Number(argv[1]);
const MODE = argv[2] || 'selfplay'; const MAX_PHASES = 3000;
if (!(G > 0 && G <= 512) || !Number.isFinite(SEED) || !TEAMS || !['selfplay', 'heuristic', 'random'].includes(MODE)
  || !(FEATURED_SHARE >= 0 && FEATURED_SHARE <= 1)) {
  throw Error('usage: actor-ou.ts G SEED MODE --teams FILE [--format gen9ou] [--featured FILE --featured-share P]');
}

/** Packed teams, one per line, kept as one buffer plus line offsets (the training pool is ~120 MB). */
class Pool {
  readonly starts: Uint32Array; readonly ends: Uint32Array;
  constructor(readonly path: string, readonly text = readFileSync(path)) {
    const starts: number[] = [], ends: number[] = [];
    for (let at = 0; at < text.length;) {
      let end = text.indexOf(10, at); if (end < 0) end = text.length;
      if (end > at) { starts.push(at); ends.push(end); }
      at = end + 1;
    }
    if (!starts.length) throw Error(`no teams in ${path}`);
    this.starts = Uint32Array.from(starts); this.ends = Uint32Array.from(ends);
  }
  get size() { return this.starts.length; }
  team(rand: () => number) { const i = Math.floor(rand() * this.size); return this.text.toString('utf8', this.starts[i], this.ends[i]); }
}
const pool = new Pool(TEAMS), featured = FEATURED ? new Pool(FEATURED) : undefined;

// Diagnostics go to stderr (the driver keeps one log per worker).
const skippedBy = new Map<string, number>(); let skippedLines = 0, abandoned = 0, gamesDone = 0;
function feed(view: any, lines: string[]) {
  for (const line of lines) {
    try { view.add([line]); } catch (error) {
      skippedLines++; const k = String((error as Error)?.message ?? error).slice(0, 120);
      skippedBy.set(k, (skippedBy.get(k) ?? 0) + 1);
    }
  }
}
function report(force = false) {
  if (!force && gamesDone % 10000) return;
  const top = [...skippedBy].sort((a, b) => b[1] - a[1]).slice(0, 5);
  process.stderr.write(`${new Date().toISOString()} games=${gamesDone} abandoned=${abandoned} skippedLines=${skippedLines} ${JSON.stringify(top)}\n`);
}

class Game {
  views: Record<Side, any> = { p1: new View('p1'), p2: new View('p2') };
  pending = new Set<Side>(); rejected: Record<Side, boolean> = { p1: false, p2: false };
  winner = ''; phases = 0; decided = 0; battle: any; baseline: Side | null; rand: () => number; keepLines: boolean;
  constructor(readonly seed: number, readonly index: number) {
    let x = (seed * 2654435761 + 97531) >>> 0 || 1;
    this.rand = () => { x ^= x << 13; x >>>= 0; x ^= x >>> 17; x ^= x << 5; x >>>= 0; return x / 4294967296; };
    this.baseline = MODE === 'selfplay' ? null : (index % 2 === 0 ? 'p2' : 'p1');
    this.keepLines = MODE === 'heuristic';
    const teams: Record<Side, string> = { p1: pool.team(this.rand), p2: pool.team(this.rand) };
    if (featured && MODE === 'selfplay' && this.rand() < FEATURED_SHARE) teams[this.rand() < 0.5 ? 'p1' : 'p2'] = featured.team(this.rand);
    const s = [seed & 0xffff, (seed >>> 16) & 0xffff, (index * 7919) & 0xffff, 4242];
    this.battle = new Battle({ formatid: FORMAT, seed: s.join(','), send: (type: string, raw: any) => {
      const data = Array.isArray(raw) ? raw.join('\n') : raw;
      if (type === 'update') {
        const m = extractChannelMessages(data, [1, 2]);
        feed(this.views.p1, m[1]); feed(this.views.p2, m[2]);
        for (const l of m[1]) { if (l.startsWith('|win|')) this.winner = l.slice(5); if (l === '|tie') this.winner = 'tie'; }
      } else if (type === 'sideupdate') {
        const at = data.indexOf('\n'), side = data.slice(0, at) as Side, lines = data.slice(at + 1).split('\n');
        feed(this.views[side], lines);
        for (const l of lines) {
          if (l.startsWith('|request|')) { const r = JSON.parse(l.slice(9)); if (r.wait) this.pending.delete(side); else if (r.side) this.pending.add(side); }
          if (l.startsWith('|error|')) { if (l.startsWith('|error|[Unavailable choice]') || l.startsWith('|error|[Invalid choice]')) this.rejected[side] = true; else throw Error(l); }
        }
      }
      if (!this.keepLines) { this.views.p1.lines.length = 0; this.views.p2.lines.length = 0; }
    } });
    this.battle.setPlayer('p1', { name: 'p1', team: teams.p1 }); this.battle.setPlayer('p2', { name: 'p2', team: teams.p2 });
    this.battle.sendUpdates();
  }
  /** Sides the driver must choose for now (baseline sides are played in-worker). */
  actors(): Side[] { return this.winner ? [] : SIDES.filter(s => this.pending.has(s) && s !== this.baseline); }
  step(choice: Record<Side, number>) {
    const subs: [Side, string][] = [];
    this.rejected = { p1: false, p2: false };
    for (const side of SIDES) {
      if (!this.pending.has(side)) continue;
      const v = this.views[side];
      let action: any;
      if (side === this.baseline) action = MODE === 'heuristic' ? heuristicAction(v.request, v.lines, FORMAT as any) : randomAction(v.request, this.rand);
      else { action = v.actions().find((a: any) => actionIndex(a) === choice[side]); if (!action) throw Error(`illegal or missing choice ${choice[side]} for ${side}`); this.decided++; }
      subs.push([side, action.id]);
    }
    this.pending.clear();
    for (const [s, c] of subs) this.battle.choose(s, c);
    this.battle.sendUpdates(); this.phases++;
  }
}

const games: Game[] = []; let counter = 0;
const fresh = (i: number) => new Game((SEED * 1000003 + counter++) >>> 0, i);
/** A fresh game may begin with baseline-only decisions; resolve them before it is reported. */
const start = (i: number) => { const g = fresh(i); while (!g.winner && !g.actors().length && g.pending.size) g.step({ p1: -1, p2: -1 }); return g; };
for (let i = 0; i < G; i++) games.push(start(i));
const finished = new Int8Array(G), winner = new Int8Array(G), turns = new Int16Array(G), rejected = new Int8Array(G * 2);

// Fixed body layout, every section 4-byte aligned (identical to actor.ts).
const pad = (n: number) => (n + 3) & ~3;
const off: Record<string, number> = {}; let size = 0;
for (const [k, n] of [['finished', G], ['winner', G], ['turns', 2 * G], ['rejected', 2 * G], ['actor', 2 * G], ['mask', 2 * G * ACT],
  ['cat', 2 * G * TOK * CAT * 2], ['num', 2 * G * TOK * NUM * 2]] as [string, number][]) { off[k] = size; size += pad(n); }
const body = Buffer.alloc(size);
const v8 = (k: string, n: number) => new Int8Array(body.buffer, body.byteOffset + off[k], n);
const fin8 = v8('finished', G), win8 = v8('winner', G), act8 = v8('actor', 2 * G), rej8 = v8('rejected', 2 * G);
const turns16 = new Int16Array(body.buffer, body.byteOffset + off.turns, G);
const mask8 = new Uint8Array(body.buffer, body.byteOffset + off.mask, 2 * G * ACT);
const cat16 = new Int16Array(body.buffer, body.byteOffset + off.cat, 2 * G * TOK * CAT);
const num16 = new Uint16Array(body.buffer, body.byteOffset + off.num, 2 * G * TOK * NUM);
const f32 = new Float32Array(1), u32 = new Uint32Array(f32.buffer);
const bf16 = (x: number) => { f32[0] = x; const b = u32[0]; return ((b + 0x7fff + ((b >>> 16) & 1)) >>> 16) & 0xffff; };

/** Encode slot i's pending decisions. If that fails, abandon the game and start a fresh one; a game
 * that already had driver decisions is reported as finished code 3 (terminal, no reward). A fresh game
 * has none, so a natural result reported for this slot in the same frame is never overwritten. */
function encodeSlot(i: number) {
  for (let attempt = 0; ; attempt++) {
    const g = games[i];
    try { return g.actors().map(side => ({ side, e: g.views[side].encode() })); } catch (error) {
      if (attempt >= 3) throw error;
      const k = 'abandoned: ' + String((error as Error)?.message ?? error).slice(0, 100);
      abandoned++; skippedBy.set(k, (skippedBy.get(k) ?? 0) + 1);
      if (g.decided) { finished[i] = 3; winner[i] = -1; turns[i] = Math.min(32767, g.views.p1.turn ?? 0); }
      g.battle.destroy(); games[i] = start(i);
    }
  }
}

let frame = 0;
function writeFrame() {
  act8.fill(0); mask8.fill(0); cat16.fill(0); num16.fill(0);
  for (let i = 0; i < G; i++) {
    for (const { side, e } of encodeSlot(i)) {
      const k = 2 * i + (side === 'p1' ? 0 : 1);
      act8[k] = 1;
      for (let a = 0; a < ACT; a++) mask8[k * ACT + a] = e.mask[a] ? 1 : 0;
      let c = k * TOK * CAT, n = k * TOK * NUM;
      for (let t = 0; t < TOK; t++) {
        const ct = e.categorical[t], nt = e.numeric[t];
        for (let j = 0; j < CAT; j++) cat16[c++] = ct[j];
        for (let j = 0; j < NUM; j++) num16[n++] = bf16(nt[j]);
      }
    }
  }
  fin8.set(finished); win8.set(winner); turns16.set(turns); rej8.set(rejected);
  const head = Buffer.alloc(16); head.writeInt32LE(MAGIC, 0); head.writeInt32LE(frame++, 4); head.writeInt32LE(G, 8); head.writeInt32LE(size, 12);
  process.stdout.write(head); return process.stdout.write(body);
}

// Exact-size binary reads from stdin.
let pending: Buffer = Buffer.alloc(0); const waiters: (() => void)[] = [];
process.stdin.on('data', (chunk: Buffer) => { pending = pending.length ? Buffer.concat([pending, chunk]) : chunk; waiters.splice(0).forEach(w => w()); });
process.stdin.on('end', () => { report(true); process.exit(0); });
async function readExact(n: number): Promise<Buffer> {
  while (pending.length < n) await new Promise<void>(r => waiters.push(r));
  const out = pending.subarray(0, n); pending = pending.subarray(n); return Buffer.from(out);
}

async function main() {
  process.stderr.write(`${new Date().toISOString()} start format=${FORMAT} mode=${MODE} G=${G} teams=${pool.size} featured=${featured?.size ?? 0} share=${FEATURED_SHARE}\n`);
  finished.fill(0); winner.fill(-1); turns.fill(0); rejected.fill(0);
  for (;;) {
    if (!writeFrame()) await new Promise<void>(r => process.stdout.once('drain', () => r()));
    const choices = new Int8Array(await readExact(2 * G));
    finished.fill(0); winner.fill(-1); turns.fill(0); rejected.fill(0);
    for (let i = 0; i < G; i++) {
      let g = games[i];
      // Advance until this slot has a driver decision or ended (baseline-only phases loop here).
      let first = true, guard = 0;
      while (!g.winner) {
        if (++guard > 5000 || (!first && !g.pending.size)) throw Error('slot ' + i + ' stalled without requests');
        const need = g.actors();
        if (!first && need.length) break;
        g.step({ p1: choices[2 * i], p2: choices[2 * i + 1] }); first = false;
        if (g.rejected.p1) rejected[2 * i] = 1; if (g.rejected.p2) rejected[2 * i + 1] = 1;
        if (g.phases >= MAX_PHASES && !g.winner) { g.winner = 'cap'; break; }
        if (g.actors().length) break;
      }
      if (g.winner) {
        finished[i] = g.winner === 'cap' ? 2 : 1;
        winner[i] = g.winner === 'p1' ? 0 : g.winner === 'p2' ? 1 : -1;
        turns[i] = Math.min(32767, g.views.p1.turn);
        g.battle.destroy(); games[i] = start(i); gamesDone++; report();
      }
    }
  }
}
main().catch(e => { report(true); process.stderr.write(String(e?.stack || e) + '\n'); process.exit(1); });
