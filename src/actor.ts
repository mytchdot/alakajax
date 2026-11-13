// SPDX-License-Identifier: GPL-3.0-only
/** native self-play actor: G synchronous Showdown gen9randombattle games per process.
 *
 * Binary protocol over stdio (little endian), one frame per step:
 *   worker -> driver: header int32[4] = [MAGIC, frame, G, bytes] then the fixed body
 *   driver -> worker: int8[G*2] chosen action index (0..19) or -1 for no decision.
 * Each body reports, per slot, the outcome of the previous step (finished/winner/turns,
 * per-side rejected choice) and, per slot-side, the current decision: actor flag,
 * legal mask, v21 categorical (int16) and numeric (bf16 bits, RNE) views. A finished slot
 * immediately starts a fresh game with newly generated teams (auto-reset).
 * Sides controlled by an in-worker baseline (heuristic/random) never appear as actors.
 */
import Showdown from 'pokemon-showdown';
import BattleModule from 'pokemon-showdown/dist/sim/battle.js';
import { View, actionIndex } from './encoders/v21.ts';
import { heuristicAction, randomAction } from './baselines.ts';
const { Battle } = Showdown as any; const { extractChannelMessages } = BattleModule as any;
type Side = 'p1' | 'p2';
const SIDES: Side[] = ['p1', 'p2'];
const MAGIC = 0x314e414a; // 'JAN1'
const TOK = 97, CAT = 8, NUM = 32, ACT = 20;
const G = Number(process.argv[2]); const SEED = Number(process.argv[3]);
// mode: selfplay | heuristic | random  (baseline plays side opp; policy plays the other)
const DETERMINISTIC_TEAMS = process.argv.includes('--deterministic-teams');
const MODE = process.argv[4] || 'selfplay'; const MAX_PHASES = 3000;
if (!(G > 0 && G <= 512) || !Number.isFinite(SEED)) throw Error('usage: actor.ts G SEED [mode]');

class Game {
  views: Record<Side, any> = { p1: new View('p1'), p2: new View('p2') };
  pending = new Set<Side>(); rejected: Record<Side, boolean> = { p1: false, p2: false };
  winner = ''; phases = 0; battle: any; baseline: Side | null; rand: () => number; keepLines: boolean;
  constructor(readonly seed: number, readonly index: number) {
    let x = (seed * 2654435761 + 97531) >>> 0 || 1;
    this.rand = () => { x ^= x << 13; x >>>= 0; x ^= x >>> 17; x ^= x << 5; x >>>= 0; return x / 4294967296; };
    this.baseline = MODE === 'selfplay' ? null : (index % 2 === 0 ? 'p2' : 'p1');
    this.keepLines = MODE === 'heuristic';
    const s = [seed & 0xffff, (seed >>> 16) & 0xffff, (index * 7919) & 0xffff, 4242];
    this.battle = new Battle({ formatid: 'gen9randombattle', seed: s.join(','), send: (type: string, raw: any) => {
      const data = Array.isArray(raw) ? raw.join('\n') : raw;
      if (type === 'update') {
        const m = extractChannelMessages(data, [1, 2]);
        this.views.p1.add(m[1]); this.views.p2.add(m[2]);
        for (const l of m[1]) { if (l.startsWith('|win|')) this.winner = l.slice(5); if (l === '|tie') this.winner = 'tie'; }
      } else if (type === 'sideupdate') {
        const at = data.indexOf('\n'), side = data.slice(0, at) as Side, lines = data.slice(at + 1).split('\n');
        this.views[side].add(lines);
        for (const l of lines) {
          if (l.startsWith('|request|')) { const r = JSON.parse(l.slice(9)); if (r.wait) this.pending.delete(side); else if (r.side) this.pending.add(side); }
          if (l.startsWith('|error|')) { if (l.startsWith('|error|[Unavailable choice]') || l.startsWith('|error|[Invalid choice]')) this.rejected[side] = true; else throw Error(l); }
        }
      }
      if (!this.keepLines) { this.views.p1.lines.length = 0; this.views.p2.lines.length = 0; }
    } });
    this.battle.setPlayer('p1', { name: 'p1', ...(DETERMINISTIC_TEAMS ? { seed: [s[0],s[1],s[2],4243].join(',') } : {}) });
    this.battle.setPlayer('p2', { name: 'p2', ...(DETERMINISTIC_TEAMS ? { seed: [s[0],s[1],s[2],4244].join(',') } : {}) });
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
      if (side === this.baseline) action = MODE === 'heuristic' ? heuristicAction(v.request, v.lines, 'gen9randombattle' as any) : randomAction(v.request, this.rand);
      else { action = v.actions().find((a: any) => actionIndex(a) === choice[side]); if (!action) throw Error(`illegal or missing choice ${choice[side]} for ${side}`); }
      subs.push([side, action.id]);
    }
    this.pending.clear();
    for (const [s, c] of subs) this.battle.choose(s, c);
    this.battle.sendUpdates(); this.phases++;
  }
}

const games: Game[] = []; let counter = 0;
const fresh = (i: number) => new Game((SEED * 1000003 + counter++) >>> 0, i);
for (let i = 0; i < G; i++) games.push(fresh(i));
const finished = new Int8Array(G), winner = new Int8Array(G), turns = new Int16Array(G), rejected = new Int8Array(G * 2);

// Fixed body layout, every section 4-byte aligned.
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

let frame = 0;
function writeFrame() {
  fin8.set(finished); win8.set(winner); turns16.set(turns); rej8.set(rejected);
  act8.fill(0); mask8.fill(0); cat16.fill(0); num16.fill(0);
  for (let i = 0; i < G; i++) {
    const g = games[i];
    for (const side of g.actors()) {
      const k = 2 * i + (side === 'p1' ? 0 : 1);
      const e = g.views[side].encode();
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
  const head = Buffer.alloc(16); head.writeInt32LE(MAGIC, 0); head.writeInt32LE(frame++, 4); head.writeInt32LE(G, 8); head.writeInt32LE(size, 12);
  process.stdout.write(head); return process.stdout.write(body);
}

// Exact-size binary reads from stdin.
let pending: Buffer = Buffer.alloc(0); const waiters: (() => void)[] = [];
process.stdin.on('data', (chunk: Buffer) => { pending = pending.length ? Buffer.concat([pending, chunk]) : chunk; waiters.splice(0).forEach(w => w()); });
process.stdin.on('end', () => process.exit(0));
async function readExact(n: number): Promise<Buffer> {
  while (pending.length < n) await new Promise<void>(r => waiters.push(r));
  const out = pending.subarray(0, n); pending = pending.subarray(n); return Buffer.from(out);
}

async function main() {
  // Baseline-only decisions (both sides baseline never happens) are resolved before the first frame.
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
        g.battle.destroy(); games[i] = g = fresh(i);
        // A fresh game may begin with baseline-only decisions; resolve them now.
        while (!g.winner && !g.actors().length && g.pending.size) g.step({ p1: -1, p2: -1 });
      }
    }
  }
}
main().catch(e => { process.stderr.write(String(e?.stack || e) + '\n'); process.exit(1); });
