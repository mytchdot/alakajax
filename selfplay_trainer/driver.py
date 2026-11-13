# SPDX-License-Identifier: GPL-3.0-only
"""Native self-play driver; original asynchronous collection/PPO segment ordering."""
import argparse, json, math, os, pickle, signal, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jax
import jax.numpy as j
import numpy as np

from types import SimpleNamespace
from . import learner as loop, model
from .optimizer import adam_empty
from .protocol import ACT, CAT, MAGIC, NODE, NUM, TOK, ROOT, layout, outcome, make_policy, to_device
C = ROOT

TASK = 'native-selfplay'


class Workers:
    """W Node OU actor processes with G game slots each; binary synchronous frames (driver.py protocol)."""

    def __init__(self, W, G, seed, mode='selfplay', log_dir=None, *, teams=None, fmt='gen9randombattle', featured=None, featured_share=0., deterministic_teams=False):
        self.W, self.G = W, G
        self.off, self.size = layout(G)
        env = dict(os.environ, TSX_DISABLE_CACHE='1', NODE_OPTIONS='--max-old-space-size=2048')
        log_dir = Path(log_dir) if log_dir else None
        extra = [] if fmt == 'gen9randombattle' else ['--teams', str(Path(teams).resolve()), '--format', fmt]
        actor_script = 'actor.ts' if fmt == 'gen9randombattle' else 'actor-ou.ts'
        if deterministic_teams and fmt == 'gen9randombattle': extra += ['--deterministic-teams']
        if featured and mode == 'selfplay':
            extra += ['--featured', str(Path(featured).resolve()), '--featured-share', str(featured_share)]
        self.procs = []
        for w in range(W):
            err = open(log_dir / f'actor-{mode}-{w}.log', 'ab') if log_dir else subprocess.DEVNULL
            self.procs.append(subprocess.Popen([NODE, '--import', 'tsx', str(C / 'src' / actor_script), str(G), str(seed * 1000 + w), mode, *extra],
                                               cwd=str(C), env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, bufsize=0))
            if hasattr(err, "close"): err.close()
        self.buf = [bytearray(16 + self.size) for _ in range(W)]
        self.pool = ThreadPoolExecutor(max(1, W))
        B = W * G
        self.frame = dict(finished=np.zeros(B, np.int8), winner=np.zeros(B, np.int8), turns=np.zeros(B, np.int16),
                          rejected=np.zeros((B, 2), np.int8), actor=np.zeros((B, 2), np.int8), mask=np.zeros((B, 2, ACT), np.uint8),
                          cat=np.zeros((B, 2, TOK, CAT), np.int16), num=np.zeros((B, 2, TOK, NUM), np.uint16))

    def _read(self, w):
        p, b = self.procs[w], self.buf[w]
        view = memoryview(b); got = 0
        while got < len(b):
            n = p.stdout.readinto(view[got:])
            if not n:
                raise RuntimeError(f'actor {w} exited (code {p.poll()})')
            got += n
        head = np.frombuffer(b, np.int32, 4)
        if head[0] != MAGIC or head[2] != self.G or head[3] != self.size:
            raise RuntimeError(f'actor {w} bad frame header {head.tolist()}')
        G, o, body = self.G, self.off, memoryview(b)[16:]
        s = slice(w * G, (w + 1) * G); f = self.frame
        f['finished'][s] = np.frombuffer(body, np.int8, G, o['finished'])
        f['winner'][s] = np.frombuffer(body, np.int8, G, o['winner'])
        f['turns'][s] = np.frombuffer(body, np.int16, G, o['turns'])
        f['rejected'][s] = np.frombuffer(body, np.int8, 2 * G, o['rejected']).reshape(G, 2)
        f['actor'][s] = np.frombuffer(body, np.int8, 2 * G, o['actor']).reshape(G, 2)
        f['mask'][s] = np.frombuffer(body, np.uint8, 2 * G * ACT, o['mask']).reshape(G, 2, ACT)
        f['cat'][s] = np.frombuffer(body, np.int16, 2 * G * TOK * CAT, o['cat']).reshape(G, 2, TOK, CAT)
        f['num'][s] = np.frombuffer(body, np.uint16, 2 * G * TOK * NUM, o['num']).reshape(G, 2, TOK, NUM)

    def read(self, timeout=300):
        futures = [self.pool.submit(self._read, w) for w in range(self.W)]
        for f in futures:
            f.result(timeout=timeout)          # raises on a dead or stalled actor
        return self.frame

    def write(self, actions):
        a = np.asarray(actions, np.int8).reshape(self.W, self.G * 2)
        for w, p in enumerate(self.procs):
            p.stdin.write(a[w].tobytes())

    def close(self):
        for p in self.procs:
            try: p.stdin.close()
            except Exception: pass
        for p in self.procs:
            try: p.wait(timeout=10)
            except subprocess.TimeoutExpired: p.kill(); p.wait()
        self.pool.shutdown(wait=False, cancel_futures=True)
        for p in self.procs:
            if p.stdout: p.stdout.close()
            if p.stderr: p.stderr.close()


def evaluate(p_a, opponent, games, *, seed, teams=None, fmt='gen9randombattle', W=4, G=32, p_b=None, log_dir=None, timeout_s=1800, config=model.SMALL, opponent_config=None, mixed=True, deterministic_teams=False):
    """Play `games` completed games of policy A versus `opponent` on `teams` (both sides draw from it):
    'heuristic' or 'random' (in-worker baselines; A plays p1 in even slots, p2 in odd slots), or
    'policy' (A versus parameters p_b with sides mirrored by slot parity). Returns A's record."""
    policy = make_policy(config); policy_b = make_policy(opponent_config or config)
    mode = 'selfplay' if opponent == 'policy' else opponent
    ws = Workers(W, G, seed, mode, log_dir, teams=teams, fmt=fmt, deterministic_teams=deterministic_teams)
    # Actors alternate sides within each worker, including when G is odd.
    B = W * G; a_side = (np.arange(B) % G) % 2
    key = jax.random.PRNGKey(seed); wins = losses = ties = capped = abandoned = done = 0; turns = 0; t0 = time.time()
    try:
        frame = ws.read()
        while done < games and time.time() - t0 < timeout_s:
            cat, num, mask, actor = to_device(frame)
            key, k1, k2 = jax.random.split(key, 3)
            act_a = np.asarray(jax.device_get(policy(p_a, cat, num, mask, actor, k1, mixed=mixed)[0]))
            if opponent == 'policy':
                act_b = np.asarray(jax.device_get(policy_b(p_b, cat, num, mask, actor, k2, mixed=mixed)[0]))
                sel = (np.arange(2)[None, :] == a_side[:, None])
                act = np.where(sel, act_a, act_b)
            else:
                act = act_a
            ws.write(act)
            frame = ws.read()
            f = frame['finished']; w = frame['winner'].astype(np.int32)
            nat = f == 1
            wins += int((nat & (w == a_side)).sum()); losses += int((nat & (w == 1 - a_side)).sum())
            ties += int((nat & (w < 0)).sum()); capped += int((f == 2).sum()); abandoned += int((f == 3).sum()); done += int(nat.sum())
            turns += int(frame['turns'][nat].sum())
    finally:
        ws.close()
    n = max(1, wins + losses + ties)
    return dict(opponent=opponent, format=fmt, games=wins + losses + ties, wins=wins, losses=losses, ties=ties, capped=capped,
                abandoned=abandoned, score=(wins + 0.5 * ties) / n, mean_turns=turns / max(1, done), wall_s=time.time() - t0)


def load_params(path):
    return jax.tree.map(j.asarray, pickle.load(open(path, 'rb'))['parameters'])


def train(args):
    """Actors on the main thread; PPO on a learner thread (one segment of policy lag at most)."""
    import queue
    policy = make_policy(args.model_config)
    L = SimpleNamespace(**vars(loop)); L.update = loop.make_update(args.model_config)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True); (out / 'checkpoints').mkdir(exist_ok=True)
    jax.config.update('jax_compilation_cache_dir', str(out / 'jax-cache'))
    mixed = args.mixed
    if args.resume:
        p, adam, meta0 = L.load_checkpoint(args.resume)
        seg0, games0 = meta0['segment'] + 1, meta0['completed_games']
    elif args.init:
        p = load_params(args.init); adam = adam_empty(p); seg0, games0 = 0, 0
    else:
        p = model.initialize(args.vocabulary_size, args.seed, args.model_config)
        adam = adam_empty(p); seg0, games0 = 0, 0
    baseline_path = args.baseline or args.init
    baseline = load_params(baseline_path) if baseline_path else jax.tree.map(j.copy, p)
    baseline_config = checkpoint_model_config(baseline_path) if baseline_path else args.model_config
    initial_step = int(jax.device_get(adam.step))
    initial_host = jax.device_get(p)
    W, G, T = args.workers, args.slots, args.horizon
    B = W * G
    if args.minibatch > T * B * 2:
        raise SystemExit(f'minibatch {args.minibatch} exceeds segment rows {T * B * 2}')
    log = open(out / 'segments.jsonl', 'a'); evals = open(out / 'evals.jsonl', 'a')
    (out / 'run-meta.json').write_text(json.dumps(dict(task=TASK, args=serial_args(args), device=str(jax.devices()[0]),
                                                      jax=jax.__version__, segment0=seg0, games0=games0, pid=os.getpid()), indent=1))
    stop = {'flag': False, 'error': None}
    signal.signal(signal.SIGTERM, lambda *a: stop.__setitem__('flag', True))
    t_start = time.time(); deadline = t_start + 60 * args.minutes
    totals = dict(games=0, rows=0, steps=0, wins_p1=0, ties=0, capped=0, abandoned=0, turns=0)
    window = dict(games=0, rows=0, wall=0.)
    shared = dict(p=p, version=0, games=games0, segment=seg0 - 1)
    learner_log = queue.Queue(); learn_q = queue.Queue(maxsize=1)
    trained_unique_rows = [0]

    def save(path_name, p_s, adam_s, seg_s):
        L.save_checkpoint(out / 'checkpoints' / path_name, p_s, adam_s,
                          dict(segment=seg_s, completed_games=shared['games'], utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                               format=args.format, encoder='v21' if args.format == 'gen9randombattle' else 'v22', model_config=args.model_config.to_dict(), vocabulary_sha256=args.vocabulary_sha256))

    def learner():
        nonlocal adam
        p_l = None; last_ckpt = time.time(); last_snap = time.time(); key_l = jax.random.PRNGKey(args.seed + 17 + seg0)
        try:
            while True:
                item = learn_q.get()
                if item is None: break
                seg_i, data, n_elig = item
                if os.environ.get('JAX_NATIVE_TEST_LEARNER_FAIL'):   # explicit fault-injection double for tests only
                    raise RuntimeError('injected learner failure (test)')
                tl = time.time()
                p_l = jax.tree.map(j.copy, shared['p'])          # private copy: donation never touches published params
                ms = []; nmb = math.ceil(n_elig / args.minibatch)
                for e in range(args.epochs):
                    key_l, sub = jax.random.split(key_l)
                    perm = L.permutation(data['eligible'], sub)
                    for kk in range(nmb):
                        p_l, adam, m = L.update(p_l, adam, data, perm, j.int32(kk), minibatch=args.minibatch, mixed=mixed)
                        ms.append(m)
                        # One queued learner step at most, so actor inference never waits behind a whole epoch.
                        jax.block_until_ready(m['total'])
                mh = jax.device_get(ms); del data
                finite = all(bool(x['finite']) for x in mh)
                mean = lambda k2: float(np.mean([x[k2] for x in mh])) if mh else None
                record = dict(segment=seg_i, learn_s=time.time() - tl, adam_steps=len(mh), adam_step_total=int(jax.device_get(adam.step)),
                                     finite=finite, loss=mean('total'), policy=mean('policy'), value=mean('value'), entropy=mean('entropy'),
                                     KL=mean('KL'), clip=mean('clip_fraction'))
                record['eligible_unique_rows'] = n_elig
                record['published'] = finite
                with (out / 'learner.jsonl').open('a') as learner_file:
                    learner_file.write(json.dumps(record) + '\n')
                learner_log.put(record)
                if finite: trained_unique_rows[0] += n_elig
                if not finite:
                    stop['error'] = 'nonfinite PPO metrics at segment %d; params not published' % seg_i; stop['flag'] = True; break
                shared['p'] = p_l; shared['version'] += 1; shared['segment'] = seg_i
                now = time.time()
                if now - last_ckpt >= args.checkpoint_every_s:
                    save('latest.pkl', p_l, adam, seg_i); last_ckpt = now
                if now - last_snap >= args.snapshot_every_s:
                    save(f'snapshot-{seg_i:08d}.pkl', p_l, adam, seg_i); last_snap = now
        except BaseException as e:
            stop['error'] = repr(e); stop['flag'] = True
            import traceback; traceback.print_exc()

    def run_evals(tag, seg_e):
        for opp in args.eval_opponents.split(','):
            if not opp: continue
            r = evaluate(shared['p'], opp, args.eval_games, seed=args.seed + 7919 * (seg_e + 1) + len(opp), teams=args.eval_teams,
                         fmt=args.format, p_b=baseline, config=args.model_config, opponent_config=baseline_config, mixed=mixed, deterministic_teams=args.deterministic_teams, log_dir=out, W=args.eval_workers, G=args.eval_slots)
            r.update(tag=tag, segment=seg_e, params_version=shared['version'], utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                     total_games=shared['games'], policy_baseline=Path(baseline_path).name if baseline_path else 'initializer')
            evals.write(json.dumps(r) + '\n'); evals.flush(); print('EVAL', json.dumps(r), flush=True)

    if args.eval_games > 0 and args.eval_at_start:
        run_evals('start', seg0)
    last_eval = time.time()
    th = threading.Thread(target=learner, daemon=True); th.start()
    workers = Workers(W, G, args.seed + seg0, 'selfplay', out, teams=args.teams, fmt=args.format,
                      featured=args.featured, featured_share=args.featured_share, deterministic_teams=args.deterministic_teams)
    key = jax.random.PRNGKey(args.seed + seg0)
    seg = seg0; last = {}
    try:
        frame = workers.read()
        key, k = jax.random.split(key)
        cat, num, mask, actor = to_device(frame)
        action, logp, value, legal = policy(shared['p'], cat, num, mask, actor, k, mixed=mixed)
        workers.write(jax.device_get(action))
        last_value = np.zeros((B, 2), np.float32)
        seg = seg0; last = {}
        while not stop['flag'] and time.time() < deadline and seg - seg0 < args.max_segments:
            t0 = time.time(); rows, outs = [], []; seg_games = 0; seg_turns = 0; sim_wait = 0.; version = shared['version']
            for t in range(T):
                row = dict(categorical=cat, numeric=num, legal=legal, action=j.maximum(action, 0), old_logp=logp, old_value=value, actor=actor)
                a_host = np.asarray(jax.device_get(actor)); v_host = np.asarray(jax.device_get(value))
                last_value = np.where(a_host, v_host, last_value)
                tw = time.time(); frame = workers.read(); sim_wait += time.time() - tw
                fin, reward, rej = outcome(frame)
                nat = frame['finished'] == 1
                seg_games += int(nat.sum()); seg_turns += int(frame['turns'][nat].sum())
                totals['wins_p1'] += int((nat & (frame['winner'] == 0)).sum()); totals['ties'] += int((nat & (frame['winner'] < 0)).sum())
                totals['capped'] += int((frame['finished'] == 2).sum()); totals['abandoned'] += int((frame['finished'] == 3).sum())
                last_value[fin] = 0.
                outs.append(dict(eligible=j.asarray(a_host & ~rej), finished=j.asarray(fin), reward=j.asarray(reward)))
                rows.append(row)
                key, k = jax.random.split(key)
                cat, num, mask, actor = to_device(frame)
                action, logp, value, legal = policy(shared['p'], cat, num, mask, actor, k, mixed=mixed)
                workers.write(jax.device_get(action))
            a_now = np.asarray(jax.device_get(actor))
            boot = j.asarray(np.where(a_now, np.asarray(jax.device_get(value)), last_value))
            R, O = L.stack(rows, outs); del rows, outs
            adv, ret = L.gae(R, O, boot, args.lam)
            data = L.flatten(R, O['eligible'], adv, ret); del R, adv, ret
            n_elig = int(jax.device_get(data['eligible'].sum()))
            tq = time.time()
            while True:
                if not th.is_alive():
                    stop['error'] = stop['error'] or 'learner thread exited'; stop['flag'] = True; break
                try:
                    learn_q.put((seg, data, n_elig), timeout=5); break
                except queue.Full:
                    continue
            queue_wait = time.time() - tq; del data
            if stop['flag'] and stop['error']:
                break
            while not learner_log.empty(): last = learner_log.get()
            wall = time.time() - t0
            totals['games'] += seg_games; totals['rows'] += n_elig; totals['turns'] += seg_turns
            shared['games'] = games0 + totals['games']
            warm = seg - seg0 < args.warmup_segments
            if not warm:
                window['games'] += seg_games; window['rows'] += n_elig; window['wall'] += wall
            rec = dict(segment=seg, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), wall_s=wall, sim_wait_s=sim_wait, queue_wait_s=queue_wait,
                       games=seg_games, rows=n_elig, behavior_params_version=version, mean_turns=seg_turns / max(1, seg_games), warmup=warm,
                       games_per_s=seg_games / wall, window_games_per_s=window['games'] / window['wall'] if window['wall'] else None,
                       total_games=shared['games'], capped_total=totals['capped'], abandoned_total=totals['abandoned'], learner=last)
            log.write(json.dumps(rec) + '\n'); log.flush()
            print(json.dumps({k2: rec[k2] for k2 in ('segment', 'wall_s', 'sim_wait_s', 'queue_wait_s', 'games', 'rows', 'games_per_s', 'window_games_per_s')}
                             | {k2: last.get(k2) for k2 in ('learn_s', 'adam_step_total', 'loss', 'entropy', 'KL')}), flush=True)
            if args.eval_games > 0 and time.time() - last_eval >= args.eval_every_s:
                run_evals('periodic', seg); last_eval = time.time()
            seg += 1
    finally:
        end_wait = time.time() + 600
        while th.is_alive() and time.time() < end_wait:
            try:
                learn_q.put(None, timeout=5); break
            except queue.Full:
                continue
        th.join(timeout=max(1., end_wait - time.time()))
        workers.close()
        if th.is_alive():
            raise RuntimeError('Learner did not stop within the shutdown deadline')
        if stop['error'] is None:
            save('latest.pkl', shared['p'], adam, shared['segment'])
    if stop['error']:
        print('STOPPED_WITH_ERROR', stop['error'], flush=True)
    totals['steps'] = int(jax.device_get(adam.step)) - initial_step
    totals['trained_unique_rows'] = trained_unique_rows[0]
    changed = any(not np.array_equal(x, y) for x, y in zip(jax.tree.leaves(initial_host), jax.tree.leaves(jax.device_get(shared['p']))))
    summary = dict(task=TASK, kind='NATIVE_SHOWDOWN_TRAINING_RUN', args=serial_args(args), segments=seg - seg0, totals=totals,
                   steady_window=window, steady_games_per_s=window['games'] / window['wall'] if window['wall'] else None,
                   steady_rows_per_s=window['rows'] / window['wall'] if window['wall'] else None,
                   all_inclusive_wall_s=time.time() - t_start, all_inclusive_games_per_s=totals['games'] / (time.time() - t_start),
                   parameters_changed=changed, adam_step_final=int(jax.device_get(adam.step)), params_version=shared['version'],
                   error=stop['error'])
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print('SUMMARY', json.dumps(summary), flush=True)
    if stop['error']:
        sys.exit(3)


def serial_args(args):
    return {k: (v.to_dict() if isinstance(v, model.ModelConfig) else v) for k,v in vars(args).items()}


def checkpoint_model_config(path):
    with open(path, 'rb') as f:
        return model.checkpoint_config(pickle.load(f))
