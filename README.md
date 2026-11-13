# Native self-play trainer

Run native Pokémon Showdown battles in Node actors and train a policy/value transformer with JAX PPO. Gen 9 Random Battles uses freshly generated teams and encoder v21. Gen 9 OU uses your validated team pool and encoder v22. The default model is compact49 (width 128, four layers, four heads); larger dimensions are configurable.

This repository contains training tooling, baseline opponents, evaluation and export. It has no online bot client or ladder automation. No trained weights, team datasets or private experiment results are included.

## Setup

Use Node **24.21.0** and Python **3.12.3** on Linux or WSL. Install Node with your preferred version manager (`nvm use` reads `.nvmrc`). From this directory:

```sh
npm ci --ignore-scripts
python3.12 -m venv .venv
. .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation --no-deps -e .
npm run typecheck
npm test
python -m unittest discover -s tests -p 'test_*.py'
```

The CPU lock uses JAX/JAXlib 0.12.0.1 and NumPy 2.4.0. The Showdown version is pinned to 0.11.11 because its ordered vocabulary and battle mechanics are part of the model interface. `--ignore-scripts` skips optional native modules that these actors do not use. Keep the editable Python installation: actors and npm dependencies live beside the Python package in this checkout.

For a compatible NVIDIA GPU, install `requirements-gpu.txt` instead of the CPU requirements. CUDA support depends on JAX's supported driver/platform matrix. CPU commands select FP32 matrix execution automatically; numeric actor observations retain their original BF16 rounding. GPU commands select the original BF16/BF16/FP32 execution. Override with `--precision fp32` or `--precision bf16` when explicitly comparing execution modes. No GPU is required for tests.

## Tiny CPU run

```sh
python -m selfplay_trainer --device cpu train --out runs/smoke \
  --width 16 --layers 1 --heads 1 --workers 1 --slots 2 \
  --horizon 4 --minibatch 8 --epochs 1 --max-segments 3 --minutes 5
python -m selfplay_trainer status --out runs/smoke
```

This is a plumbing check with a small, randomly initialized model. It is not a strength test. A run writes logs, JAX compilation cache and checkpoints only beneath `--out`; team files are read only. Stop gracefully with SIGTERM or use the supervisor's STOP command. A CPU with little memory should use one actor and small batches.

## Random Battles training

```sh
python -m selfplay_trainer train --out runs/random \
  --workers 10 --slots 128 --horizon 16 --minibatch 2048 --epochs 2 \
  --minutes 60 --eval-games 500 --eval-at-start
```

The unqualified command uses JAX's available device. Add `--device cpu` before `train` to force CPU. New runs initialize random parameters; `--init CHECKPOINT` initializes parameters from your own checkpoint with fresh Adam. `--resume CHECKPOINT` restores parameters, Adam moments/step and run counters. Resume starts fresh actor battles and RNG streams based on the resumed segment, as in the original pipeline; it does not restore an interrupted battle or reproduce an uninterrupted trajectory.

```sh
python -m selfplay_trainer train --out runs/random \
  --resume runs/random/checkpoints/latest.pkl --minutes 60
```

Set `--baseline CHECKPOINT` for periodic head-to-head evaluation against an earlier checkpoint. Otherwise a new run compares against its initializer, and a resumed run against its restored policy. `--eval-every-s` defaults to four hours; checkpoints default to 15 minutes and snapshots to six hours. Evaluations use separate actor seeds and alternate sides by slot. They can finish a few games beyond the requested count because slots complete in parallel. Logs distinguish natural completions, capped and abandoned games. An OU encoder can skip unsupported protocol lines; diagnostics count those skips. Abandoned games end with zero reward. These are the original pipeline's behaviors.

## OU with your own teams

Supply Showdown team exports separated by `---` or `=== team name ===`, or packed teams with one team per line. Normalize, validate under native Showdown's format rules, deduplicate and optionally split them locally:

```sh
npm run teams -- --input my-teams.txt --out runs/pools/train.txt \
  --eval-out runs/pools/heldout.txt --heldout-fraction 0.1 --seed 17
python -m selfplay_trainer train --format gen9ou --out runs/ou \
  --teams runs/pools/train.txt --eval-teams runs/pools/heldout.txt \
  --eval-games 500 --minutes 60
```

The split needs enough unique teams to produce at least one held-out team. No team data is downloaded automatically. If you independently download a public dataset, check its license and terms first; no dataset license is supplied by this project's GPL license. The optional `--featured FILE --featured-share 0.25` uses your packed teams on one side of some self-play games. Evaluation keeps its own pool.

## Evaluation and portable export

```sh
python -m selfplay_trainer --device cpu eval \
  --checkpoint runs/random/checkpoints/latest.pkl --opponent heuristic \
  --games 20 --workers 1 --slots 2 --output runs/random/heuristic.json
python -m selfplay_trainer --device cpu eval \
  --checkpoint runs/random/checkpoints/latest.pkl --opponent policy \
  --other runs/earlier/checkpoints/latest.pkl --games 500
python -m selfplay_trainer export \
  --checkpoint runs/random/checkpoints/latest.pkl --out outputs/policy
```

OU evaluation additionally needs `--teams HELDOUT_POOL`. The manifest records dimensions, format, vocabulary hash, ordered tensor shapes and **element** offsets; `weights.bin` contains little-endian FP32 parameters. The default schema is `psai-native-transformer-v2`; `--schema v1` supports existing 128/4/4 consumers. `selfplay_trainer.export.load_export()` validates the hash and reconstructs parameters. Checkpoint files use Python pickle: load only checkpoints you trust. Exports contain no optimizer or machine-specific provenance.

## Long runs

```sh
python -m selfplay_trainer.supervise start --out runs/long --hours 24 -- \
  train --workers 10 --slots 128
python -m selfplay_trainer.supervise status --out runs/long
python -m selfplay_trainer.supervise stop --out runs/long
```

The supervisor holds an output-directory lock, sends graceful termination to its own driver and resumes the latest durable checkpoint after a failed driver (three retries by default). Its original deadline survives supervisor restarts. A successful bounded driver exits the supervisor. To resume after STOP, remove only that run's `STOP` file and start again. Inspect failure logs before doing so. Separate output directories can run independently; reserve your own hardware resources.

All supervisor file arguments, including team pools and initial/baseline checkpoints, resolve relative to the caller's directory. This also works when the checkout is nested inside another repository.

## Throughput and hardware

The original small-model native pipeline was measured at roughly **150 completed Random Battles games/s** and **110–128 OU games/s** on an RTX 5070 with a Ryzen 5 9600X. These historical figures describe the source pipeline, not a fresh benchmark of this extracted release. They depend on game lengths, model dimensions, batching, evaluation and checkpoint costs. The CPU smoke run is for correctness. No playing-strength or ladder-rank claim follows from throughput.

## Public interface and embedding

The checkout can be a submodule such as `trainer/` inside another repository. Run setup in that directory and install it editable into the consuming environment. Paths supplied on the CLI are relative to the caller's working directory; actors resolve resources relative to this repository. Only configured run/export outputs are written.

- CLI: `selfplay-trainer` or `python -m selfplay_trainer`: `train`, `eval`, `export`, `status`.
- Python: `selfplay_trainer.model` (`ModelConfig`, `initialize`, `forward`), `learner` (`make_update`, `gae`, `flatten`, checkpoints), `optimizer`, `protocol`, `export`.
- TypeScript, with a TS-aware runtime such as tsx: `src/index.ts`, `src/encoders/v21.ts`, `src/encoders/v22.ts`, `src/legal.ts`. The npm package exports the same encoder and legal-action modules. `View` accepts only one player's extracted protocol channel and their private requests; never feed omniscient battle logs.

See [public interfaces](docs/public-interface.md) for import examples and tensor contracts. Keep `PYTHONDONTWRITEBYTECODE=1` in an embedding process's environment to prevent Python from creating interpreter cache files in a read-only checkout.

To add a singles format, validate its team rules, verify encoder coverage using native battle logs, add its CLI format choice and route to the fixed-team actor. Keep the observation dimensions, ordered vocabulary, action mapping and private-information boundary explicit. A format accepted by Showdown is not automatically encoder-qualified. See [fidelity](docs/fidelity.md) and [license notices](THIRD_PARTY_NOTICES.md).

All original project code is GPL-3.0-only; see LICENSE and third-party notices.

Random Battles originally seeds battle mechanics but leaves native team generation entropy-seeded. This remains the default for fidelity. Use `--deterministic-teams` with `--seed` to seed both team generators explicitly; that opt-in mode is deterministic for the pinned simulator and action sequence. OU team sampling already follows the actor seed. Fixed-trajectory tests seed team generation explicitly.

For Python embedding, import `selfplay_trainer` before importing JAX, or set `JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest NVIDIA_TF32_OVERRIDE=0` explicitly. The CLI preserves these original numeric defaults, including x64 random permutation draws; parameter tensors remain FP32.
