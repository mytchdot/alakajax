# Public interfaces

Install the Python package editable from this checkout and install its own npm dependencies. The checkout may live at any depth inside another repository. It finds its Node actors relative to its own package files; it never looks for a parent repository or its settings.

The stable commands are `selfplay-trainer train`, `eval`, `export` and `status`. `python -m selfplay_trainer` has the same interface. Global `--device cpu|gpu|auto` precedes the command. The supervisor is `python -m selfplay_trainer.supervise start|stop|status`. Run each command with `--help` for flags. File arguments are relative to the caller's working directory; `--out` selects the run or export directory. The supervisor preserves that interpretation for input paths. Keep `PYTHONDONTWRITEBYTECODE=1` set when embedding to suppress Python's automatic source cache writes.

## TypeScript

Use a TypeScript-aware runtime such as the pinned tsx. A consumer can import these files directly or use the npm package exports `selfplay-trainer/encoders/v21`, `selfplay-trainer/encoders/v22` and `selfplay-trainer/legal` after linking the local package.

```ts
import { View } from './trainer/src/encoders/v22.ts';
import { legalActions } from './trainer/src/legal.ts';

const view = new View('p1');
// ownLines contains this player's extracted public channel and private requests.
view.add(ownLines);
const { categorical, numeric, mask } = view.encode();
const actions = legalActions(view.request!);
```

`View`, `vocabulary` and `actionIndex` are exported by both encoder modules. The root `src/index.ts` exports `ViewV21`, `ViewV22`, `vocabulary`, `actionIndex`, `legalActions`, `readRequest` and the `Action`/`BattleRequest` types. v21 serves Random Battles; v22 adds OU observations. A view must receive only its player's information. Stream extraction is the consumer's responsibility.

An encoding contains categorical integers `[97, 8]`, numeric values `[97, 32]` and a boolean legal mask `[20]`. Actors round numeric observations to BF16 using round-to-nearest-even before transport. The ordered vocabulary's SHA-256 is part of the interface. Move actions occupy 0–7, switches 8–13, and team-preview choices 14–19; always obtain action indices through `actionIndex`.

## Python

Import the package before JAX so the default x64 and matrix-precision settings take effect. Select a device through `JAX_PLATFORMS` before importing JAX when embedding.

```python
from selfplay_trainer import model, learner
from selfplay_trainer.optimizer import adam_empty

config = model.ModelConfig(width=128, layers=4, heads=4)
parameters = model.initialize(vocabulary_size=3412, seed=7, config=config)
adam = adam_empty(parameters)
update = learner.make_update(config)
```

`model.forward(parameters, categorical, numeric, legal, mixed, config)` accepts batched source tensors `[B,97,8]`, `[B,97,32]` and `[B,20]`. It returns policy logits `[B,20]`, values `[B]` and the auxiliary opponent head `[B,20]`. `mixed=False` uses FP32 matrix execution for CPU. The CLI selects this fallback automatically.

`learner` exposes acting-row `gae`, `flatten`, `permutation`, `make_update`, `save_checkpoint` and `load_checkpoint`. Updates consume a flattened data dictionary containing `categorical`, `numeric`, `legal`, `action`, `old_logp`, `old_value`, `eligible`, `advantage` and `returns`. `make_update(config)` returns the JIT-compiled `(parameters, adam, data, permutation, minibatch_index, *, minibatch, mixed)` function. It donates the input parameter and Adam buffers; callers must not retain aliases to them. The driver makes private copies before updating published parameters.

`save_checkpoint(Path, parameters, adam, metadata)` writes parameters, both Adam moments and its step atomically. `load_checkpoint(Path)` returns those parameters, Adam and metadata. Checkpoints are trusted Python pickle files. Run metadata records model dimensions, encoder, format and vocabulary; resume restores learner state and counters while starting new actor battles.

`selfplay_trainer.export.export(checkpoint, directory, schema='v2')` writes portable parameter-only `manifest.json` and `weights.bin`. `load_export(directory)` checks the binary hash and returns parameters and the manifest. Tensor offsets count little-endian FP32 elements. v1 supports 128/4/4 consumers; v2 declares configurable dimensions. Neither export contains optimizer state or private provenance.

The binary actor protocol is documented at the top of `src/actor.ts` and implemented by `selfplay_trainer.protocol` and `driver.Workers`. Its current header magic and tensor layout are versioned contracts; changing them requires updating both ends and their parity tests.
