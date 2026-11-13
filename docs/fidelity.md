# Fidelity and boundaries

The release extracts the existing native actor, binary protocol, encoder inheritance chain, compact49/configurable model, GAE, PPO objective and Adam arithmetic. Encoders v21 and v22 depend on shared earlier encoder layers; those layers are encoder code only. No GPU battle-simulator code is included.

The encoder's Dex import now comes directly from `pokemon-showdown`. Only the necessary request types, `readRequest`, ordered `legalActions`, uniform-random action selection and damage heuristic were copied from their shared modules. Their algorithms and ordering remain unchanged. The actor's seed derivation, RNE BF16 numeric packing, legal masks, invalid-choice flags, auto-reset, OU skip/abandon behavior and zero-reward truncations are preserved.

The PPO objective is unchanged: undiscounted acting-row GAE with lambda 0.95, eligible-row minibatch advantage normalization, clipped ratio 0.8–1.2, half squared value loss and entropy coefficient 0.01. The auxiliary head is finite-checked without a target. Adam retains learning rate 0.0003, moments 0.9/0.999, epsilon 1e-8 and bias correction. The same packed 49-token transformer uses exact-erf GELU. All 97 input packets and all numeric fields reach the model. The configurable path preserves the small model's initialization RNG order and math.

Intentional cleanup:

- CLI/configuration replaces fixed executable, cache, checkpoint and team-pool paths. New runs support random initialization; no weights are bundled.
- CPU uses FP32 matrix operations by default. Actor numeric values are still BF16-rounded. This is an explicit execution-mode choice, not a claim of BF16/FP32 identity.
- Model dimensions and vocabulary identity are validated before actors start. Larger-model accumulation and recovery variants are excluded; the original PPO update is used.
- Checkpoints remain parameters plus Adam and counters, with model/format/vocabulary metadata. Resume restarts actors; exact live battle/RNG continuation is not promised by this pipeline.
- Initial actor reads and compilation now enter the cleanup boundary. Killed actor processes are reaped. Shutdown refuses to save if the learner has not joined.
- Export omits machine paths, private task labels and private timeline metadata. Its tensor ordering and little-endian FP32 encoding stay compatible with the v1 and v2 schemas.
- The supervisor is a portable wrapper around the existing train/resume/STOP workflow. No changes are applied to any consuming repository.
- The supervisor resolves team/checkpoint inputs relative to its caller before launching a child from this checkout. Evaluation assigns policy sides by each worker's local slot index, matching the actor even for an odd number of slots; the original even-slot workloads are unchanged. These orchestration repairs do not change training tensors or the learner.

Tests use synthetic native battles and random parameters. Fixed-trajectory encoder digests come from the source v21/v22 encoders, not from this release's implementation. Release preparation also compares byte-identical tensors on fixed native logs and compares the extracted objective, GAE, optimizer and policy with the source functions. These are implementation checks, not playing-strength evidence.

Random Battles originally seeds battle mechanics but leaves native team generation entropy-seeded. This remains the default for fidelity. Use `--deterministic-teams` with `--seed` to seed both team generators explicitly; that opt-in mode is deterministic for the pinned simulator and action sequence. OU team sampling already follows the actor seed. Fixed-trajectory tests seed team generation explicitly.

The CLI carries the source launcher's x64-enabled JAX setting, highest default matmul precision, TF32 override and GPU compiler defaults. x64 affects the default dtype used by random permutation draws; dropping it changes the shuffle sequence. The GPU UUID and all device-specific paths were removed. The importable package sets numeric environment defaults before JAX is imported; embedding applications that import JAX first must set those defaults themselves.
