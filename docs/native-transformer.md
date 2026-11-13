# Configurable transformer

`selfplay_trainer.model.ModelConfig(width, layers, heads)` defines the model dimensions. Width, layers and heads must be positive integers, and width must divide evenly by heads. Compact49 defaults to 128/4/4. For example:

```sh
python -m selfplay_trainer train --out runs/larger \
  --width 384 --layers 6 --heads 12 --minibatch 256
```

This initializes fresh parameters unless `--init` or `--resume` is supplied. Checkpoints declare dimensions in `meta.model_config`; legacy checkpoints without that field are accepted only with the original tensor shapes. Resume rejects incompatible dimensions before actors start. Opposing policies in evaluation may use different dimensions.

All models use the same v21/v22 vocabulary, 97 source packets packed into 49 tokens, 20 legal action slots, exact-erf GELU and original PPO/Adam recipe. Larger models consume more memory and learner time. The example is a configuration, not a throughput or strength promise. Gradient accumulation and experimental recovery objectives are not included.

The v2 export manifest stores dimensions explicitly. The v1 schema is accepted only for the legacy 128/4/4 model. See README for export and CPU execution details.
