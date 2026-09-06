# Transformer encoder

[`TransformerEncoder`](../../models/transformer_encoder.py) maps an
[environment observation](environment.md) to point embeddings `H [B,N,d]`
and a global embedding `g [B,d]`. Here `B` counts episodes, `N` points,
and `d` is the embedding width. Decoder, action/value heads, and PPO training
remain unimplemented; see the [research model](../research/model.md).

## Configuration and computation

`load_config(path)` reads the `encoder` section of
[`transformer_gru.yaml`](../../parameters/transformer_gru.yaml) into `EncoderConfig`.
All four fields are required, without defaults: `model_dim` (embedding width),
`num_layers` (depth), `num_heads` (attention heads), and `feedforward_dim`
(feed-forward hidden width).

For coordinate width `D`, each point has `D+4` features, ordered as
`[coordinates, weight, selected, received_signal, received_signal - threshold]`.
The encoder ignores `contributions`.

```text
Features [B,N,D+4] -> Linear -> GELU -> Linear -> [B,N,d]
                 -> pre-norm transformer layers -> LayerNorm -> H
Mean(H, points), steps_remaining/max_steps, feasible
                 -> concatenate [B,d+2] -> Linear -> GELU -> Linear -> g [B,d]
```

Both projections have hidden width `d`. Transformer layers use GELU, zero
dropout, independently initialized weights, and full attention without positional
encoding. Reordering points reorders `H` and leaves `g` unchanged within
floating-point tolerance.

## Usage

Continue from the environment example, using its `config` and `observation`:

```python
from models.transformer_encoder import TransformerEncoder, load_config

encoder = TransformerEncoder(
    config=load_config("parameters/transformer_gru.yaml"),
    coordinate_dim=observation["points"].shape[-1],
    threshold=config.threshold,
    max_steps=config.max_steps,
).to(observation["points"])
H, g = encoder(observation)
```

The caller controls initialization seeds and matches the model's device and
floating dtype to the observation. Forward preserves inputs and autograd.
Batch size and point count may vary between calls; coordinate width stays fixed.
Use positive dimensions and horizon, with `model_dim` divisible by `num_heads`.
There is no input validation or padding support. Shape annotations document
the tensors without adding runtime checks.
