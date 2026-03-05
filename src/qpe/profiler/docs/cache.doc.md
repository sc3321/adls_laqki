# ProfileCache

`ProfileCache` stores per-layer profiling results on disk using a hierarchical directory layout.

## Purpose

- Keep profile files human-browsable by GPU, model, layer, and batch size.
- Track completion status for partial and full profiling runs.
- Reuse cached records when all required precision files already exist.

## Directory Layout

Root: `root_dir` (default `.qpe_cache/profiles`)

- `v{qpe_version}/`
  - `{gpu_name_sanitized}/`
    - `_precisions.json`
    - `{model_id_sanitized}/`
      - `_status.json`
      - `{layer_name_sanitized}/`
        - `_meta.json`
        - `bs_{batch_size}/`
          - `{precision}.json`

Example path:
- `.qpe_cache/profiles/v2.0/NVIDIA_Tesla_T4/meta-llama--Llama-3-8B/model.layers.0.self_attn.q_proj/bs_1/FP16.json`

## Sanitization Rules

- GPU name: spaces, `/`, `\`, `:` become `_`.
- Model id: `/` and `\` become `--`; spaces and `:` become `_`.
- Layer name: `/`, `\`, `:` become `_`.

## Metadata Files

- `_precisions.json`:
  - Stored at GPU level.
  - Contains the expected precision set for completeness checks.

- `_meta.json`:
  - Stored at layer level.
  - Contains layer metadata such as shape, dtype, and param count.

- `_status.json`:
  - Stored at model level.
  - Cached rollup of completion status by layer and batch size.

## Completion Semantics

- Batch complete:
  - `is_batch_complete(model_id, layer_name, batch_size)` is true when that batch directory has all expected precision files from `_precisions.json`.

- Layer complete:
  - `is_layer_complete(model_id, layer_name)` is true when at least one batch exists and every discovered batch for that layer is complete.

- Model complete:
  - `is_model_complete(model_id)` is true when at least one layer exists and every discovered layer is complete.

`verify(model_id)` rebuilds `_status.json` by scanning files on disk, so stale or corrupted status data can be recovered.

## Main API

- `put(model_id, layer_name, batch_size, precision, data, layer_meta)`
- `get(model_id, layer_name, batch_size, precision)`
- `is_batch_complete(model_id, layer_name, batch_size)`
- `is_layer_complete(model_id, layer_name)`
- `is_model_complete(model_id)`
- `verify(model_id)`
