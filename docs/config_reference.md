# Config Reference

## Site Splits

Site-heldout behavior is configured only through `data.validation_sites` and
`data.test_sites`.

```yaml
data:
  train_sites: []          # optional allow-list; empty means all eligible sites
  val_sites: []            # optional legacy temporal-validation allow-list
  validation_sites: []     # full-site heldout validation sites
  test_sites: []           # full-site heldout final test sites
split:
  train: 0.8
  val: 0.2
  test: 0.0
  datetime_column: time
```

Rules:

- `data.validation_sites` and `data.test_sites` must be mutually exclusive
- training sites are all eligible sites minus validation minus test
- `data.holdout_sites` and `split.norac_holdout` are removed and rejected at runtime
- site identifiers must exactly match `configs/sites.yaml` nearshore `name` values
- temporal fractions under `split.*` still control row-wise chronology inside the selected sites

## Model: `coastal_transformer`

```yaml
model:
  architecture: coastal_transformer
  coastal_transformer:
    sequence_encoder:
      type: transformer  # transformer | lstm
      transformer:
        model_dim: 128
        num_layers: 3
        num_heads: 4
        ff_multiplier: 3.0
        attn_dropout: 0.2
        ff_dropout: 0.25
        rope_base: 10000.0
        use_sdpa: true
      lstm:
        hidden_dim: 128
        num_layers: 2
        bidirectional: false
        dropout: 0.2
        pooling: last  # last | mean | attention
        layer_norm: true
```

### Temporal encoder behavior

- `sequence_encoder.type` defaults to `transformer` when omitted.
- Both temporal encoders return `dynamic_tokens` with shape `[B, T, model_dim]`.
- Canonical transformer keys live under
  `model.coastal_transformer.sequence_encoder.transformer.*`.

### LSTM options

- `hidden_dim`: LSTM hidden size per direction.
- `num_layers`: number of stacked LSTM layers (`>= 1`).
- `bidirectional`: if true, LSTM runs forward+backward and projects back to `model_dim`.
- `dropout`: LSTM/intermediate dropout (for `num_layers=1`, recurrent dropout is internally forced to `0.0` by PyTorch behavior).
- `pooling`: optional pooled summary mode (`last`, `mean`, `attention`).
- `layer_norm`: apply `LayerNorm` after output projection.

### Branch switches

- `coastal_transformer.use_static` overrides `data.use_static_features` when present.
- `coastal_transformer.bathy.enabled` controls bathymetry branch (and still requires `data.use_bathymetry=true`).
- `coastal_transformer.experts.enabled` controls expert heads.
- `coastal_transformer.multi_source.enabled` controls model-side multi-source path.
  Runtime keeps existing auto-enable compatibility when data multi-source is enabled and source tensors are available.
- `model.coastal_transformer.multi_source.use_geometry_features` enables the
  per-source `source_geometry` tensor used by the multi-source encoder.

### Bathymetry branch config

```yaml
model:
  coastal_transformer:
    bathy:
      enabled: true
      version: v2
      in_channels: 6
      patch_size: 128
      resolution_m: 50
      channels:
        - depth
        - land_sea_mask
        - slope_magnitude
        - distance_to_land
        - curvature_laplacian
        - shallow_breaking_mask
      shallow_breaking_depth_m: 15.0
```

- Runtime validates `in_channels` and `patch_size` against `point_centric_X_bathy.npz`.
- Older 2-channel bathy files still load when the config matches their artifact shape.

## Canonical Training Keys

- Optimizer: `training.optimizer.lr`, `training.optimizer.weight_decay`
- Scheduler:
  `training.scheduler.type`,
  `training.scheduler.cosine.*`,
  `training.scheduler.step.*`,
  `training.scheduler.plateau.*`
- Static branch: `model.coastal_transformer.static_branch.*`
- Bathy branch: `model.coastal_transformer.bathy.*`

Only the canonical keys above are read at runtime.

## Transfer Representation

```yaml
data:
  targets:
    mode: transfer
    transfer_reference: weighted_partitioned
    transfer_representation: legacy   # legacy | residual_correction
    residual_correction:
      enabled: false
      bound_method: tanh              # tanh | clamp | none
      max_abs_log_hs: 1.25
      max_abs_tp: 6.0
      max_abs_dir_deg: 120.0
      max_abs_dp_deg: 120.0
      zero_init_output_head: true
```

- `legacy` preserves the prior transfer runtime.
- `residual_correction` keeps the stored transfer artifact contract but trains
  against raw residual targets and reconstructs physical metrics from
  `reference + residual`.

## Static Regularization

```yaml
data:
  static_regularization:
    enabled: true
    noise:
      enabled: true
      std: 0.02
      apply_after_standardization: true
      train_only: true
    group_dropout:
      enabled: true
      p: 0.15
      train_only: true
      mode: per_sample   # per_batch | per_sample
      replacement_value: 0.0
      groups: {}
```

- Applied only on the training loader.
- Operates on runtime `x_static` tensors and keeps stored arrays unchanged.

## Training Sampler

```yaml
training:
  sampler:
    enabled: true
    strategy: site_balanced   # hs_squared | site_balanced
    site_balanced:
      replacement: true
      train_only: true
      min_samples_per_site: 1
      combine_with_hs_weight: false
      hs_power: 2.0
      normalize_within_site: true
```

- `hs_squared` keeps the existing storm-focused weighting.
- `site_balanced` equalizes expected site probability on the training split only.

## Geometry / route curtain config

Configured in `configs/preprocess.yaml`:

```yaml
static_features:
  route_curtain:
    mode: local_k_nearest   # global | local_k_nearest
    use_multi_source_config: true
    padding_m: 2000
    qa_out_path: data/processed/route_curtain_qa.csv
```

- `global` preserves the old single-curtain routing behavior.
- `local_k_nearest` builds one local curtain per NORAC site from the same
  `data.multi_source.k_nearest` NORA3 wave sources used by dynamic preprocessing.
- `use_multi_source_config=true` is the leakage-safe mode because it prevents
  static routing from silently choosing a different source cluster.
