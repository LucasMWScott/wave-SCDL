# Point Identity And Splits

## Canonical Point Identity

The stable nearshore point identifier is the exact `name` string from
`configs/sites.yaml` under `nearshore_sites`.

Use that exact string everywhere:

- split config
- static geometry joins
- target extraction
- point-centric metadata
- evaluation grouping
- notebook diagnostics

Do not:

- use validation-subset position as a point ID
- use `site_index` as a join key
- cast numeric-looking suffixes to integers
- rely on accidental row order without checking `target_sites`

## Canonical Site Order

Canonical site order is the original `nearshore_sites` YAML order after
filtering. Point-centric artifacts should preserve that order when written.

For site-indexed artifacts loaded later by name, `target_sites` is the row-order
contract:

- `point_centric_Y_targets.npz`
- `point_centric_X_static.npz`
- `point_centric_physics.npz`
- `point_centric_X_dynamic_sources.npz`
- `point_centric_source_geometry.npz`
- `point_centric_X_bathy.npz`

## Split Configuration

Supported split keys:

- `data.validation_sites`
- `data.test_sites`
- `data.train_sites`
- `data.val_sites`

Removed keys:

- `data.holdout_sites`
- `split.norac_holdout`

Current behavior:

- train sites = eligible sites minus validation minus test
- validation sites = `data.validation_sites`
- test sites = `data.test_sites`
- if `data.validation_sites` is empty, validation falls back to the legacy temporal mode inside the selected validation sites

## Metadata Expectations

`point_centric_metadata.json` should be the first place to confirm identity
alignment.

Check:

- `point_identity.canonical_site_identifier`
- `point_identity.canonical_site_order`
- `point_identity.site_rows`
- `point_identity.artifact_site_orders`
- `splits.train_sites`
- `splits.val_sites`
- `splits.test_sites`

## Practical Guardrails

- Rebuild artifacts after changing split lists.
- If a site-indexed artifact arrives in a different row order, reindex it by
  `target_sites` before use.
- File discovery for NORAC/NORA3 parameter CSVs must match the exact site name,
  not a loose substring.
