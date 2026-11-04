# dist-s1-prod-check

Validation checks for the OPERA DIST-S1 production system. Cross-references product metadata
(GeoTIFF tags), NASA CMR granule metadata, and the expected enumeration from
[dist-s1-enumerator](https://github.com/opera-adt/dist-s1-enumerator) at global scale.

## Checks

1. **download-metadata** - DIST-S1 + RTC-S1 granule metadata from CMR into geoparquet tables
   (direct CMR queries, temporally chunked and threaded; no per-product API calls).
2. **check-duplicates** - DIST-S1 granules for the same pass (same tile, acquisition times within
   5 minutes). The last-processed granule of a pass is taken as the correct one; every other check
   runs on that granule only.
3. **check-ordering** - each product was processed after the previous product in its MGRS tile,
   using only the last-processed version of each pass. If a pass is reprocessed after the next
   product was made, that next product is flagged (`prev_n_versions` > 1).
4. **check-confirmation** - each product's `prior_dist_s1_product` tag points to the previous
   pass in its tile time series.
5. **check-inputs** - recorded pre/post RTC-S1 inputs match a re-enumeration, burst pass by burst
   pass (`--mode offline` re-enumerates from the local RTC table; `--mode online` hits ASF per product).
6. **check-coverage** - every RTC-S1 pass with baseline imagery has a DIST-S1 product.

Each check writes timestamped CSVs (all results + failures with expected vs. actual) and an
HTML report with a clickable map of failing MGRS tiles (popups show product ids, reasons, and
layer links) and filterable tables with CSV export. `run-all` produces one combined report.

## Setup

```bash
curl -fsSL https://pixi.sh/install.sh | bash   # if pixi is missing
pixi install
```

Earthdata credentials are required (the DIST-S1 CMR collection and the DAAC GeoTIFFs are
gated): put them in `~/.netrc` (`machine urs.earthdata.nasa.gov login ... password ...`) or
let `earthaccess.login()` prompt once.

## Weekend run (checks 1-6, global, Jan-Mar 2026)

```bash
pixi shell
nohup dist-s1-prod-check run-all --start 2026-01-01 --stop 2026-03-01 \
    --data-dir data --out-dir outputs > run_all.log 2>&1 &
```

Everything heavy is resumable: metadata tables are only re-downloaded with `--refresh`, tag
fetching (one DAAC read per product, ~230k globally) caches to `data/dist_s1_tags.parquet`, and
the offline inputs check checkpoints per tile to `data/inputs_check_results.parquet` — a rerun
of the same command picks up where it left off. The inputs check runs in `--inputs-workers`
processes (default 8; raise on a many-core server). `--refresh` clears the inputs checkpoint
along with the metadata tables since results depend on the RTC table. CSVs are written as each
check completes and the HTML report is written even if a later check fails, so a crash never
discards finished work.

Over an AOI instead:

```bash
dist-s1-prod-check run-all --start 2026-01-01 --stop 2026-03-01 --bbox -120 34 -118 36
```

**Fast iteration on a sample** (reuses all serialized data; seconds-to-minutes per run):

```bash
dist-s1-prod-check run-all --start 2026-01-01 --stop 2026-03-01 \
    --data-dir data_global --out-dir outputs_sample --sample-tiles 100
```

`--sample-tiles N` restricts every check to N random MGRS tiles but keeps each tile's full
product time series, so chain-based checks (confirmation, ordering, coverage) stay valid.
Same `--seed` (default 42) means the same tiles every run; change it to see different ones.
Tags fetched for the sample land in the shared cache, so the eventual full run reuses them.

Individual checks after `download-metadata` / `fetch-tags`:

```bash
dist-s1-prod-check download-metadata --start 2026-01-01 --stop 2026-03-01
dist-s1-prod-check fetch-tags --workers 16
dist-s1-prod-check check-duplicates
dist-s1-prod-check check-ordering
dist-s1-prod-check check-confirmation
dist-s1-prod-check check-inputs            # offline, uses the RTC table
dist-s1-prod-check check-inputs --mode online --sample 100   # recent products, live ASF
dist-s1-prod-check check-coverage --start 2026-01-01 --stop 2026-03-01
```

## Notebooks

```bash
pixi run jupyter lab
```

- `notebooks/0_download_and_check.ipynb` - the full workflow as library calls.
- `notebooks/1_investigate_aoi.ipynb` - products + RTC inputs over a bbox.

## Memory at global scale

CMR pages are parsed and discarded as they stream in (never accumulated as raw JSON), the
inputs check enumerates and compares one MGRS tile at a time, and coverage groups are
prefiltered to the check window. Measured: 680 B/row for the RTC table, so the global Jan-Mar
2026 run peaks around 12-16 GB (RTC table ~5 GB + LUT-joined frame ~7 GB + bounded transients).

## Notes

- AOI runs resolve the bbox to its overlapping MGRS tiles and widen the RTC-S1 query to the
  full burst footprints of those tiles (stored in `data/target_mgrs_tiles.txt`); without this,
  edge products appear to use "unexpected" inputs. Antimeridian-crossing AOIs are not handled.
- The first product per tile inside the queried window cannot have its prior validated
  (the true prior falls before the window); it is skipped unless `--expect-none-at-start`
  (mission-start semantics) is passed.
- The HTML report loads Leaflet and basemap tiles from CDNs, so it needs internet when opened;
  all data is embedded in the file.
- Coverage baseline logic mirrors the enumerator defaults: lookback windows at 365/730/1095
  days, 60-day width, max 4/3/3 pre-images per burst, min 1.
- **Polarization**: only dual-polarization RTC-S1 (`VV+VH`, `HH+HV`) is ever used - single-pol
  granules are dropped when the RTC input frame is built and are never post-images or baseline
  imagery. A burst's baseline must also match its post-image: a `VV+VH` post-image can only be
  paired with `VV+VH` pre-images, and likewise for `HH+HV`. Coverage counts a burst as having a
  baseline only when the lookback windows hold same-polarization acquisitions, and the inputs
  check reports `Pre RTC baseline polarization mismatch` for products that violate this.
- **Duplicates**: a pass is one MGRS tile (DIST-S1) or one burst (RTC-S1) with acquisition times
  at most 5 minutes apart, so granule names are not used to find duplicates. One RTC-S1
  acquisition processed twice can carry acquisition times a second apart
  ([opera-sds-pcm#1550](https://github.com/nasa/opera-sds-pcm/pull/1550)). Checks use exactly one
  RTC-S1 per burst pass (the last processed) and one DIST-S1 product per pass (the last
  processed). The inputs check accepts any copy of an expected burst pass and reports
  `Duplicate RTC inputs for one burst pass` when a product uses more than one.
