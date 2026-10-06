import shutil
from pathlib import Path

import click
import geopandas as gpd
import pandas as pd

from dist_s1_prod_check import checks, coverage, inputs, report, tags
from dist_s1_prod_check.constants import (
    DELTA_LOOKBACK_DAYS,
    DELTA_WINDOW_DAYS,
    DIST_S1_PARQUET,
    INPUTS_CHECKPOINT_PARQUET,
    RTC_CHUNKS_DIR,
    RTC_S1_PARQUET,
    TAGS_PARQUET,
    TARGET_TILES_FILE,
    VENUES,
    VENUE_FILE,
    VERTEX_SEARCH_URL,
)


ASF_LINK_COLUMNS = ['asf_post_rtc_url', 'asf_baseline_rtc_url', 'asf_dist_s1_url']
COVERAGE_DESCRIPTION = (
    'RTC-S1 passes with baseline imagery that should have triggered a DIST-S1 product but have none. '
    'The ASF links open Vertex searches over the tile: asf_post_rtc_url for the post RTC-S1 inputs '
    '(post date ±1 day), asf_baseline_rtc_url for the first-anniversary baseline RTC-S1 on the same track(s) '
    f'(post date -{DELTA_LOOKBACK_DAYS[0] + DELTA_WINDOW_DAYS} to -{DELTA_LOOKBACK_DAYS[0]} days), and '
    'asf_dist_s1_url for any DIST-S1 product of this tile (post date ±1 day).'
)


def _stamp() -> str:
    return pd.Timestamp.now().strftime('%Y%m%dT%H%M%S')


def _load_dist(data_dir: Path) -> gpd.GeoDataFrame:
    path = data_dir / DIST_S1_PARQUET
    if not path.exists():
        raise click.ClickException(f'{path} not found; run `dist-s1-prod-check download-metadata` first.')
    return gpd.read_parquet(path)


def _load_rtc(data_dir: Path) -> gpd.GeoDataFrame:
    path = data_dir / RTC_S1_PARQUET
    if not path.exists():
        raise click.ClickException(f'{path} not found; run `dist-s1-prod-check download-metadata` first.')
    return gpd.read_parquet(path)


def _load_tags(data_dir: Path) -> pd.DataFrame:
    path = data_dir / TAGS_PARQUET
    if not path.exists():
        raise click.ClickException(f'{path} not found; run `dist-s1-prod-check fetch-tags` first.')
    return pd.read_parquet(path)


def _all_tile_geoms() -> gpd.GeoDataFrame:
    from dist_s1_enumerator.mgrs_burst_data import get_mgrs_table

    return get_mgrs_table()[['mgrs_tile_id', 'geometry']]


def _resolve_bboxes(bbox: tuple | None, data_dir: Path) -> tuple[tuple | None, tuple | None]:
    """Resolve an AOI bbox to (dist_bbox, rtc_bbox) and persist the target tile list for later checks."""
    from dist_s1_prod_check.cmr import resolve_target_tiles

    tiles_path = data_dir / TARGET_TILES_FILE
    if bbox is None:
        tiles_path.unlink(missing_ok=True)
        return None, None
    tile_ids, rtc_bbox = resolve_target_tiles(bbox)
    tiles_path.write_text('\n'.join(tile_ids))
    click.echo(f'AOI overlaps {len(tile_ids)} MGRS tiles; RTC-S1 bbox widened to {rtc_bbox}')
    return bbox, rtc_bbox


def _pulled_at(data_dir: Path) -> str:
    modified = pd.Timestamp((data_dir / DIST_S1_PARQUET).stat().st_mtime, unit='s')
    return f'{modified:%Y-%m-%d %H:%M} UTC'


def _load_target_tiles(data_dir: Path) -> list[str] | None:
    tiles_path = data_dir / TARGET_TILES_FILE
    if not tiles_path.exists():
        return None
    return tiles_path.read_text().split()


def _build_rtc_frame(df_rtc: gpd.GeoDataFrame, data_dir: Path) -> gpd.GeoDataFrame:
    from dist_s1_prod_check.cmr import build_rtc_input_frame

    return build_rtc_input_frame(df_rtc, mgrs_tile_ids=_load_target_tiles(data_dir))


def _write_csvs(out_dir: Path, stamp: str, csvs: dict[str, pd.DataFrame]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in csvs.items():
        df.drop(columns=[c for c in ['geometry'] if c in df.columns]).to_csv(
            out_dir / f'{name}_{stamp}.csv', index=False
        )


def _write_report(
    out_dir: Path, stamp: str, tile_geoms: gpd.GeoDataFrame, sections: list[dict], title: str, subtitle: str
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = report.write_html_report(sections, tile_geoms, out_dir / f'report_{stamp}.html', title, subtitle)
    click.echo(f'Wrote {html_path}')


def _write_outputs(
    out_dir: Path,
    stamp: str,
    tile_geoms: gpd.GeoDataFrame,
    sections: list[dict],
    csvs: dict[str, pd.DataFrame],
    title: str,
    subtitle: str,
) -> None:
    _write_csvs(out_dir, stamp, csvs)
    _write_report(out_dir, stamp, tile_geoms, sections, title, subtitle)


data_dir_option = click.option('--data-dir', type=click.Path(path_type=Path), default=Path('data'), show_default=True)
out_dir_option = click.option('--out-dir', type=click.Path(path_type=Path), default=Path('outputs'), show_default=True)
venue_option = click.option(
    '--venue',
    type=click.Choice(VENUES, case_sensitive=False),
    default='PROD',
    show_default=True,
    callback=lambda ctx, param, value: value.upper(),
    help='ASF/CMR venue of the DIST-S1 products (RTC-S1 inputs always come from PROD).',
)


def _rtc_chunks_dir(data_dir: Path, rtc_bbox: tuple | None) -> Path:
    suffix = '_' + '_'.join(f'{v:g}' for v in rtc_bbox) if rtc_bbox else ''
    return data_dir / f'{RTC_CHUNKS_DIR}{suffix}'


def _write_rtc(df_rtc: gpd.GeoDataFrame, data_dir: Path, chunks_dir: Path) -> None:
    df_rtc.to_parquet(data_dir / RTC_S1_PARQUET, compression='zstd')
    shutil.rmtree(chunks_dir, ignore_errors=True)


def _filter_processing_time(
    df_dist: gpd.GeoDataFrame, processing_start: str | None, processing_stop: str | None
) -> gpd.GeoDataFrame:
    """Keep products processed in [processing_start, processing_stop), e.g. one campaign in a venue holding several."""
    keep = pd.Series(True, index=df_dist.index)
    if processing_start:
        keep &= df_dist.processing_time >= pd.Timestamp(processing_start)
    if processing_stop:
        keep &= df_dist.processing_time < pd.Timestamp(processing_stop)
    click.echo(f'Processing-time filter kept {int(keep.sum())} of {len(df_dist)} DIST-S1 products')
    return df_dist[keep].reset_index(drop=True)


def _check_venue(data_dir: Path, venue: str) -> None:
    """Refuse to reuse a data dir whose cached tables came from another venue (PROD/UAT product ids can collide)."""
    marker = data_dir / VENUE_FILE
    has_cache = any((data_dir / name).exists() for name in (DIST_S1_PARQUET, TAGS_PARQUET, INPUTS_CHECKPOINT_PARQUET))
    cached = marker.read_text().strip() if marker.exists() else ('PROD' if has_cache else venue)
    if cached != venue:
        raise click.ClickException(f'{data_dir} holds {cached} data; use a separate --data-dir for {venue}.')
    data_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text(f'{venue}\n')


@click.group()
def cli() -> None:
    """Validate the DIST-S1 production system."""


@cli.command()
@click.option('--start', required=True, help='Start of acquisition period, e.g. 2026-01-01.')
@click.option('--stop', required=True, help='End of acquisition period, e.g. 2026-03-01.')
@click.option('--bbox', type=float, nargs=4, default=None, help='AOI as lon_min lat_min lon_max lat_max.')
@click.option(
    '--lookback/--no-lookback', default=True, show_default=True, help='Also fetch RTC-S1 metadata for baseline windows.'
)
@click.option('--skip-rtc', is_flag=True, help='Only fetch DIST-S1 metadata.')
@click.option('--chunk-days', type=int, default=2, show_default=True)
@click.option('--workers', type=int, default=8, show_default=True)
@venue_option
@data_dir_option
def download_metadata(
    start: str,
    stop: str,
    bbox: tuple,
    lookback: bool,
    skip_rtc: bool,
    chunk_days: int,
    workers: int,
    venue: str,
    data_dir: Path,
) -> None:
    """Download DIST-S1 and RTC-S1 granule metadata from CMR into geoparquet tables (check 1)."""
    from dist_s1_prod_check import cmr

    _check_venue(data_dir, venue)
    bbox = tuple(bbox) if bbox else None
    dist_bbox, rtc_bbox = _resolve_bboxes(bbox, data_dir)

    df_dist = cmr.get_dist_s1_table(
        start, stop, bbox=dist_bbox, chunk_days=max(chunk_days, 7), max_workers=workers, venue=venue
    )
    df_dist.to_parquet(data_dir / DIST_S1_PARQUET, compression='zstd')
    click.echo(f'DIST-S1: {len(df_dist)} granules -> {data_dir / DIST_S1_PARQUET}')

    if skip_rtc:
        return
    chunks_dir = _rtc_chunks_dir(data_dir, rtc_bbox)
    if lookback:
        df_rtc = cmr.get_rtc_s1_table_with_lookback(
            start, stop, bbox=rtc_bbox, chunk_days=chunk_days, max_workers=workers, cache_dir=chunks_dir
        )
    else:
        df_rtc = cmr.get_rtc_s1_table(
            start, stop, bbox=rtc_bbox, chunk_days=chunk_days, max_workers=workers, cache_dir=chunks_dir
        )
    _write_rtc(df_rtc, data_dir, chunks_dir)
    click.echo(f'RTC-S1: {len(df_rtc)} granules -> {data_dir / RTC_S1_PARQUET}')


tag_url_template_option = click.option(
    '--tag-url-template',
    default=None,
    help='Read product tags from this location instead of the DAAC URL, formatted per product, e.g. '
    "'s3://BUCKET/PREFIX/{opera_id}/{opera_id}_GEN-DIST-STATUS.tif' (fields: opera_id, mgrs_tile_id, "
    'acq_time, processing_time).',
)


@cli.command()
@click.option('--workers', type=int, default=16, show_default=True)
@click.option('--sample', type=int, default=0, show_default=True, help='Randomly sample N products (0 = all).')
@click.option('--seed', type=int, default=42, show_default=True)
@tag_url_template_option
@data_dir_option
def fetch_tags(workers: int, sample: int, seed: int, tag_url_template: str | None, data_dir: Path) -> None:
    """Fetch product GeoTIFF tags (inputs, prior product) from the DAAC; resumable."""
    df_dist = checks.deduplicate_products(_load_dist(data_dir))
    if sample:
        df_dist = df_dist.sample(n=min(sample, len(df_dist)), random_state=seed)
    df_tags = tags.fetch_tags_table(
        df_dist, data_dir / TAGS_PARQUET, max_workers=workers, url_template=tag_url_template
    )
    n_err = int((df_tags.error != '').sum())
    click.echo(f'Tags: {len(df_tags)} fetched, {n_err} errors -> {data_dir / TAGS_PARQUET}')


def _wkt_by_tile(tile_geoms: gpd.GeoDataFrame) -> dict[str, str]:
    from shapely import to_wkt

    return {row.mgrs_tile_id: to_wkt(row.geometry, rounding_precision=4) for row in tile_geoms.itertuples()}


def _append_tile_wkt(df: pd.DataFrame, wkt_map: dict[str, str]) -> pd.DataFrame:
    df = df.copy()
    df['mgrs_tile_wkt'] = df.mgrs_tile_id.map(wkt_map)
    return df


def _vertex_polygon(wkt: str) -> str:
    from shapely import from_wkt, to_wkt

    geom = from_wkt(wkt)
    if geom.geom_type == 'MultiPolygon':
        geom = max(geom.geoms, key=lambda g: g.area)
    return to_wkt(geom, rounding_precision=4).replace(', ', ',').replace('POLYGON (', 'POLYGON(')


def _vertex_url(product_type: str, polygon: str, start: pd.Timestamp, end: pd.Timestamp, **filters: str) -> str:
    from urllib.parse import quote

    params = {
        'dataset': 'OPERA-S1',
        'productTypes': product_type,
        'polygon': quote(polygon),
        **filters,
        'start': f'{start:%Y-%m-%dT%H:%M:%SZ}',
        'end': f'{end:%Y-%m-%dT%H:%M:%SZ}',
        'resultsLoaded': 'true',
    }
    return VERTEX_SEARCH_URL + '?' + '&'.join(f'{key}={value}' for key, value in params.items())


def _asf_search_links(
    wkt: str | None, mgrs_tile_id: str, track_token: str, post_acq_dt: pd.Timestamp
) -> dict[str, str]:
    if not isinstance(wkt, str) or not wkt:
        return dict.fromkeys(ASF_LINK_COLUMNS, '')
    polygon = _vertex_polygon(wkt)
    post_start, post_end = post_acq_dt - pd.Timedelta(days=1), post_acq_dt + pd.Timedelta(days=1)
    baseline_end = post_acq_dt - pd.Timedelta(days=DELTA_LOOKBACK_DAYS[0])
    baseline_start = baseline_end - pd.Timedelta(days=DELTA_WINDOW_DAYS)
    tracks = sorted(int(track) for track in str(track_token).split('_'))
    return {
        'asf_post_rtc_url': _vertex_url('RTC', polygon, post_start, post_end),
        'asf_baseline_rtc_url': _vertex_url(
            'RTC', polygon, baseline_start, baseline_end, path=f'{tracks[0]}-{tracks[-1]}'
        ),
        'asf_dist_s1_url': _vertex_url('DIST-ALERT-S1', polygon, post_start, post_end, tileID=mgrs_tile_id),
    }


def _issue_summary(df_bad: pd.DataFrame) -> dict | None:
    if df_bad.empty:
        return None
    counts = df_bad.issue_type.value_counts().reset_index()
    counts.columns = ['issue_type', 'n_rows']
    return {
        'title': 'Issue types (click a row to filter the table above)',
        'columns': ['issue_type', 'n_rows'],
        'records': counts.to_dict('records'),
    }


def _duplicates_section(
    df_dist: gpd.GeoDataFrame, wkt_map: dict[str, str]
) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    df_dup = _append_tile_wkt(checks.check_duplicates(df_dist), wkt_map)
    section = report.build_section(
        'duplicates',
        'Duplicates',
        'DIST-S1 granules for the same pass (same tile, acquisition times within 5 minutes). The last-processed '
        'granule of each pass is taken as the correct one; every other check runs on that granule only.',
        df_dup,
        len(df_dist),
    )
    return [section], {'duplicates': df_dup}


def _ordering_section(df_dedup: pd.DataFrame, wkt_map: dict[str, str]) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    df_order = checks.check_processing_order(df_dedup)
    df_bad = _append_tile_wkt(df_order[df_order.out_of_order], wkt_map)
    section = report.build_section(
        'ordering',
        'Processing order',
        'Products processed before the previous product in their MGRS tile, using only the last-processed '
        'version of each pass (see Duplicates). A flag with prev_n_versions > 1 means the previous pass was '
        'reprocessed after this product was made.',
        df_bad,
        int(df_order.evaluated.sum()),
    )
    return [section], {'ordering_failures': df_bad, 'ordering': df_order}


def _confirmation_section(
    df_dedup: pd.DataFrame, df_tags: pd.DataFrame, expect_none_at_start: bool, wkt_map: dict[str, str]
) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    df_conf = checks.check_confirmation(df_dedup, df_tags, expect_none_at_start=expect_none_at_start)
    df_conf = df_conf[df_conf.opera_id.isin(df_tags.opera_id)]
    df_bad = _append_tile_wkt(df_conf[~df_conf.confirmed_ok], wkt_map)
    section = report.build_section(
        'confirmation',
        'Confirmation chain',
        'Products whose prior_dist_s1_product tag does not point to the previous product in the tile time series.',
        df_bad,
        len(df_conf),
    )
    tiles = checks.summarize_by_tile(df_conf.assign(fail=~df_conf.confirmed_ok), 'fail')
    return [section], {'confirmation_failures': df_bad, 'confirmation_tiles': tiles}


def _inputs_section(
    df_tags: pd.DataFrame,
    df_frame: gpd.GeoDataFrame | None,
    mode: str,
    workers: int,
    wkt_map: dict[str, str],
    checkpoint_path: Path | None = None,
) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    if mode == 'offline':
        df_results = inputs.check_inputs_offline(df_tags, df_frame, checkpoint_path=checkpoint_path, workers=workers)
    else:
        df_results = inputs.check_inputs_online(df_tags, max_workers=workers)
    df_bad = _append_tile_wkt(df_results[df_results.inputs_correct != True], wkt_map)  # noqa: E712
    n_checked = df_results.opera_id.nunique()

    post_mask = df_bad.issue_type.str.contains('Post', case=False)
    pre_mask = df_bad.issue_type.str.contains('Pre', case=False) & ~post_mask
    sections = [
        report.build_section(
            'inputs',
            'Product inputs',
            'Products whose recorded pre/post RTC-S1 inputs differ from the re-enumerated expected inputs.',
            df_bad,
            n_checked,
            summary=_issue_summary(df_bad),
        ),
        report.build_section(
            'inputs_post',
            'Inputs: post issues',
            'Subset of product-input issues involving post RTC-S1 inputs (wrong or missing post images).',
            df_bad[post_mask],
            n_checked,
            summary=_issue_summary(df_bad[post_mask]),
        ),
        report.build_section(
            'inputs_pre',
            'Inputs: pre issues',
            'Subset of product-input issues involving pre (baseline) RTC-S1 inputs.',
            df_bad[pre_mask],
            n_checked,
            summary=_issue_summary(df_bad[pre_mask]),
        ),
    ]
    csvs = {
        'inputs_failures': df_bad,
        'inputs_failures_post': df_bad[post_mask],
        'inputs_failures_pre': df_bad[pre_mask],
        'inputs': df_results,
    }
    return sections, csvs


def _coverage_section(
    df_dedup: pd.DataFrame, df_frame: gpd.GeoDataFrame, start: str, stop: str, wkt_map: dict[str, str]
) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    df_groups = coverage.expected_pass_groups(df_frame, start, stop)
    df_cov = coverage.check_coverage(df_groups, df_dedup, df_rtc_frame=df_frame)
    df_missing = df_cov[~df_cov.product_found & df_cov.missing_product_expected]
    df_missing = _append_tile_wkt(df_missing.drop(columns=['burst_ids', 'burst_pols'], errors='ignore'), wkt_map)
    links = [
        _asf_search_links(row.mgrs_tile_wkt, row.mgrs_tile_id, row.track_token, row.post_acq_dt)
        for row in df_missing.itertuples()
    ]
    df_missing = pd.concat([df_missing, pd.DataFrame(links, index=df_missing.index, columns=ASF_LINK_COLUMNS)], axis=1)
    df_orphans = coverage.check_orphan_products(df_groups, df_dedup)

    summary = None
    df_by_tile = pd.DataFrame(columns=['mgrs_tile_id', 'n_missing', 'missing_post_dates'])
    if not df_missing.empty:
        df_by_tile = (
            df_missing.groupby('mgrs_tile_id')
            .agg(
                n_missing=('post_acq_dt', 'size'),
                missing_post_dates=('post_acq_dt', lambda s: ';'.join(pd.to_datetime(s).dt.strftime('%Y-%m-%d'))),
            )
            .reset_index()
            .sort_values('n_missing', ascending=False)
        )
        summary = {
            'title': 'Missing products by tile (click a row to filter the table above)',
            'columns': ['mgrs_tile_id', 'n_missing', 'missing_post_dates'],
            'records': df_by_tile.to_dict('records'),
        }
    section = report.build_section(
        'coverage',
        'Coverage',
        COVERAGE_DESCRIPTION,
        df_missing,
        len(df_cov),
        summary=summary,
    )
    csvs = {
        'coverage_missing': df_missing,
        'coverage_missing_by_tile': df_by_tile,
        'coverage': df_cov,
        'coverage_orphans': df_orphans,
    }
    return [section], csvs


@cli.command()
@data_dir_option
@out_dir_option
def check_duplicates(data_dir: Path, out_dir: Path) -> None:
    """Check for duplicated DIST-S1 products (check 2)."""
    df_dist = _load_dist(data_dir)
    sections, csvs = _duplicates_section(df_dist, _wkt_by_tile(_all_tile_geoms()))
    _write_outputs(out_dir, _stamp(), _all_tile_geoms(), sections, csvs, 'DIST-S1 duplicates', '')


@cli.command()
@data_dir_option
@out_dir_option
def check_ordering(data_dir: Path, out_dir: Path) -> None:
    """Check processing order matches acquisition order (check 3)."""
    df_dist = _load_dist(data_dir)
    df_dedup = checks.deduplicate_products(df_dist)
    sections, csvs = _ordering_section(df_dedup, _wkt_by_tile(_all_tile_geoms()))
    _write_outputs(out_dir, _stamp(), _all_tile_geoms(), sections, csvs, 'DIST-S1 processing order', '')


@cli.command()
@click.option(
    '--expect-none-at-start', is_flag=True, help='Require the first product per tile to have no prior (mission start).'
)
@data_dir_option
@out_dir_option
def check_confirmation(expect_none_at_start: bool, data_dir: Path, out_dir: Path) -> None:
    """Check the confirmation chain via prior product tags (check 4)."""
    df_dist = _load_dist(data_dir)
    df_dedup = checks.deduplicate_products(df_dist)
    sections, csvs = _confirmation_section(
        df_dedup, _load_tags(data_dir), expect_none_at_start, _wkt_by_tile(_all_tile_geoms())
    )
    _write_outputs(out_dir, _stamp(), _all_tile_geoms(), sections, csvs, 'DIST-S1 confirmation chain', '')


@cli.command()
@click.option('--mode', type=click.Choice(['offline', 'online']), default='offline', show_default=True)
@click.option('--sample', type=int, default=0, show_default=True, help='Randomly sample N products (0 = all).')
@click.option('--seed', type=int, default=42, show_default=True)
@click.option('--workers', type=int, default=8, show_default=True)
@data_dir_option
@out_dir_option
def check_inputs(mode: str, sample: int, seed: int, workers: int, data_dir: Path, out_dir: Path) -> None:
    """Check product inputs against re-enumerated expected inputs (check 5)."""
    df_dist = _load_dist(data_dir)
    df_tags = _load_tags(data_dir)
    df_tags = df_tags[df_tags.opera_id.isin(checks.deduplicate_products(df_dist).opera_id)]
    if sample:
        df_tags = df_tags.sample(n=min(sample, len(df_tags)), random_state=seed)
    df_frame = _build_rtc_frame(_load_rtc(data_dir), data_dir) if mode == 'offline' else None
    sections, csvs = _inputs_section(
        df_tags,
        df_frame,
        mode,
        workers,
        _wkt_by_tile(_all_tile_geoms()),
        checkpoint_path=data_dir / INPUTS_CHECKPOINT_PARQUET,
    )
    _write_outputs(out_dir, _stamp(), _all_tile_geoms(), sections, csvs, 'DIST-S1 product inputs', '')


@cli.command()
@click.option('--start', required=True)
@click.option('--stop', required=True)
@data_dir_option
@out_dir_option
def check_coverage(start: str, stop: str, data_dir: Path, out_dir: Path) -> None:
    """Check all triggerable DIST-S1 products exist given available RTC-S1 inputs (check 6)."""
    df_dist = _load_dist(data_dir)
    df_dedup = checks.deduplicate_products(df_dist)
    df_frame = _build_rtc_frame(_load_rtc(data_dir), data_dir)
    sections, csvs = _coverage_section(df_dedup, df_frame, start, stop, _wkt_by_tile(_all_tile_geoms()))
    _write_outputs(out_dir, _stamp(), _all_tile_geoms(), sections, csvs, 'DIST-S1 coverage', '')


@cli.command()
@click.option('--start', required=True, help='Start of acquisition period, e.g. 2026-01-01.')
@click.option('--stop', required=True, help='End of acquisition period, e.g. 2026-03-01.')
@click.option('--bbox', type=float, nargs=4, default=None)
@click.option('--workers', type=int, default=16, show_default=True)
@click.option('--cmr-workers', type=int, default=8, show_default=True)
@click.option(
    '--inputs-workers',
    type=int,
    default=8,
    show_default=True,
    help='Processes for the offline inputs check (the long stage).',
)
@click.option(
    '--sample-inputs', type=int, default=0, show_default=True, help='Sample N products for the inputs check (0 = all).'
)
@click.option(
    '--sample-tiles',
    type=int,
    default=0,
    show_default=True,
    help='Restrict every check to N random MGRS tiles (full time series per tile) for fast iteration (0 = all).',
)
@click.option('--seed', type=int, default=42, show_default=True)
@click.option('--expect-none-at-start', is_flag=True)
@click.option('--refresh', is_flag=True, help='Re-download metadata tables even if present.')
@click.option(
    '--processing-start',
    default=None,
    help='Only check DIST-S1 products processed at or after this time, e.g. 2026-09-24 (isolates one campaign).',
)
@click.option('--processing-stop', default=None, help='Only check DIST-S1 products processed before this time.')
@tag_url_template_option
@venue_option
@data_dir_option
@out_dir_option
def run_all(
    start: str,
    stop: str,
    bbox: tuple,
    workers: int,
    cmr_workers: int,
    inputs_workers: int,
    sample_inputs: int,
    sample_tiles: int,
    seed: int,
    expect_none_at_start: bool,
    refresh: bool,
    processing_start: str | None,
    processing_stop: str | None,
    tag_url_template: str | None,
    venue: str,
    data_dir: Path,
    out_dir: Path,
) -> None:
    """Run checks 1-6 end to end: download metadata, fetch tags, run every check, write CSVs and an HTML report."""
    from dist_s1_prod_check import cmr

    stamp = _stamp()
    _check_venue(data_dir, venue)
    bbox = tuple(bbox) if bbox else None

    need_dist = refresh or not (data_dir / DIST_S1_PARQUET).exists()
    need_rtc = refresh or not (data_dir / RTC_S1_PARQUET).exists()
    if need_dist or need_rtc:
        (data_dir / INPUTS_CHECKPOINT_PARQUET).unlink(missing_ok=True)
        dist_bbox, rtc_bbox = _resolve_bboxes(bbox, data_dir)
        chunks_dir = _rtc_chunks_dir(data_dir, rtc_bbox)
        if refresh:
            shutil.rmtree(chunks_dir, ignore_errors=True)
        if need_dist:
            df_dist = cmr.get_dist_s1_table(start, stop, bbox=dist_bbox, max_workers=cmr_workers, venue=venue)
            df_dist.to_parquet(data_dir / DIST_S1_PARQUET, compression='zstd')
        if need_rtc:
            df_rtc = cmr.get_rtc_s1_table_with_lookback(
                start, stop, bbox=rtc_bbox, max_workers=cmr_workers, cache_dir=chunks_dir
            )
            _write_rtc(df_rtc, data_dir, chunks_dir)
            del df_rtc
    df_dist = _load_dist(data_dir)
    if processing_start or processing_stop:
        df_dist = _filter_processing_time(df_dist, processing_start, processing_stop)
    if sample_tiles:
        all_tiles = pd.Series(sorted(df_dist.mgrs_tile_id.unique()))
        keep = set(all_tiles.sample(n=min(sample_tiles, len(all_tiles)), random_state=seed))
        df_dist = df_dist[df_dist.mgrs_tile_id.isin(keep)].reset_index(drop=True)
        click.echo(f'Sampled {len(keep)} of {len(all_tiles)} tiles')
    click.echo(f'DIST-S1 products: {len(df_dist)}')
    df_rtc = _load_rtc(data_dir)
    click.echo(f'RTC-S1 granules: {len(df_rtc)}')
    from dist_s1_prod_check.cmr import build_rtc_input_frame

    frame_tiles = sorted(keep) if sample_tiles else _load_target_tiles(data_dir)
    df_frame = build_rtc_input_frame(df_rtc, mgrs_tile_ids=frame_tiles)

    df_dedup = checks.deduplicate_products(df_dist)
    df_tags_products = (
        df_dedup if not sample_inputs else df_dedup.sample(n=min(sample_inputs, len(df_dedup)), random_state=42)
    )
    df_tags = tags.fetch_tags_table(
        df_tags_products, data_dir / TAGS_PARQUET, max_workers=workers, url_template=tag_url_template
    )
    df_tags = df_tags[df_tags.opera_id.isin(df_tags_products.opera_id)]

    wkt_map = _wkt_by_tile(_all_tile_geoms())
    subtitle = (
        f'{start} to {stop}'
        + (f' over {bbox}' if bbox else ' (global)')
        + (f' - sample of {sample_tiles} tiles' if sample_tiles else '')
        + (f' - {venue} venue' if venue != 'PROD' else '')
        + (
            f' - processed {processing_start or "start"} to {processing_stop or "now"}'
            if processing_start or processing_stop
            else ''
        )
        + f' - CMR metadata pulled {_pulled_at(data_dir)}'
    )
    title = f'DIST-S1 Production Check {pd.Timestamp(stamp):%Y-%m-%d}'
    sections = []
    try:
        for build in (
            lambda: _duplicates_section(df_dist, wkt_map),
            lambda: _ordering_section(df_dedup, wkt_map),
            lambda: _confirmation_section(df_dedup, df_tags, expect_none_at_start, wkt_map),
            lambda: _inputs_section(
                df_tags,
                df_frame,
                'offline',
                inputs_workers,
                wkt_map,
                checkpoint_path=data_dir / INPUTS_CHECKPOINT_PARQUET,
            ),
            lambda: _coverage_section(df_dedup, df_frame, start, stop, wkt_map),
        ):
            built_sections, section_csvs = build()
            for section in built_sections:
                click.echo(f'{section["title"]}: {section["n_failures"]} flagged of {section["n_checked"]} checked')
            sections.extend(built_sections)
            _write_csvs(out_dir, stamp, section_csvs)
    finally:
        if sections:
            _write_report(out_dir, stamp, _all_tile_geoms(), sections, title, subtitle)
