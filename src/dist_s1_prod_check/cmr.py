from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from warnings import warn

import earthaccess
import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from dist_s1_enumerator.asf import append_pass_data, extract_pass_id, format_polarization
from dist_s1_enumerator.mgrs_burst_data import (
    get_burst_table,
    get_lut_by_mgrs_tile_ids,
    get_mgrs_burst_lut,
    get_mgrs_table,
    get_mgrs_tiles_overlapping_geometry,
)
from shapely.geometry import box
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

from dist_s1_prod_check.constants import (
    CMR_GRANULES_URL,
    CMR_GRANULES_URLS,
    CMR_PAGE_SIZE,
    DELTA_LOOKBACK_DAYS,
    DELTA_WINDOW_DAYS,
    DIST_S1_CONCEPT_IDS,
    DUAL_POLARIZATIONS,
    LAYER_URL_MAP,
    RTC_S1_CONCEPT_ID,
)
from dist_s1_prod_check.ids import get_pass_start_times, get_track_number


DateLike = str | datetime | pd.Timestamp


def get_edl_token(venue: str = 'PROD') -> str:
    auth = earthaccess.login(system=earthaccess.UAT if venue == 'UAT' else earthaccess.PROD)
    assert auth.authenticated, 'Earthdata login failed; check ~/.netrc or EARTHDATA_* env vars.'
    return auth.token['access_token']


def _fmt_dt(t: DateLike) -> str:
    ts = pd.Timestamp(t)
    ts = ts.tz_localize('UTC') if ts.tz is None else ts.tz_convert('UTC')
    return ts.strftime('%Y-%m-%dT%H:%M:%SZ')


@retry(stop=stop_after_attempt(10), wait=wait_random_exponential(multiplier=1, max=60), reraise=True)
def _get_page(params: dict, headers: dict, cmr_url: str = CMR_GRANULES_URL) -> tuple[list[dict], str | None]:
    resp = requests.get(cmr_url, params=params, headers=headers, timeout=120)
    resp.raise_for_status()
    return resp.json().get('items', []), resp.headers.get('CMR-Search-After')


def search_granules(
    collection_concept_id: str,
    start_time: DateLike,
    stop_time: DateLike,
    token: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    granule_name_pattern: str | None = None,
    parse_fn: Callable[[dict], dict] | None = None,
    cmr_url: str = CMR_GRANULES_URL,
) -> list[dict]:
    """Search CMR granules; with `parse_fn`, each page is parsed as it arrives so raw UMM JSON is never accumulated."""
    params: dict = {
        'collection_concept_id': collection_concept_id,
        'temporal': f'{_fmt_dt(start_time)},{_fmt_dt(stop_time)}',
        'page_size': CMR_PAGE_SIZE,
    }
    if bbox is not None:
        params['bounding_box'] = ','.join(map(str, bbox))
    if granule_name_pattern is not None:
        params['readable_granule_name'] = granule_name_pattern
        params['options[readable_granule_name][pattern]'] = 'true'

    headers = {'Authorization': f'Bearer {token}'} if token else {}
    items: list[dict] = []
    search_after = None
    while True:
        page_headers = headers | ({'CMR-Search-After': search_after} if search_after else {})
        page_items, search_after = _get_page(params, page_headers, cmr_url)
        items.extend(map(parse_fn, page_items) if parse_fn else page_items)
        if not page_items or search_after is None:
            return items


def _date_chunks(start_time: DateLike, stop_time: DateLike, chunk_days: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start, stop = pd.Timestamp(start_time), pd.Timestamp(stop_time)
    edges = pd.date_range(start, stop, freq=f'{chunk_days}D')
    if len(edges) == 0 or edges[-1] < stop:
        edges = edges.append(pd.DatetimeIndex([stop]))
    return list(zip(edges[:-1], edges[1:]))


def search_granules_chunked(
    collection_concept_id: str,
    start_time: DateLike,
    stop_time: DateLike,
    token: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    chunk_days: int = 3,
    max_workers: int = 8,
    desc: str = 'CMR chunks',
    parse_fn: Callable[[dict], dict] | None = None,
    cmr_url: str = CMR_GRANULES_URL,
) -> list[dict]:
    """Search CMR granules over a time range by splitting into temporal chunks queried concurrently."""
    chunks = _date_chunks(start_time, stop_time, chunk_days)

    def _one(chunk: tuple[pd.Timestamp, pd.Timestamp]) -> list[dict]:
        return search_granules(
            collection_concept_id, chunk[0], chunk[1], token=token, bbox=bbox, parse_fn=parse_fn, cmr_url=cmr_url
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        with tqdm(total=len(chunks), desc=desc, unit='chunk') as pbar:
            results = []
            for chunk_items in pool.map(_one, chunks):
                results.append(chunk_items)
                pbar.update(1)
                pbar.set_postfix(granules=sum(len(r) for r in results))
    return [item for chunk_items in results for item in chunk_items]


def _search_chunked_to_df(
    collection_concept_id: str,
    start_time: DateLike,
    stop_time: DateLike,
    parse_fn: Callable[[dict], dict],
    token: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    chunk_days: int = 3,
    max_workers: int = 8,
    desc: str = 'CMR chunks',
    cmr_url: str = CMR_GRANULES_URL,
) -> pd.DataFrame:
    """Chunked CMR search where each chunk is converted to a compact DataFrame as it completes.

    Peak memory is bounded by the final concatenated table plus one in-flight chunk per worker,
    never the whole corpus as python dicts.
    """
    chunks = _date_chunks(start_time, stop_time, chunk_days)

    def _one(chunk: tuple[pd.Timestamp, pd.Timestamp]) -> pd.DataFrame:
        records = search_granules(
            collection_concept_id, chunk[0], chunk[1], token=token, bbox=bbox, parse_fn=parse_fn, cmr_url=cmr_url
        )
        return pd.DataFrame(records)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        with tqdm(total=len(chunks), desc=desc, unit='chunk') as pbar:
            frames = []
            n_granules = 0
            for df_chunk in pool.map(_one, chunks):
                if not df_chunk.empty:
                    frames.append(df_chunk)
                    n_granules += len(df_chunk)
                pbar.update(1)
                pbar.set_postfix(granules=n_granules)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def resolve_target_tiles(
    bbox: tuple[float, float, float, float], margin_deg: float = 0.25
) -> tuple[list[str], tuple[float, float, float, float]]:
    """Resolve a lon/lat bbox to the DIST-S1 MGRS tiles it overlaps plus a widened bbox for RTC-S1 queries.

    The widened bbox covers the footprints of every burst feeding those tiles (with a margin for
    per-acquisition footprint jitter) so no product input is missed at the AOI edge. Does not handle
    antimeridian-crossing AOIs.
    """
    tiles = get_mgrs_tiles_overlapping_geometry(box(*bbox))
    tile_ids = sorted(tiles.mgrs_tile_id)
    burst_ids = sorted(get_lut_by_mgrs_tile_ids(tile_ids).jpl_burst_id.unique())
    lon_min, lat_min, lon_max, lat_max = get_burst_table(burst_ids).total_bounds
    widened = (
        max(lon_min - margin_deg, -180.0),
        max(lat_min - margin_deg, -90.0),
        min(lon_max + margin_deg, 180.0),
        min(lat_max + margin_deg, 90.0),
    )
    return tile_ids, widened


def parse_dist_s1_granule(item: dict) -> dict:
    umm = item['umm']
    opera_id = umm['GranuleUR']
    tokens = opera_id.split('_')
    urls = [
        u['URL'] for u in umm.get('RelatedUrls', []) if u['URL'].startswith('https://') and u['URL'].endswith('.tif')
    ]
    url_cols = {
        col: next((u for u in urls if u.endswith(f'{suffix}.tif')), None) for col, suffix in LAYER_URL_MAP.items()
    }
    return {
        'opera_id': opera_id,
        'mgrs_tile_id': tokens[3].lstrip('T'),
        'acq_time': pd.Timestamp(tokens[4].rstrip('Z')),
        'processing_time': pd.Timestamp(tokens[5].rstrip('Z')),
        'sat': tokens[6],
        **url_cols,
    }


def parse_rtc_s1_granule(item: dict) -> dict:
    umm = item['umm']
    opera_id = umm['GranuleUR']
    tokens = opera_id.split('_')
    jpl_burst_id = tokens[3]
    attrs = {a['Name']: a['Values'] for a in umm.get('AdditionalAttributes', [])}
    urls = [
        u['URL'] for u in umm.get('RelatedUrls', []) if u['URL'].startswith('https://') and u['URL'].endswith('.tif')
    ]
    return {
        'opera_id': opera_id,
        'jpl_burst_id': jpl_burst_id,
        'acq_dt': pd.Timestamp(tokens[4].rstrip('Z')).tz_localize('UTC'),
        'processing_dt': pd.Timestamp(tokens[5].rstrip('Z')).tz_localize('UTC'),
        'track_number': get_track_number(jpl_burst_id),
        'polarizations': format_polarization(attrs.get('POLARIZATION', [])),
        'url_copol': next((u for u in urls if u.endswith(('_VV.tif', '_HH.tif'))), ''),
        'url_crosspol': next((u for u in urls if u.endswith(('_VH.tif', '_HV.tif'))), ''),
    }


def is_latest_rtc_in_burst_pass(df_rtc: pd.DataFrame) -> np.ndarray:
    """Mask keeping exactly one RTC-S1 granule per burst and pass: the one processed last.

    Besides reprocessed granules sharing a name prefix, this folds granules of one acquisition whose
    acquisition times differ by a second (opera-sds-pcm#1550), which granule names alone cannot catch.
    Only three columns are touched, so a global table is never sorted or copied whole.
    """
    df = df_rtc[['jpl_burst_id', 'acq_dt', 'processing_dt']].reset_index(drop=True)
    df['pass_start'] = get_pass_start_times(df, 'jpl_burst_id', 'acq_dt')
    superseded = df.sort_values('processing_dt').duplicated(subset=['jpl_burst_id', 'pass_start'], keep='last')
    return ~superseded.sort_index().to_numpy()


def get_dist_s1_table(
    start_time: DateLike,
    stop_time: DateLike,
    bbox: tuple[float, float, float, float] | None = None,
    token: str | None = None,
    chunk_days: int = 7,
    max_workers: int = 8,
    venue: str = 'PROD',
) -> gpd.GeoDataFrame:
    """Download DIST-S1 granule metadata from the PROD or UAT CMR into a GeoDataFrame with MGRS tile footprints."""
    token = token or get_edl_token(venue)
    df = _search_chunked_to_df(
        DIST_S1_CONCEPT_IDS[venue],
        start_time,
        stop_time,
        parse_dist_s1_granule,
        token=token,
        bbox=bbox,
        chunk_days=chunk_days,
        max_workers=max_workers,
        desc=f'DIST-S1 CMR ({venue})',
        cmr_url=CMR_GRANULES_URLS[venue],
    )
    if df.empty:
        raise RuntimeError('No DIST-S1 granules returned; check the time range and Earthdata credentials.')
    df = df.drop_duplicates(subset='opera_id')
    tile_geoms = get_mgrs_table()[['mgrs_tile_id', 'geometry']]
    df = tile_geoms.merge(df, on='mgrs_tile_id', how='right')
    df = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
    return df.sort_values(['mgrs_tile_id', 'acq_time']).reset_index(drop=True)


def get_rtc_s1_table(
    start_time: DateLike,
    stop_time: DateLike,
    bbox: tuple[float, float, float, float] | None = None,
    chunk_days: int = 2,
    max_workers: int = 8,
) -> gpd.GeoDataFrame:
    """Download RTC-S1 granule metadata from CMR into a GeoDataFrame with burst footprints."""
    df = _search_chunked_to_df(
        RTC_S1_CONCEPT_ID,
        start_time,
        stop_time,
        parse_rtc_s1_granule,
        bbox=bbox,
        chunk_days=chunk_days,
        max_workers=max_workers,
        desc='RTC-S1 CMR',
    )
    if df.empty:
        raise RuntimeError('No RTC-S1 granules returned; check the time range.')
    df = df.drop_duplicates(subset='opera_id')
    df = df[is_latest_rtc_in_burst_pass(df)]
    burst_geoms = get_burst_table(sorted(df.jpl_burst_id.unique()))[['jpl_burst_id', 'geometry']]
    df = burst_geoms.merge(df, on='jpl_burst_id', how='right')
    df = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
    return df.sort_values(['jpl_burst_id', 'acq_dt']).reset_index(drop=True)


def lookback_date_ranges(
    start_time: DateLike,
    stop_time: DateLike,
    delta_lookback_days: tuple[int, ...] = DELTA_LOOKBACK_DAYS,
    delta_window_days: int = DELTA_WINDOW_DAYS,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Date ranges of RTC-S1 metadata needed to re-enumerate baselines for products in [start_time, stop_time]."""
    start, stop = pd.Timestamp(start_time), pd.Timestamp(stop_time)
    ranges = [
        (start - pd.Timedelta(days=delta + delta_window_days), stop - pd.Timedelta(days=delta))
        for delta in delta_lookback_days
    ]
    return sorted(ranges) + [(start, stop)]


def get_rtc_s1_table_with_lookback(
    start_time: DateLike,
    stop_time: DateLike,
    bbox: tuple[float, float, float, float] | None = None,
    delta_lookback_days: tuple[int, ...] = DELTA_LOOKBACK_DAYS,
    delta_window_days: int = DELTA_WINDOW_DAYS,
    chunk_days: int = 2,
    max_workers: int = 8,
) -> gpd.GeoDataFrame:
    tables = [
        get_rtc_s1_table(t0, t1, bbox=bbox, chunk_days=chunk_days, max_workers=max_workers)
        for t0, t1 in lookback_date_ranges(start_time, stop_time, delta_lookback_days, delta_window_days)
    ]
    df = pd.concat(tables).drop_duplicates(subset='opera_id')
    df = df[is_latest_rtc_in_burst_pass(df)]
    return (
        gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
        .sort_values(['jpl_burst_id', 'acq_dt'])
        .reset_index(drop=True)
    )


def build_rtc_input_frame(df_rtc: gpd.GeoDataFrame, mgrs_tile_ids: list[str] | None = None) -> gpd.GeoDataFrame:
    """Shape an RTC-S1 metadata table into the enumerator's rtc_s1_schema (LUT join adds one row per MGRS tile).

    Single-polarization granules are dropped here: DIST-S1 never uses them, as a post-image or in a baseline.
    Duplicates of a burst pass are dropped next, keeping the last processed, so every check sees exactly
    one RTC-S1 per burst pass (also for tables downloaded before that rule was applied at download).
    """
    dual_pol = df_rtc.polarizations.isin(DUAL_POLARIZATIONS)
    n_dropped = int((~dual_pol).sum())
    if n_dropped:
        warn(f'Dropping {n_dropped} RTC-S1 granules that are not dual polarization ({", ".join(DUAL_POLARIZATIONS)}).')
    valid = (dual_pol & df_rtc.geometry.notna()).to_numpy()
    keep = valid.copy()
    keep[valid] = is_latest_rtc_in_burst_pass(df_rtc.loc[valid, ['jpl_burst_id', 'acq_dt', 'processing_dt']])
    n_superseded = int(valid.sum() - keep.sum())
    if n_superseded:
        warn(f'Dropping {n_superseded} RTC-S1 granules duplicating a burst pass; kept the last processed.')
    df = df_rtc[keep]
    df['acq_dt'] = pd.to_datetime(df.acq_dt, utc=True)
    df['pass_id'] = df.acq_dt.map(extract_pass_id)
    df['url_copol'] = df.url_copol.fillna('')
    df['url_crosspol'] = df.url_crosspol.fillna('')
    if mgrs_tile_ids is None:
        lut = get_mgrs_burst_lut()
        mgrs_tile_ids = sorted(lut[lut.jpl_burst_id.isin(df.jpl_burst_id.unique())].mgrs_tile_id.unique())
    df = append_pass_data(df, mgrs_tile_ids)
    return gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
