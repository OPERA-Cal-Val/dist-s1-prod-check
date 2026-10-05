from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import pandas as pd
from shapely.geometry import box
from tqdm import tqdm

from dist_s1_prod_check.cmr import get_dist_s1_table
from dist_s1_prod_check.constants import CUMULUS_RTC_BASE, DEFAULT_TAG_WORKERS
from dist_s1_prod_check.ids import get_acq_timestamp_naive, get_burst_id
from dist_s1_prod_check.inputs import parse_actual_inputs
from dist_s1_prod_check.tags import get_tags


def get_products_over_aoi(
    bbox: tuple[float, float, float, float],
    start_time: str | pd.Timestamp | None = None,
    stop_time: str | pd.Timestamp | None = None,
    df_dist: gpd.GeoDataFrame | None = None,
    venue: str = 'PROD',
) -> gpd.GeoDataFrame:
    """Get DIST-S1 products over a lon/lat bbox, from a local table if provided, otherwise from CMR."""
    if df_dist is None:
        return get_dist_s1_table(start_time, stop_time, bbox=bbox, venue=venue)
    aoi = box(*bbox)
    df = df_dist[df_dist.intersects(aoi)]
    if start_time is not None:
        df = df[df.acq_time >= pd.Timestamp(start_time)]
    if stop_time is not None:
        df = df[df.acq_time < pd.Timestamp(stop_time)]
    return df.sort_values(['mgrs_tile_id', 'acq_time']).reset_index(drop=True)


def _rtc_urls(rtc_opera_id: str) -> dict[str, str]:
    return {
        'url_copol': f'{CUMULUS_RTC_BASE}{rtc_opera_id}/{rtc_opera_id}_VV.tif',
        'url_crosspol': f'{CUMULUS_RTC_BASE}{rtc_opera_id}/{rtc_opera_id}_VH.tif',
    }


def get_rtc_inputs_for_products(
    df_products: gpd.GeoDataFrame,
    max_workers: int = DEFAULT_TAG_WORKERS,
) -> pd.DataFrame:
    """Fetch the RTC-S1 inputs (pre and post, with cumulus urls) for each DIST-S1 product via its tags."""

    def _one(record: dict) -> pd.DataFrame:
        tags = get_tags(record['alert_url'])
        df = parse_actual_inputs(tags.get('pre_rtc_opera_ids', ''), tags.get('post_rtc_opera_ids', ''))
        df['dist_s1_opera_id'] = record['opera_id']
        return df

    records = df_products[['opera_id', 'alert_url']].to_dict('records')
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        frames = list(tqdm(pool.map(_one, records), total=len(records), desc='Fetching product inputs'))
    df = pd.concat(frames, ignore_index=True)
    df['acq_time'] = df.opera_id.map(get_acq_timestamp_naive)
    df['jpl_burst_id'] = df.opera_id.map(get_burst_id)
    urls = pd.DataFrame([_rtc_urls(oid) for oid in df.opera_id])
    return pd.concat([df, urls], axis=1)
