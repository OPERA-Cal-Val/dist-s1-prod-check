from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import rasterio
from rasterio.env import Env
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

from dist_s1_prod_check.constants import DEFAULT_TAG_WORKERS


COOKIE_FILE = str(Path.home() / '.dist_s1_prod_check_cookies.txt')

TAG_KEYS = ['mgrs_tile_id', 'pre_rtc_opera_ids', 'post_rtc_opera_ids', 'prior_dist_s1_product']


@retry(stop=stop_after_attempt(10), wait=wait_random_exponential(multiplier=1, max=60), reraise=True)
def get_tags(url: str) -> dict[str, str]:
    gdal_env = {
        'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
        'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tif',
        'GDAL_HTTP_COOKIEFILE': COOKIE_FILE,
        'GDAL_HTTP_COOKIEJAR': COOKIE_FILE,
    }
    with Env(**gdal_env):
        with rasterio.open(url) as ds:
            return ds.tags()


def _preflight(url: str) -> None:
    """Read one product without retries so an access problem fails in seconds, not after hours of backoff."""
    try:
        get_tags.retry_with(stop=stop_after_attempt(1))(url)
    except Exception as e:
        raise RuntimeError(f'Cannot read product tags from {url}: {type(e).__name__}: {e}') from e


def _fetch_one(record: dict) -> dict:
    out = {'opera_id': record['opera_id'], 'error': ''}
    try:
        tags = get_tags(record['alert_url'])
        out.update({key: tags.get(key, '') for key in TAG_KEYS})
    except Exception as e:
        out.update(dict.fromkeys(TAG_KEYS, ''))
        out['error'] = f'{type(e).__name__}: {e}'
    return out


def fetch_tags_table(
    df_products: pd.DataFrame,
    out_path: str | Path,
    max_workers: int = DEFAULT_TAG_WORKERS,
    batch_size: int = 2_000,
    retry_errors: bool = True,
    url_template: str | None = None,
) -> pd.DataFrame:
    """Fetch DIST-S1 GeoTIFF metadata tags for each product, resumably cached in a parquet file.

    Reads `pre_rtc_opera_ids`, `post_rtc_opera_ids`, `mgrs_tile_id`, and `prior_dist_s1_product`
    from each product's GEN-DIST-STATUS layer. Products already present in `out_path` are skipped,
    so interrupted runs resume where they left off.

    `url_template` reads the layer from somewhere other than the CMR `alert_url`, e.g. the producing
    venue's S3 bucket. It is formatted with each product row, so `{opera_id}`, `{mgrs_tile_id}` and
    `{acq_time:%Y%m%d}` are available.
    """
    out_path = Path(out_path)
    if url_template:
        urls = [url_template.format(**row) for row in df_products.to_dict('records')]
        df_products = df_products.assign(alert_url=urls)
    df_existing = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame(columns=['opera_id', 'error'])
    if retry_errors:
        df_existing = df_existing[df_existing.error == '']

    todo = df_products[~df_products.opera_id.isin(df_existing.opera_id)][['opera_id', 'alert_url']]
    todo = todo[todo.alert_url.notna()]
    records = todo.to_dict('records')
    if not records:
        return df_existing
    _preflight(records[0]['alert_url'])

    batches = [records[i : i + batch_size] for i in range(0, len(records), batch_size)]
    with tqdm(total=len(records), desc='Fetching tags') as pbar:
        for batch in batches:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                results = []
                for result in pool.map(_fetch_one, batch):
                    results.append(result)
                    pbar.update(1)
            df_existing = pd.concat([df_existing, pd.DataFrame(results)], ignore_index=True)
            tmp_path = out_path.with_suffix('.parquet.tmp')
            df_existing.to_parquet(tmp_path, compression='zstd')
            tmp_path.replace(out_path)
    return df_existing
