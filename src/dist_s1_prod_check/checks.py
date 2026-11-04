import pandas as pd

from dist_s1_prod_check.constants import PASS_TOLERANCE_SECONDS
from dist_s1_prod_check.ids import get_acq_timestamp_naive, get_mgrs_tile_id, get_pass_start_times


PASS_KEY = ['mgrs_tile_id', 'pass_start']


def _with_pass_start(df_dist: pd.DataFrame) -> pd.DataFrame:
    return df_dist.assign(pass_start=get_pass_start_times(df_dist, 'mgrs_tile_id', 'acq_time'))


def deduplicate_products(df_dist: pd.DataFrame) -> pd.DataFrame:
    """Keep one product per pass (MGRS tile, acquisition times within `PASS_TOLERANCE_SECONDS`): the last processed."""
    df = _with_pass_start(df_dist).sort_values([*PASS_KEY, 'processing_time'])
    df['n_versions'] = df.groupby(PASS_KEY)['opera_id'].transform('size')
    df = df[~df.duplicated(subset=PASS_KEY, keep='last')]
    return df.sort_values(['mgrs_tile_id', 'acq_time']).reset_index(drop=True)


def check_duplicates(df_dist: pd.DataFrame) -> pd.DataFrame:
    """Flag DIST-S1 granules for the same pass (reprocessed or duplicated products).

    A pass is one MGRS tile with acquisition times within `PASS_TOLERANCE_SECONDS`, so copies whose
    acquisition time differs by a second (from duplicated RTC-S1 inputs) are caught too. The
    last-processed granule of each pass is taken as the correct one; the earlier ones are superseded.
    """
    df = _with_pass_start(df_dist)
    dup_mask = df.duplicated(subset=PASS_KEY, keep=False)
    df_dup = df[dup_mask].sort_values([*PASS_KEY, 'processing_time']).copy()
    cols = [
        'opera_id',
        'mgrs_tile_id',
        'pass_start',
        'acq_time',
        'processing_time',
        'n_versions',
        'is_latest',
        'latest_opera_id',
        'reprocessing_lag_hours',
        'reason',
    ]
    if df_dup.empty:
        return pd.DataFrame(columns=cols)
    grouped = df_dup.groupby(PASS_KEY)
    df_dup['n_versions'] = grouped['opera_id'].transform('size')
    df_dup['latest_opera_id'] = grouped['opera_id'].transform('last')
    df_dup['is_latest'] = df_dup.opera_id == df_dup.latest_opera_id
    first_processing = grouped['processing_time'].transform('first')
    df_dup['reprocessing_lag_hours'] = (df_dup.processing_time - first_processing).dt.total_seconds() / 3600
    df_dup['reason'] = [
        f'{n} granules share the pass starting {start}; ' + ('latest, kept' if latest else f'superseded by {keep}')
        for n, start, latest, keep in zip(
            df_dup.n_versions, df_dup.pass_start, df_dup.is_latest, df_dup.latest_opera_id
        )
    ]
    extra = [c for c in ['alert_url'] if c in df_dup.columns]
    return df_dup[cols + extra].reset_index(drop=True)


def check_processing_order(df_dist_dedup: pd.DataFrame) -> pd.DataFrame:
    """Check each product was processed after the previous product in its MGRS tile.

    Runs on the last-processed version of each pass only (`deduplicate_products`); superseded
    versions are the subject of `check_duplicates`. A pass reprocessed after its successor was made
    flags that successor, which was built on a version that has since been replaced
    (`prev_n_versions` > 1). Each product is compared to its predecessor only, so one late product
    is one failure, not a cascade.
    """
    df = df_dist_dedup.sort_values(['mgrs_tile_id', 'acq_time']).copy()
    by_tile = df.groupby('mgrs_tile_id')
    df['prev_opera_id'] = by_tile['opera_id'].shift(1)
    df['prev_acq_time'] = by_tile['acq_time'].shift(1)
    df['prev_processing_time'] = by_tile['processing_time'].shift(1)
    df['prev_n_versions'] = by_tile['n_versions'].shift(1)
    df['processing_lag_hours'] = (df.processing_time - df.prev_processing_time).dt.total_seconds() / 3600
    df['evaluated'] = df.prev_opera_id.notna()
    df['out_of_order'] = df.evaluated & (df.processing_lag_hours < 0)
    df['reason'] = [
        f'processed {-lag:.1f} h before the previous product in the tile ({prev})' if bad else ''
        for bad, lag, prev in zip(df.out_of_order, df.processing_lag_hours, df.prev_opera_id)
    ]
    cols = [
        'opera_id',
        'mgrs_tile_id',
        'acq_time',
        'processing_time',
        'n_versions',
        'prev_opera_id',
        'prev_acq_time',
        'prev_processing_time',
        'prev_n_versions',
        'processing_lag_hours',
        'evaluated',
        'out_of_order',
        'reason',
    ]
    extra = [c for c in ['alert_url'] if c in df.columns]
    return df[cols + extra].reset_index(drop=True)


def summarize_by_tile(df_products: pd.DataFrame, fail_col: str) -> pd.DataFrame:
    def _one(group: pd.DataFrame) -> pd.Series:
        bad = group[group[fail_col]]
        return pd.Series(
            {
                'n_products': len(group),
                'n_failures': len(bad),
                'ok': bad.empty,
                'bad_opera_ids': ';'.join(bad.opera_id),
            }
        )

    return df_products.groupby('mgrs_tile_id').apply(_one, include_groups=False).reset_index()


def _same_pass(opera_id_a: str, opera_id_b: str) -> bool:
    if 'None' in (opera_id_a, opera_id_b):
        return opera_id_a == opera_id_b
    gap_seconds = abs(get_acq_timestamp_naive(opera_id_a) - get_acq_timestamp_naive(opera_id_b)).total_seconds()
    return get_mgrs_tile_id(opera_id_a) == get_mgrs_tile_id(opera_id_b) and gap_seconds <= PASS_TOLERANCE_SECONDS


def check_confirmation(
    df_dist_dedup: pd.DataFrame,
    df_tags: pd.DataFrame,
    expect_none_at_start: bool = False,
) -> pd.DataFrame:
    """Check each product's `prior_dist_s1_product` tag points to the previous pass in its tile's time series.

    Any version of that pass counts as the right prior; whether the product predates the kept
    version is reported by `check_processing_order`.
    """
    df = df_dist_dedup.merge(df_tags[['opera_id', 'prior_dist_s1_product', 'error']], on='opera_id', how='left')
    df = df.sort_values(['mgrs_tile_id', 'acq_time'])

    df['tag_missing'] = df.error.isna() | (df.error != '') | df.prior_dist_s1_product.isna()
    df['tile_incomplete'] = df.groupby('mgrs_tile_id')['tag_missing'].transform('any')
    df['prior_used'] = df.prior_dist_s1_product.fillna('').map(lambda p: p.split('/')[-1] or 'None')
    df['prior_expected'] = df.groupby('mgrs_tile_id')['opera_id'].shift(1).fillna('None')
    df['is_first_in_window'] = df.prior_expected == 'None'

    df['confirmed_ok'] = [_same_pass(used, expected) for used, expected in zip(df.prior_used, df.prior_expected)]
    if not expect_none_at_start:
        df.loc[df.is_first_in_window, 'confirmed_ok'] = True
    df.loc[df.tag_missing, 'confirmed_ok'] = False

    def _reason(row: pd.Series) -> str:
        if row.tag_missing:
            return 'could not read prior_dist_s1_product tag'
        if not row.confirmed_ok:
            return f'prior product used: {row.prior_used}; expected: {row.prior_expected}'
        return ''

    df['reason'] = df.apply(_reason, axis=1)

    cols = [
        'opera_id',
        'mgrs_tile_id',
        'acq_time',
        'processing_time',
        'prior_used',
        'prior_expected',
        'is_first_in_window',
        'tag_missing',
        'tile_incomplete',
        'confirmed_ok',
        'reason',
    ]
    extra = [c for c in ['alert_url'] if c in df.columns]
    return df[cols + extra].reset_index(drop=True)
