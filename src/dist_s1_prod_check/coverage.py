import numpy as np
import pandas as pd

from dist_s1_prod_check.constants import (
    DELTA_LOOKBACK_DAYS,
    DELTA_WINDOW_DAYS,
    DUAL_POLARIZATIONS,
    PASS_TOLERANCE_SECONDS,
)


def _to_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True)


def _burst_pol_key(df: pd.DataFrame) -> pd.Series:
    return df.jpl_burst_id + '|' + df.polarizations


def _dual_pol_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.polarizations.isin(DUAL_POLARIZATIONS)]


def expected_pass_groups(
    df_rtc_frame: pd.DataFrame,
    start_time: str | pd.Timestamp,
    stop_time: str | pd.Timestamp,
) -> pd.DataFrame:
    """Group RTC-S1 metadata into MGRS passes; each group is a DIST-S1 product that should exist.

    Rows are prefiltered to the window (with a 1-day margin so boundary passes group correctly)
    before aggregating, so lookback-era passes are never materialized as groups. Single-polarization
    granules are excluded: they are never DIST-S1 post-images.

    Each group carries `burst_pols`, the post-image `burst_id|polarization` keys, so a burst's
    baseline can only be drawn from acquisitions of that same polarization.
    """
    margin = pd.Timedelta(days=1)
    start_utc = pd.Timestamp(start_time).tz_localize('UTC')
    stop_utc = pd.Timestamp(stop_time).tz_localize('UTC')
    near_window = (df_rtc_frame.acq_dt >= start_utc - margin) & (df_rtc_frame.acq_dt < stop_utc + margin)
    df_post = _dual_pol_only(df_rtc_frame[near_window])
    groups = (
        df_post.assign(burst_pol=_burst_pol_key(df_post))
        .groupby(['mgrs_tile_id', 'acq_group_id_within_mgrs_tile', 'pass_id'])
        .agg(
            post_acq_dt=('acq_dt', 'min'),
            n_bursts=('jpl_burst_id', 'nunique'),
            track_token=('track_token', 'first'),
            burst_ids=('jpl_burst_id', 'unique'),
            burst_pols=('burst_pol', 'unique'),
        )
        .reset_index()
    )
    in_window = (groups.post_acq_dt >= start_utc) & (groups.post_acq_dt < stop_utc)
    return groups[in_window].reset_index(drop=True)


def _baseline_burst_counts(
    acq_by_burst_pol: dict[str, np.ndarray],
    burst_pols: list[str],
    post_acq_dt: pd.Timestamp,
    delta_lookback_days: tuple[int, ...],
    delta_window_days: int,
) -> int:
    """Count post-image bursts with at least one baseline acquisition of the *same* polarization.

    Each burst is resolved against its own polarization, so one MGRS tile freely mixes VV+VH and
    HH+HV bursts; only a burst's own baseline is constrained to match its own post-image.
    """
    bursts_with_baseline = set()
    for burst_pol in burst_pols:
        acqs = acq_by_burst_pol.get(burst_pol)
        if acqs is None:
            continue
        n_pre = 0
        for delta in delta_lookback_days:
            stop = post_acq_dt - pd.Timedelta(days=delta)
            start = stop - pd.Timedelta(days=delta_window_days)
            n_pre += np.searchsorted(acqs, stop.to_datetime64()) - np.searchsorted(acqs, start.to_datetime64())
        if n_pre > 0:
            bursts_with_baseline.add(burst_pol.split('|')[0])
    return len(bursts_with_baseline)


def check_coverage(
    df_expected_groups: pd.DataFrame,
    df_dist_dedup: pd.DataFrame,
    df_rtc_frame: pd.DataFrame | None = None,
    tolerance_seconds: int = PASS_TOLERANCE_SECONDS,
    delta_lookback_days: tuple[int, ...] = DELTA_LOOKBACK_DAYS,
    delta_window_days: int = DELTA_WINDOW_DAYS,
) -> pd.DataFrame:
    """Check every expected MGRS pass has a DIST-S1 product; missing passes are checked for baseline availability.

    A burst only counts as having a baseline when the lookback windows hold acquisitions of the same
    dual polarization as the post-image, matching how the enumerator builds a baseline. A product
    matches a pass only within `tolerance_seconds` (one pass), so it cannot stand in for the pass of
    another track over the same tile.
    """
    df_dist = df_dist_dedup.copy()
    df_dist['acq_time_utc'] = _to_utc(df_dist.acq_time)
    products_by_tile = {tile: group.reset_index(drop=True) for tile, group in df_dist.groupby('mgrs_tile_id')}

    acq_by_burst_pol: dict[str, np.ndarray] = {}
    if df_rtc_frame is not None:
        cols = ['jpl_burst_id', 'polarizations', 'opera_id', 'acq_dt']
        df_bursts = _dual_pol_only(df_rtc_frame[cols]).drop_duplicates(subset='opera_id')
        acq_by_burst_pol = {
            burst_pol: np.sort(group.acq_dt.values) for burst_pol, group in df_bursts.groupby(_burst_pol_key(df_bursts))
        }

    records = []
    for row in df_expected_groups.itertuples():
        products = products_by_tile.get(row.mgrs_tile_id)
        matched_opera_id, time_diff_s = None, None
        if products is not None:
            diffs = (products.acq_time_utc - row.post_acq_dt).abs().dt.total_seconds()
            best = diffs.idxmin()
            if diffs[best] <= tolerance_seconds:
                matched_opera_id, time_diff_s = products.opera_id[best], diffs[best]
        product_found = matched_opera_id is not None

        record = {
            'mgrs_tile_id': row.mgrs_tile_id,
            'track_token': row.track_token,
            'post_acq_dt': row.post_acq_dt,
            'pass_id': row.pass_id,
            'n_bursts': row.n_bursts,
            'product_found': product_found,
            'matched_opera_id': matched_opera_id or '',
            'time_diff_seconds': time_diff_s,
            'n_bursts_with_baseline': None,
            'missing_product_expected': False,
            'reason': '',
        }
        if not product_found:
            if acq_by_burst_pol:
                n_baseline = _baseline_burst_counts(
                    acq_by_burst_pol, list(row.burst_pols), row.post_acq_dt, delta_lookback_days, delta_window_days
                )
                record['n_bursts_with_baseline'] = n_baseline
                record['missing_product_expected'] = n_baseline > 0
                detail = (
                    f'{n_baseline}/{row.n_bursts} bursts have baseline imagery'
                    if n_baseline > 0
                    else 'no baseline imagery; product not expected'
                )
            else:
                record['missing_product_expected'] = True
                detail = 'baseline not checked (no lookback RTC metadata)'
            record['reason'] = f'no DIST-S1 product for RTC pass at {row.post_acq_dt} ({detail})'
        records.append(record)
    return pd.DataFrame(records)


def check_orphan_products(
    df_expected_groups: pd.DataFrame,
    df_dist_dedup: pd.DataFrame,
    tolerance_seconds: int = PASS_TOLERANCE_SECONDS,
) -> pd.DataFrame:
    """Find DIST-S1 products with no corresponding RTC-S1 pass in the expected groups."""
    df_dist = df_dist_dedup.copy()
    df_dist['acq_time_utc'] = _to_utc(df_dist.acq_time)
    groups_by_tile = {
        tile: group.post_acq_dt.sort_values().reset_index(drop=True)
        for tile, group in df_expected_groups.groupby('mgrs_tile_id')
    }

    records = []
    for row in df_dist.itertuples():
        passes = groups_by_tile.get(row.mgrs_tile_id)
        min_diff = None if passes is None else (passes - row.acq_time_utc).abs().dt.total_seconds().min()
        if min_diff is None or min_diff > tolerance_seconds:
            records.append(
                {
                    'opera_id': row.opera_id,
                    'mgrs_tile_id': row.mgrs_tile_id,
                    'acq_time': row.acq_time,
                    'nearest_pass_diff_seconds': min_diff,
                    'reason': 'DIST-S1 product exists but no matching RTC pass found in expected groups',
                }
            )
    return pd.DataFrame(records)
