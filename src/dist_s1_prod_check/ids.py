from pathlib import Path

import pandas as pd

from dist_s1_prod_check.constants import PASS_TOLERANCE_SECONDS


def get_pass_start_times(
    df: pd.DataFrame, group_col: str, time_col: str, tolerance_seconds: float = PASS_TOLERANCE_SECONDS
) -> pd.Series:
    """First acquisition time of each row's pass; rows of one `group_col` acquired close together share a pass.

    Rows at most `tolerance_seconds` apart are one pass, and (`group_col`, pass start) identifies it.
    Granule names alone cannot: one RTC-S1 acquisition processed twice can carry acquisition times a
    second apart, and the DIST-S1 products built from each copy inherit that difference.
    """
    df_sorted = df[[group_col, time_col]].reset_index(drop=True).sort_values([group_col, time_col])
    gap_seconds = df_sorted.groupby(group_col)[time_col].diff().dt.total_seconds()
    pass_start = df_sorted[time_col].where(~(gap_seconds <= tolerance_seconds)).ffill().sort_index()
    return pd.Series(pass_start.array, index=df.index)


def get_opera_id_trunc(opera_id: str) -> str:
    return '_'.join(Path(opera_id).name.split('_')[:5])


def get_burst_id(opera_rtc_id: str) -> str:
    return Path(opera_rtc_id).name.split('_')[3]


def get_track_number(jpl_burst_id: str) -> int:
    return int(jpl_burst_id.split('-')[0][1:])


def get_acq_timestamp(opera_id: str) -> pd.Timestamp:
    return pd.Timestamp(Path(opera_id).name.split('_')[4].rstrip('Z')).tz_localize('UTC')


def get_acq_timestamp_naive(opera_id: str) -> pd.Timestamp:
    return pd.Timestamp(Path(opera_id).name.split('_')[4].rstrip('Z'))


def get_processing_timestamp_naive(opera_id: str) -> pd.Timestamp:
    return pd.Timestamp(Path(opera_id).name.split('_')[5].rstrip('Z'))


def get_mgrs_tile_id(opera_dist_s1_id: str) -> str:
    return Path(opera_dist_s1_id).name.split('_')[3].lstrip('T')


def get_opera_id_from_url(url: str) -> str:
    return Path(url).parent.name
