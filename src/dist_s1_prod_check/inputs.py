from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout
from pathlib import Path

import geopandas as gpd
import pandas as pd
from dist_s1_enumerator import enumerate_dist_s1_products, enumerate_one_dist_s1_product
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from dist_s1_prod_check.constants import (
    DELTA_LOOKBACK_DAYS,
    DELTA_WINDOW_DAYS,
    MAX_PRE_IMGS_PER_BURST,
    POST_DATE_BUFFER_SECONDS,
)
from dist_s1_prod_check.ids import (
    get_acq_timestamp,
    get_burst_id,
    get_opera_id_trunc,
    get_pass_start_times,
    get_track_number,
)


N_S1_TRACKS = 175


def parse_actual_inputs(pre_rtc_opera_ids: str, post_rtc_opera_ids: str) -> pd.DataFrame:
    pre_ids = [p for p in pre_rtc_opera_ids.split(',') if p]
    post_ids = [p for p in post_rtc_opera_ids.split(',') if p]
    df = pd.DataFrame(
        {
            'opera_id': pre_ids + post_ids,
            'input_category': ['pre'] * len(pre_ids) + ['post'] * len(post_ids),
        }
    )
    df['opera_id_trunc'] = df.opera_id.map(get_opera_id_trunc)
    df['jpl_burst_id'] = df.opera_id.map(get_burst_id)
    df['track_number'] = df.jpl_burst_id.map(get_track_number)
    df['acq_dt'] = df.opera_id.map(get_acq_timestamp)
    return df


def build_expected_products(df_rtc_frame: gpd.GeoDataFrame, mgrs_tile_ids: list[str] | None = None) -> pd.DataFrame:
    """Enumerate all expected DIST-S1 products (with pre/post inputs) offline from an RTC-S1 metadata table.

    Enumerates tile by tile via groupby so a global-scale frame is scanned once, not once per tile.
    """
    keep = None if mgrs_tile_ids is None else set(mgrs_tile_ids)
    frames = []
    product_id_offset = 0
    groups = df_rtc_frame.groupby('mgrs_tile_id', sort=True)
    with Path('/dev/null').open('w') as devnull, redirect_stdout(devnull):
        for tile, df_tile in tqdm(groups, total=groups.ngroups, desc='Enumerating expected products'):
            if keep is not None and tile not in keep:
                continue
            df_exp = enumerate_dist_s1_products(
                df_tile.reset_index(drop=True), mgrs_tile_ids=[tile], tqdm_enabled=False
            )
            if df_exp.empty:
                continue
            df_exp['product_id'] = df_exp['product_id'] + product_id_offset
            product_id_offset = int(df_exp['product_id'].max()) + 1
            frames.append(pd.DataFrame(df_exp.drop(columns='geometry')))
    df_expected = pd.concat(frames, ignore_index=True)
    df_expected['opera_id_trunc'] = df_expected.opera_id.map(get_opera_id_trunc)
    return df_expected


def _tracks_adjacent(track_numbers: list[int]) -> bool:
    if len(track_numbers) <= 1:
        return True
    if len(track_numbers) > 2:
        return False
    diff = abs(track_numbers[0] - track_numbers[1])
    return diff == 1 or diff == N_S1_TRACKS - 1


def _polarization_issues(df_actual: pd.DataFrame, pol_by_opera_id: dict[str, str]) -> list[dict]:
    """Flag baselines whose polarization differs from their burst's post-image.

    A burst's baseline must be built from acquisitions of the same dual polarization as its post-image
    (VV+VH with VV+VH, HH+HV with HH+HV). Inputs absent from `pol_by_opera_id` (outside the RTC table)
    are skipped; the RTC ID comparisons already report those.
    """
    pols = df_actual.opera_id_trunc.map(pol_by_opera_id)
    df = df_actual.assign(polarizations=pols)[pols.notna()]
    if df.empty:
        return []

    df_post = df[df.input_category == 'post']
    post_pol_by_burst = dict(zip(df_post.jpl_burst_id, df_post.polarizations))
    df_pre = df[df.input_category == 'pre']
    mismatched = [
        (r.opera_id_trunc, r.polarizations, post_pol_by_burst[r.jpl_burst_id])
        for r in df_pre.itertuples()
        if r.jpl_burst_id in post_pol_by_burst and r.polarizations != post_pol_by_burst[r.jpl_burst_id]
    ]
    if not mismatched:
        return []
    detail = ';'.join(f'{opera_id}={pol} vs post-image {post_pol}' for opera_id, pol, post_pol in mismatched)
    return [{'issue_type': 'Pre RTC baseline polarization mismatch', 'issue_value': detail}]


def compare_inputs(
    df_actual: pd.DataFrame,
    df_expected_product: pd.DataFrame,
    pol_by_opera_id: dict[str, str] | None = None,
) -> list[dict]:
    """Compare a product's recorded RTC-S1 inputs with the expected ones, matching inputs by burst pass.

    Any copy of a burst pass satisfies an expected input, since duplicated RTC-S1 granules of one
    acquisition can differ by a second in their names; using more than one copy is its own issue.
    """
    issues = []
    if pol_by_opera_id:
        issues.extend(_polarization_issues(df_actual, pol_by_opera_id))

    track_numbers = sorted(df_actual.track_number.unique().tolist())
    if not _tracks_adjacent(track_numbers):
        issues.append({'issue_type': 'Too many track numbers present', 'issue_value': str(track_numbers)})

    df_post = df_actual[df_actual.input_category == 'post']
    df_pre = df_actual[df_actual.input_category == 'pre']
    post_span = df_post.acq_dt.max() - df_post.acq_dt.min()
    if post_span.days > 1:
        issues.append({'issue_type': 'Post-dates span too long', 'issue_value': str(post_span)})

    exp_post = df_expected_product[df_expected_product.input_category == 'post']
    exp_pre = df_expected_product[df_expected_product.input_category == 'pre']

    bursts_expected = sorted(df_expected_product.jpl_burst_id.unique())
    bursts_actual = sorted(df_actual.jpl_burst_id.unique())
    if bursts_expected != bursts_actual:
        issues.append(
            {
                'issue_type': 'Burst ID mismatch',
                'issue_value': f'expected={bursts_expected}; found={bursts_actual}',
            }
        )

    cols = ['opera_id_trunc', 'jpl_burst_id', 'acq_dt']
    df_ids = pd.concat([df_actual[cols], df_expected_product[cols]]).drop_duplicates('opera_id_trunc')
    df_ids['pass_start'] = get_pass_start_times(df_ids, 'jpl_burst_id', 'acq_dt')
    pass_by_id = dict(zip(df_ids.opera_id_trunc, zip(df_ids.jpl_burst_id, df_ids.pass_start)))

    actual_passes = df_actual.opera_id_trunc.map(pass_by_id)
    df_dup = df_actual[actual_passes.duplicated(keep=False)]
    if not df_dup.empty:
        issues.append(
            {'issue_type': 'Duplicate RTC inputs for one burst pass', 'issue_value': ';'.join(df_dup.opera_id_trunc)}
        )

    comparisons = [
        ('Pre RTC IDs expected but not found', exp_pre, df_pre),
        ('Pre RTC IDs found but not expected', df_pre, exp_pre),
        ('Post RTC IDs expected but not found', exp_post, df_post),
        ('Post RTC IDs found but not expected', df_post, exp_post),
    ]
    for issue_type, df_left, df_right in comparisons:
        right_passes = set(df_right.opera_id_trunc.map(pass_by_id))
        diff = sorted({opera_id for opera_id in df_left.opera_id_trunc if pass_by_id[opera_id] not in right_passes})
        if diff:
            issues.append({'issue_type': issue_type, 'issue_value': ';'.join(diff)})

    return issues


def _expected_product_index(df_expected: pd.DataFrame) -> dict[str, pd.DataFrame]:
    posts = df_expected[df_expected.input_category == 'post']
    idx = (
        posts.groupby(['mgrs_tile_id', 'product_id'])
        .agg(post_min_dt=('acq_dt', 'min'), track_numbers=('track_number', 'unique'))
        .reset_index()
    )
    return {tile: group for tile, group in idx.groupby('mgrs_tile_id')}


def _match_expected_product_id(
    index_by_tile: dict[str, pd.DataFrame], mgrs_tile_id: str, track_numbers: list[int], post_min_dt: pd.Timestamp
) -> int | None:
    candidates = index_by_tile.get(mgrs_tile_id)
    if candidates is None:
        return None
    time_ok = (candidates.post_min_dt - post_min_dt).abs().dt.total_seconds() < POST_DATE_BUFFER_SECONDS
    track_ok = candidates.track_numbers.map(lambda tracks: bool(set(tracks) & set(track_numbers)))
    matches = candidates[time_ok & track_ok]
    if len(matches) != 1:
        return None
    return int(matches.product_id.iloc[0])


def _check_products_against_expected(
    df_tags_tile: pd.DataFrame, df_expected: pd.DataFrame, pol_by_opera_id: dict[str, str] | None = None
) -> list[dict]:
    index_by_tile = _expected_product_index(df_expected)
    expected_by_product = dict(tuple(df_expected.groupby('product_id')))

    records = []
    for row in df_tags_tile.itertuples():
        base = {'opera_id': row.opera_id, 'mgrs_tile_id': row.mgrs_tile_id}
        df_actual = parse_actual_inputs(row.pre_rtc_opera_ids, row.post_rtc_opera_ids)
        track_numbers = sorted(df_actual.track_number.unique().tolist())
        post_min_dt = df_actual[df_actual.input_category == 'post'].acq_dt.min()
        product_id = _match_expected_product_id(index_by_tile, row.mgrs_tile_id, track_numbers, post_min_dt)
        if product_id is None:
            records.append(
                {
                    **base,
                    'inputs_correct': False,
                    'issue_type': 'No expected product found',
                    'issue_value': f'tile={row.mgrs_tile_id}; tracks={track_numbers}; post={post_min_dt}',
                }
            )
            continue
        issues = compare_inputs(df_actual, expected_by_product[product_id], pol_by_opera_id=pol_by_opera_id)
        if not issues:
            records.append({**base, 'inputs_correct': True, 'issue_type': '', 'issue_value': ''})
        records.extend({**base, 'inputs_correct': False, **issue} for issue in issues)
    return records


def _check_one_tile(tile: str, df_tags_tile: pd.DataFrame, df_tile_frame: gpd.GeoDataFrame) -> list[dict]:
    try:
        with Path('/dev/null').open('w') as devnull, redirect_stdout(devnull):
            df_expected = enumerate_dist_s1_products(
                df_tile_frame.reset_index(drop=True), mgrs_tile_ids=[tile], tqdm_enabled=False
            )
        df_expected = pd.DataFrame(df_expected.drop(columns='geometry'))
        df_expected['opera_id_trunc'] = df_expected.opera_id.map(get_opera_id_trunc)
        pol_by_opera_id = dict(zip(df_tile_frame.opera_id.map(get_opera_id_trunc), df_tile_frame.polarizations))
        return _check_products_against_expected(df_tags_tile, df_expected, pol_by_opera_id=pol_by_opera_id)
    except Exception as e:
        return [
            {
                'opera_id': row.opera_id,
                'mgrs_tile_id': tile,
                'inputs_correct': None,
                'issue_type': 'Enumeration error',
                'issue_value': f'{type(e).__name__}: {e}',
            }
            for row in df_tags_tile.itertuples()
        ]


def check_inputs_offline(
    df_tags: pd.DataFrame,
    df_rtc_frame: gpd.GeoDataFrame,
    checkpoint_path: str | Path | None = None,
    workers: int = 1,
    checkpoint_every: int = 100,
) -> pd.DataFrame:
    """Compare each product's recorded RTC inputs against an offline re-enumeration from the RTC-S1 table.

    Enumerates expected products one MGRS tile at a time (in `workers` processes when > 1) and
    discards them after comparison, so peak memory is bounded by a single tile's inputs. With
    `checkpoint_path`, results are persisted every `checkpoint_every` tiles and prior results are
    reused, so interrupted runs resume instead of restarting.
    """
    records: list[dict] = []
    done_opera: set[str] = set()
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
    if checkpoint_path and checkpoint_path.exists():
        df_ck = pd.read_parquet(checkpoint_path)
        records = df_ck.to_dict('records')
        done_opera = set(df_ck.opera_id)

    requested_opera = set(df_tags.opera_id)
    df_todo = df_tags[~df_tags.opera_id.isin(done_opera)]
    df_err = df_todo[df_todo.error != '']
    records.extend(
        {
            'opera_id': row.opera_id,
            'mgrs_tile_id': row.mgrs_tile_id,
            'inputs_correct': None,
            'issue_type': 'Tag read error',
            'issue_value': row.error,
        }
        for row in df_err.itertuples()
    )
    df_ok = df_todo[df_todo.error == '']
    tags_by_tile = dict(tuple(df_ok.groupby('mgrs_tile_id')))
    frame_tiles = set(df_rtc_frame.mgrs_tile_id.unique())

    def _flush() -> None:
        if checkpoint_path is None:
            return
        tmp_path = checkpoint_path.with_suffix('.parquet.tmp')
        pd.DataFrame(records).to_parquet(tmp_path, compression='zstd')
        tmp_path.replace(checkpoint_path)

    frame_groups = df_rtc_frame.groupby('mgrs_tile_id')
    tiles_todo = []
    for tile, df_tags_tile in sorted(tags_by_tile.items()):
        if tile not in frame_tiles:
            records.extend(
                {
                    'opera_id': row.opera_id,
                    'mgrs_tile_id': tile,
                    'inputs_correct': False,
                    'issue_type': 'No expected product found',
                    'issue_value': f'tile={tile} absent from RTC-S1 table',
                }
                for row in df_tags_tile.itertuples()
            )
            continue
        tiles_todo.append(tile)

    with tqdm(total=len(tiles_todo), desc='Checking inputs by tile') as pbar:
        if workers <= 1:
            for k, tile in enumerate(tiles_todo):
                records.extend(_check_one_tile(tile, tags_by_tile[tile], frame_groups.get_group(tile)))
                pbar.update(1)
                if (k + 1) % checkpoint_every == 0:
                    _flush()
        else:
            window = workers * 4
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for i in range(0, len(tiles_todo), window):
                    futures = [
                        pool.submit(_check_one_tile, tile, tags_by_tile[tile], frame_groups.get_group(tile))
                        for tile in tiles_todo[i : i + window]
                    ]
                    for future in as_completed(futures):
                        records.extend(future.result())
                        pbar.update(1)
                    _flush()
    _flush()
    df = pd.DataFrame(records)
    return df[df.opera_id.isin(requested_opera)].reset_index(drop=True)


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=60), reraise=True)
def _enumerate_one(mgrs_tile_id: str, track_number: int, post_date: str) -> pd.DataFrame:
    df = enumerate_one_dist_s1_product(
        mgrs_tile_id=mgrs_tile_id,
        track_number=track_number,
        post_date=post_date,
        lookback_strategy='multi_window',
        delta_lookback_days=DELTA_LOOKBACK_DAYS,
        max_pre_imgs_per_burst=MAX_PRE_IMGS_PER_BURST,
        delta_window_days=DELTA_WINDOW_DAYS,
        tqdm_enabled=False,
    )
    df = pd.DataFrame(df.drop(columns='geometry'))
    df['opera_id_trunc'] = df.opera_id.map(get_opera_id_trunc)
    return df


def check_inputs_online(df_tags: pd.DataFrame, max_workers: int = 8) -> pd.DataFrame:
    """Compare recorded RTC inputs against a live per-product re-enumeration (slow; use for small batches)."""

    def _one(row: tuple) -> list[dict]:
        base = {'opera_id': row.opera_id, 'mgrs_tile_id': row.mgrs_tile_id}
        if row.error:
            return [{**base, 'inputs_correct': None, 'issue_type': 'Tag read error', 'issue_value': row.error}]
        try:
            df_actual = parse_actual_inputs(row.pre_rtc_opera_ids, row.post_rtc_opera_ids)
            df_post = df_actual[df_actual.input_category == 'post']
            track_number = int(df_actual.track_number.iloc[0])
            df_expected = _enumerate_one(row.mgrs_tile_id, track_number, str(df_post.acq_dt.min().date()))
            pol_by_opera_id = dict(zip(df_expected.opera_id_trunc, df_expected.polarizations))
            issues = compare_inputs(df_actual, df_expected, pol_by_opera_id=pol_by_opera_id)
        except Exception as e:
            return [
                {
                    **base,
                    'inputs_correct': None,
                    'issue_type': 'Enumeration error',
                    'issue_value': f'{type(e).__name__}: {e}',
                }
            ]
        if not issues:
            return [{**base, 'inputs_correct': True, 'issue_type': '', 'issue_value': ''}]
        return [{**base, 'inputs_correct': False, **issue} for issue in issues]

    rows = list(df_tags.itertuples())
    with (
        Path('/dev/null').open('w') as devnull,
        redirect_stdout(devnull),
        ThreadPoolExecutor(max_workers=max_workers) as pool,
    ):
        results = list(tqdm(pool.map(_one, rows), total=len(rows), desc='Checking inputs (online)'))
    return pd.DataFrame([record for result in results for record in result])
