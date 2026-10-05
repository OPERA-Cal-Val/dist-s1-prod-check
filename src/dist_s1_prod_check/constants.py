DIST_S1_CONCEPT_ID = 'C4090131664-ASF'
RTC_S1_CONCEPT_ID = 'C2777436413-ASF'
CMR_GRANULES_URL = 'https://cmr.earthdata.nasa.gov/search/granules.umm_json'
CMR_PAGE_SIZE = 2000

# DIST-S1 products can be checked in ASF's PROD or UAT venue. RTC-S1 inputs are always read from PROD CMR,
# since the UAT test venue consumes the operational RTC-S1 record.
VENUES = ('PROD', 'UAT')
DIST_S1_CONCEPT_IDS = {'PROD': DIST_S1_CONCEPT_ID, 'UAT': 'C1275699127-ASF'}
CMR_GRANULES_URLS = {'PROD': CMR_GRANULES_URL, 'UAT': 'https://cmr.uat.earthdata.nasa.gov/search/granules.umm_json'}
VENUE_FILE = 'venue.txt'

LAYER_URL_MAP = {
    'alert_url': 'GEN-DIST-STATUS',
    'metric_url': 'GEN-METRIC',
    'alert_acq_url': 'GEN-DIST-STATUS-ACQ',
}

DUAL_POLARIZATIONS = ('VV+VH', 'HH+HV')

CUMULUS_RTC_BASE = 'https://cumulus.asf.earthdatacloud.nasa.gov/OPERA/OPERA_L2_RTC-S1/'
VERTEX_SEARCH_URL = 'https://search.asf.alaska.edu/#/'

DELTA_LOOKBACK_DAYS = (365, 730, 1095)
MAX_PRE_IMGS_PER_BURST = (4, 3, 3)
DELTA_WINDOW_DAYS = 60
POST_DATE_BUFFER_SECONDS = 300
PASS_TOLERANCE_SECONDS = 300

DEFAULT_TAG_WORKERS = 16
DEFAULT_CMR_WORKERS = 8

DIST_S1_PARQUET = 'dist_s1_products.parquet'
RTC_S1_PARQUET = 'rtc_s1_products.parquet'
TAGS_PARQUET = 'dist_s1_tags.parquet'
INPUTS_CHECKPOINT_PARQUET = 'inputs_check_results.parquet'
TARGET_TILES_FILE = 'target_mgrs_tiles.txt'
