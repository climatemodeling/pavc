import pandas as pd
import requests
import warnings
from urllib3.exceptions import InsecureRequestWarning

# ── suppress the InsecureRequestWarning when using verify=False ──
warnings.simplefilter('ignore', InsecureRequestWarning)
session = requests.Session()

# 1) load & dedupe your NEON table
data_path = '../data/plot_data/neon/input_data/neon_cover_BARR_TOOL_1m2Data_2016-2023.csv'
df = pd.read_csv(data_path, usecols=['siteID', 'endDate']).drop_duplicates()

# 2) pull ALL camera metadata (paginated) and extract Sitename strings
all_cameras = []
url = "https://phenocam.nau.edu/api/cameras/?limit=1000"
while url:
    r = session.get(url, verify=False, timeout=10)
    r.raise_for_status()
    js = r.json()
    all_cameras += [cam['Sitename'] for cam in js['results']]
    url = js.get('next')

# 3) build a siteID → [camera names] lookup
site_to_cams = {}
for site in df['siteID'].unique():
    cams = []
    for name in all_cameras:
        parts = name.split('.')
        # only consider names like NEON.<zone>.<siteID>...
        if len(parts) >= 3 and parts[2] == site:
            cams.append(name)
    site_to_cams[site] = cams

# 4) pre-fetch each camera’s full midday-image listing exactly once
cam_to_midday = {}
for cams in site_to_cams.values():
    for cam in cams:
        if cam in cam_to_midday:
            continue
        # first call to get the total count
        init = session.get(
            f"https://phenocam.nau.edu/api/middayimages/?site={cam}&limit=1",
            verify=False, timeout=10
        )
        init.raise_for_status()
        total = init.json()['count']
        # fetch all entries in one go
        full = session.get(
            f"https://phenocam.nau.edu/api/middayimages/?site={cam}&limit={total}",
            verify=False, timeout=10
        )
        full.raise_for_status()
        cam_to_midday[cam] = full.json()['results']

# 5) define the row-wise function to filter by date and build <img> tags
def get_midday_images_html(row) -> str:
    cams = site_to_cams.get(row['siteID'], [])
    if not cams:
        return ''
    target_date = pd.to_datetime(row['endDate']).strftime('%Y-%m-%d')
    urls = []
    for cam in cams:
        for entry in cam_to_midday.get(cam, []):
            if entry['imgdate'] == target_date:
                # prefix the path with the domain
                urls.append("https://phenocam.nau.edu" + entry['imgpath'])
    if not urls:
        return ''
    return ', '.join(u for u in urls)

# 6) apply row-wise — now each row is just an in-memory filter
df['image_html'] = df.apply(get_midday_images_html, axis=1)
df.to_csv('/mnt/poseidon/remotesensing/arctic/alaska_pft_fcover_harmonization/data/plot_data/neon/temp_data/neon_imgs.csv')