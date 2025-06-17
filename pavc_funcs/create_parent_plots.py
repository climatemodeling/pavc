import pandas as pd
import numpy as np
from pyproj import Transformer
import warnings
from ast import literal_eval
 
# suppress warnings
warnings.filterwarnings('ignore')

# parameters
src = 'akveg'
dst = 'VEG'
vers = '03'
wdir = '../../data/training/Test_06/fcover'
dist_thres = 55          # meters
small_area_thr = 314     # m-2 threshold for small plots

# --- 0) Set export naming schema ---
schema = {
    'surveyYear':'year',
    'latitudeY':'latitude',
    'longitudeX':'longitude',
    'dataSource':'source',
    'dataSubsource':'subsource',
    'plotArea':'plot area m2',
    'plotShape':'plot shape',
    'surveyMethod':'field sampling method',
    'fcoverScale':'cover measurement',
    'baregroundCover':'bare ground top cover (%)',
    'bryophyteCover':'bryophyte total cover (%)',
    'deciduousShrubCover':'deciduous shrub total cover (%)',
    'deciduousTreeCover':'deciduous tree total cover (%)',
    'evergreenShrubCover':'evergreen shrub total cover (%)',
    'evergreenTreeCover':'evergreen tree total cover (%)',
    'forbCover':'forb total cover (%)',
    'graminoidCover':'graminoid total cover (%)',
    'lichenCover':'lichen total cover (%)',
    'litterCover':'litter total cover (%)',
    'nonvascularCover':'non-vascular total cover (%)',
    'otherCover':'other total cover (%)',
    'waterCover':'water top cover (%)'}


# --- 1) Read & export child_data ---
# read cover and plot info
override_dtype = {'plotVisit': str}
cover = pd.read_csv(f'../data/plot_data/{src}/output_data/{src}_standard_pft_fcover.csv', dtype=override_dtype)
info = pd.read_csv(f'../data/plot_data/{src}/output_data/{src}_plot_info.csv', dtype=override_dtype)
info = info[[
    'plotVisit','plotName', 'surveyYear',
    'latitudeY','longitudeX','dataSource','dataSubsource',
    'plotArea','plotShape','surveyMethod','fcoverScale', 
]]

# replace plotVisit names with the ones used in test 06, training data 01
merged_df = pd.merge(cover, info[['plotVisit', 'plotName']], on='plotVisit', how='left')
cover['plotVisit'] = merged_df['plotName']

# merge and filter by year ≥ 2010
orig_child = pd.read_csv(f'{wdir}/{dst}_fcover_child_01.csv')
child = cover.merge(info, left_on='plotVisit', right_on='plotName', how='left', suffixes=('','_x'))
child = child.drop(columns=[col for col in child.columns.to_list() if col.endswith('_x')])
child = child[child['surveyYear'] >= 2010].reset_index(drop=True)
if src == 'neon':
    child['plotVisit'] = child['plotVisit'].str.replace(
        r'(\d+)_(\d+)_(\d+)',
        r'\1.\3.\2',
        regex=True
    )
child = child[child['plotVisit'].isin(orig_child['Site Code'])].reset_index(drop=True)

# create non-vascular
child['nonvascularCover'] = child[['bryophyteCover', 'lichenCover']].sum(axis=1, skipna=True)

# export prep
child_out = child.copy()
child_out = child_out.rename(columns=schema)
child_out.index = child_out['plotVisit']
child_out.index.name = 'Site Code'
child_out = child_out.drop(columns=['plotVisit', 'plotName'])

# formatting and export
child_out = child_out.sort_index()
sorted_cols = [
    'bare ground top cover (%)', 'bryophyte total cover (%)', 
    'deciduous shrub total cover (%)', 'deciduous tree total cover (%)', 
    'evergreen shrub total cover (%)', 'evergreen tree total cover (%)', 
    'forb total cover (%)', 'graminoid total cover (%)', 
    'lichen total cover (%)', 'litter total cover (%)', 
    'non-vascular total cover (%)', 'water top cover (%)', 'other total cover (%)',
    'latitude', 'longitude', 
    'plot area m2', 'plot shape', 'source', 
    'year', 'field sampling method', 
    'cover measurement']
child_out = child_out[sorted_cols]
child_out.to_csv(f'{wdir}/{dst}_fcover_child_{vers}.csv')

# project to UTM once, and stash coords on every row
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32606", always_xy=True)
xy = np.array([transformer.transform(lon, lat)
               for lon, lat in zip(child.longitudeX, child.latitudeY)])
child['x'], child['y'] = xy[:,0], xy[:,1]

# --- 2) cluster small plots by distance ---
def group_by_distance(points, threshold):
    labels = np.full(len(points), -1, dtype=int)
    clusters = {}
    cid = 0
    for i, pt in enumerate(points):
        for k, members in clusters.items():
            if np.any(np.linalg.norm(np.vstack(members) - pt, axis=1) <= threshold):
                clusters[k].append(pt)
                labels[i] = k
                break
        else:
            clusters[cid] = [pt]
            labels[i] = cid
            cid += 1
    return labels

# --- recalc area & shape with list comprehensions instead of apply() ---
# helper for max-pairwise distance (unchanged)
def max_pairwise_dist(xs, ys):
    pts = np.column_stack((xs, ys))
    if pts.shape[0] < 2:
        return 0.0
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    return d.max()

# helper for computing each parent’s area
def compute_parent_area(xs, ys, orig_list):
    if len(xs) < 2:
        return orig_list[0]
    dmax = max_pairwise_dist(xs, ys)
    return np.pi * (dmax / 2) ** 2

# 1) Give every row a default group_id = its own plotVisit
child['group_id'] = child['plotVisit']

# 2) Identify the small ones, cluster *only* those, and override group_id there
small_mask = child['plotArea'] < small_area_thr
labels = np.array([], dtype=int)
if small_mask.any():
    coords = child.loc[small_mask, ['x','y']].to_numpy()
    labels = group_by_distance(coords, dist_thres)
    child.loc[small_mask, 'group_id'] = labels.astype(str)

# 3) Did *any* small cluster actually merge >1 point?
merged = pd.Series(labels).value_counts().gt(1).any() if small_mask.any() else False

# --- build parents DF ---
parents = child.set_index('plotVisit')
# aggregated to parent flag
if small_mask.any():
    group_counts = pd.Series(labels).value_counts()
else:
    group_counts = pd.Series(dtype=int)

# create aggregate column
parents['aggregated to parent'] = parents['group_id'].map(
    lambda g: 'yes' if group_counts.get(int(g) if str(g).isdigit() else g, 0) > 1 else 'no'
)

# create set of child site codes
parents['child site codes'] = parents['group_id'].map(
    lambda g: set(child.loc[child['group_id']==g, 'plotVisit'])
)

# 5) Decide whether to rebuild *all* plotVisit values or leave them alone
# rename visits if merged
if merged:
    # first assign new plotVisit index based on group
    new_visits = (
        parents['group_id'] + '_' +
        parents['surveyYear'].astype(str) + '_' +
        parents['dataSource']
    )
    parents.index = new_visits
    parents.index.name = 'plotVisit'

    # attempt to match canonical Site Code from orig_parent if available
    orig_parent = pd.read_csv(f'{wdir}/{dst}_fcover_parent_01.csv', dtype=object)
    raw_col = 'child site codes' if 'child site codes' in orig_parent.columns else (
              'child_site_codes' if 'child_site_codes' in orig_parent.columns else None)
    if raw_col:
        orig_parent['child site codes'] = orig_parent[raw_col].apply(literal_eval)
        def norm_set(lst):
            return frozenset(s.strip() for s in lst)
        mapping = { norm_set(c): sid for sid,c in zip(orig_parent['Site Code'], orig_parent['child site codes']) }
        old_index = parents.index.to_series()
        parent_keys = parents['child site codes'].apply(norm_set)
        new_idx = [mapping.get(k, old) for k, old in zip(parent_keys, old_index)]
        parents.index = new_idx
    parents.index.name = 'Site Code'
else:
    parents.index.name = 'Site Code'

# collapse duplicates by aggregating covers and taking first for metadata
cover_cols = [c for c in parents.columns if 'Cover' in c]
info_cols = [c for c in parents.columns if c not in cover_cols]
agg_dict = {c: 'mean' for c in cover_cols}
agg_dict.update({c: 'first' for c in info_cols})
parents = parents.reset_index().groupby('Site Code', as_index=True).agg(agg_dict)

# drop helpers and reorder columns
parents = parents.drop(columns=['group_id','x','y','plotName'], errors=True)
cover_cols_sorted = sorted([c for c in parents.columns if 'Cover' in c])
info_cols_sorted = [c for c in parents.columns if c not in cover_cols_sorted]
parents = parents[cover_cols_sorted + info_cols_sorted]

# export parents
parents_out = parents.rename(columns=schema)
parents_out.index.name = 'Site Code'
# parents_out = parents_out[parents_out['Site Code'].isin(orig_child['Site Code'])].reset_index(drop=True)
parents_out.to_csv(f'{wdir}/{dst}_fcover_parent_{vers}.csv')