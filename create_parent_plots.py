import pandas as pd
import numpy as np
from pyproj import Transformer
import warnings
 
# suppress warnings
warnings.filterwarnings('ignore')

# parameters
src = 'akveg'
dst = 'VEG'
vers = '03'
wdir = '../data/training/Test_06/fcover'
dist_thres = 55          # meters
small_area_thr = 314     # m² threshold for small plots

# --- 0) Set export naming schema ---
schema = {'surveyYear':'year',
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
cover = pd.read_csv(f'{src}/output_data/{src}_standard_pft_fcover.csv', dtype=override_dtype)
info = pd.read_csv(f'{src}/output_data/{src}_plot_info.csv', dtype=override_dtype)
info = info[[
    'plotVisit','surveyYear',
    'latitudeY','longitudeX','dataSource','dataSubsource',
    'plotArea','plotShape','surveyMethod','fcoverScale'
]]

# merge and filter by year ≥ 2010
child = pd.merge(cover, info, on='plotVisit', how='outer')
child = child[child['surveyYear'] >= 2010].reset_index(drop=True)

# export
child_out = child.copy()
child_out = child_out.rename(columns=schema)
child_out.index = child_out['plotVisit']
child_out.index.name = 'Site Code'
child_out = child_out.drop(columns=['plotVisit'])
child_out.to_csv(f'{wdir}/{dst}_fcover_child_{vers}.csv')

# project to UTM once, and stash coords on every row
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32606", always_xy=True)
xy = np.array([transformer.transform(lon, lat)
               for lon, lat in zip(child.longitudeX, child.latitudeY)])
child['x'], child['y'] = xy[:,0], xy[:,1]

# split small vs large
child_small = child[child.plotArea < small_area_thr].copy()
child_large = child[child.plotArea >= small_area_thr].copy()

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

if not child_small.empty:
    coords = child_small[['x','y']].to_numpy()
    child_small['group_id'] = group_by_distance(coords, dist_thres)
else:
    child_small['group_id'] = np.array([], dtype=int)

# --- 3) aggregate small-plot clusters, keeping all coords in lists ---
cover_cols = [c for c in child_small.columns if 'Cover' in c]
agg = child_small.groupby(
    ['group_id','surveyYear','dataSource'], as_index=False
).agg(
    latitudeY       = ('latitudeY','mean'),
    longitudeX      = ('longitudeX','mean'),
    x_coords        = ('x',      lambda s: list(s)),
    y_coords        = ('y',      lambda s: list(s)),
    child_plotVisit = ('plotVisit', list),
    plotArea_orig   = ('plotArea', list),
    plotShape_orig  = ('plotShape', list),
    # compute the MOST FREQUENT string in each group
    dataSubsource   = (
        'dataSubsource',
        lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]
    ),
    surveyMethod    = (
        'surveyMethod',
        lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]
    ),
    fcoverScale     = (
        'fcoverScale',
        lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]
    ),
    **{col: (col,'mean') for col in cover_cols}
)

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

# now build the new plotArea column
agg['plotArea'] = [
    compute_parent_area(xs, ys, orig) 
    for xs, ys, orig in zip(agg.x_coords, agg.y_coords, agg.plotArea_orig)
]

# and similarly set shape = 'circle' whenever we merged >1 child
agg['plotShape'] = [
    'circle' if len(xs) > 1 else orig_shape[0]
    for xs, orig_shape in zip(agg.x_coords, agg.plotShape_orig)
]

# build plotVisit ID
agg['plotVisit'] = np.where(
    agg.child_plotVisit.str.len() > 1,
    agg.group_id.astype(str)
      + '_' + agg.surveyYear.astype(str)
      + '_' + agg.dataSource,
    agg.child_plotVisit.str[0]
)

# finalize parents_small
parents_small = agg.set_index('plotVisit').drop(
    columns=['group_id','plotArea_orig','plotShape_orig','x_coords','y_coords']
)

# --- 4) large plots pass through as before ---
parents_large = child_large.set_index('plotVisit')

# --- 5) concatenate and export ---
parents = pd.concat([parents_small, parents_large], axis=0)
parents = parents.drop(columns=['x','y'], errors=True)
parents.index.name = 'plotVisit'

# reorder columns
cover_cols_sorted = sorted([c for c in parents.columns if 'Cover' in c])
info_cols_sorted  = [c for c in parents.columns if c not in cover_cols_sorted]
parents = parents[cover_cols_sorted + info_cols_sorted]

# export
parents_out = parents.copy()
parents_out = parents_out.rename(columns=schema)
parents_out.index.name = 'Site Code'
parents_out.to_csv(f'{wdir}/{dst}_fcover_parent_{vers}.csv', index=True)