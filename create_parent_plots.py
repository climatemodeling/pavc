### AVA ###

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# load cover file
p = '/mnt/poseidon/remotesensing/arctic/data/training/Test_05/temp/'
f = 'AVA_fcover_child.csv'
akava = pd.read_csv(p + f)

# only plots with very small sizes need to be processed, 
# the other remains the same, discarding all plots earlier than 2010
akava['plot_radius_m'].astype(float)
akava_small_fcover = akava[(akava['plot_radius_m']<10) &  (akava['year']>=2010)]
akava_large_fcover = akava[(akava['plot_radius_m']>=10) &  (akava['year']>=2010)]

selected_columns = ['Site Code','year','latitude', 'longitude', 'source', 'subsource', 'plot_radius_m']
akava_small = akava_small_fcover[selected_columns]
# akava_small = akava.fillna(0)
akava_small.head()

#### transform geographic to utm so that distance caculation is more intuitive
from pyproj import Transformer
src_crs = "EPSG:4326"
target_crs = "EPSG:32606"
transformer = Transformer.from_crs(src_crs, target_crs)

lon = akava_small['longitude'].to_numpy()
lat = akava_small['latitude'].to_numpy()

#### store the projected coords
projcoords = []   
for i in range(0,akava_small.shape[0]):
    xcoord, ycoord = lon[i],lat[i]
    projcoords.append(transformer.transform(ycoord,xcoord))
    
#### calculate the distance matrix of all small plots (for examination)
from scipy.spatial.distance import cdist
coord = np.array(projcoords)
dist_mat = cdist(coord, coord, 'euclidean')
# dist_mat[:3,:] <= 60

### group pixels/plots based on their euclidean distance
def group_pixels_by_distance(pixel_data, distance_threshold):
    """
        pixel_data: ndarray of coordinate pair: n by 2, default is utm projection
        distance_threshold: threshold used for grouping, default is 60m
        
        return:
        a list of values indicating the group id of each pixel
    
    """
    cluster_id = 0
    pixel_clusters = {}
   
    def expand_cluster(pixel, cluster_id):
        if pixel_clusters.get(cluster_id) is None:
            pixel_clusters[cluster_id] = []
       
        pixel_clusters[cluster_id].append(pixel)

    cluster_array = np.full(len(pixel_data), -1)  # Initialize with -1 (unassigned)
   
    for i, pixel in enumerate(pixel_data):
        assigned = False
       
        for c_id, cluster_pixels in pixel_clusters.items():
            cluster_pixels = np.array(cluster_pixels)
            distances = np.linalg.norm(cluster_pixels - pixel, axis=1)
            if np.any(distances <= distance_threshold):
                expand_cluster(pixel, c_id)
                assigned = True
                cluster_array[i] = c_id
                break
       
        if not assigned:
            expand_cluster(pixel, cluster_id)
            cluster_array[i] = cluster_id
            cluster_id += 1

    return cluster_array

dist_thres = 55
coord = np.array(projcoords)  ## UTM coords, unit is meter
group = group_pixels_by_distance(coord, dist_thres)

### add the group id to df for aggregation
akava_small['group_id'] = group

groups = akava_small.groupby(['group_id', 
                              'year', 'source']).agg({'latitude':'mean',
                                            'longitude':'mean',
                                            'Site Code':list,
                                            'plot_radius_m':list,
                                            'subsource':set})

def get_plot_size(rowlst):
    if len(rowlst) == 1:
        val = rowlst[0]
    else:
        val = 55
    return val
groups['plot_radius_m'] = groups['plot_radius_m'].apply(lambda row: get_plot_size(row))
groups

groups2 = groups.explode('Site Code')
groups2.reset_index(inplace=True)
groups2['parent_id'] = (groups2['group_id'].astype(str) + 
                        '_' + groups2['year'].astype(str) +
                        '_' + groups2['source'])
groups2.rename(columns={'latitude':'parent_latitude',
                        'longitude':'parent_longitude'},
               inplace=True)
groups2.drop(columns=['group_id', 'year', 'source', 'subsource'], inplace=True)
groups2

groups2.set_index('Site Code', inplace=True)
asf = akava_small_fcover.set_index('Site Code')
asf.drop(columns=['plot_radius_m'], inplace=True)
joined = pd.concat([asf, groups2], axis=1)
joined.columns

info_cols = ['parent_latitude', 'parent_longitude', 'plot_radius_m'] #create parent coord by finding mean
data_cols = [col for col in joined.columns if 'cover (%)' in col]
anci_cols = set(joined.columns) - set(data_cols) - set(info_cols)

info_dict = dict.fromkeys(info_cols, 'mean')
data_dict = dict.fromkeys(data_cols, 'mean')
anci_dict = dict.fromkeys(anci_cols, set)
d = {**data_dict, **info_dict, **anci_dict}

parent_plots = joined.groupby('parent_id').agg(d)

parent_plots.drop(columns=['latitude', 'longitude', 'parent_id', 
                           'source'],
                 inplace=True)

parent_plots['year'] = parent_plots['year'].explode()
parent_plots['source'] = 'AKAVA'
parent_plots['subsource'] = parent_plots['subsource'].apply(list)
parent_plots.rename(columns={'Site Code':'child_site_codes',
                             'parent_longitude':'longitude',
                             'parent_latitude':'latitude'}, 
                    inplace=True)
parent_plots.index.name = 'Site Code'

p = '/mnt/poseidon/remotesensing/arctic/data/training/Test_05/temp/'
f = 'AVA_fcover_parent.csv'
parent_plots.to_csv(p + f)