import requests
import geopandas as gpd
import pandas as pd
import numpy as np
from datetime import date, timedelta
from pyogrio import read_dataframe
import glob
import geopandas as gpd
import os
import standardize_pft_funcs as spf
import chardet
import tarfile
from urllib.request import urlretrieve
import regex as re
from shapely.validation import make_valid

"""
CAVEATS:
The functions in this script were used in the standardization
python jupyter notebooks. They are specific to these notebooks
and are generalized *enough* to work on fcover dataframes that have
been properly formatted. Some of these functions are completely dependent
on the presence of specific input data that is formatted in a particular 
way. For these functions to work, all data should be downloaded and
maintained in the structure in which it is packaged. If input files
or dataframe are changed in anyway, there is no guarantee that these 
functions will remain usable!
"""

##########################################################################################
# Main functions that are used in the notebooks. Roughly in order of usage.
##########################################################################################

def get_unique_species(DFRAME, SCOL, DNAME, SAVE=False, OUTP=False):
    
    """
    Main function that creates a dataframe of unique species names
    for a dataframe containing a specified species name column
    
    DRAME  (dataframe): species-level fcover dataframe
    SCOL      (string): column containing species names, 
                        e.g. 'speciesName'
    DNAME     (string): source dataset name, e.g. 'ava'
    SAVE        (bool): whether or not to save unique species 
                        name to .csv
    OUTP      (string): path to directory where .csv will be saved
    """

    # Get unique species names
    unique_species = DFRAME.groupby([SCOL]).first()
    unique_species = unique_species.reset_index()
    unique_species_df = pd.DataFrame(unique_species[SCOL])
    
    # Export
    if SAVE:
        if not OUTP:
            print('OUTP requires path to output folder.')
        else:
            try:
                path = f'{OUTP}/{DNAME}_unique_species.csv'
                unique_species_df.to_csv(path)
                print(f'Saved unique species list to {path}.')
            except Exception as e:
                print(e)
                
    # Return dataframe
    return unique_species_df


def add_leaf_retention(species_pft, evergrndecid, ret_col_name):
    
    """
    Main function that adds a leaf retention column to a species-level
    dataframe. It matches the species name in one dataframe to the
    species name in the macanander 22 supplementary table.
    
    species_pft  (dataframe): dataframe with species-level fcover data
    evergrndecid (dataframe): dataframe with evergreen/deciduous info
    ret_col_name    (string): name of the new leaf retention column name
    
    """
    
    # add evergreen/deciduous information
    species_pft_l = species_pft.to_numpy()
    evergrndecid_l = evergrndecid.to_numpy()
    
    newlist = []
    for row in species_pft_l:

        # get first 2 words in data species name
        speciesname = row[1].split()[:2]
        newrow = []

        for row2 in evergrndecid_l:
            # get first 2 words in evergreen/decid species name
            speciesname2 = row2[1].split()[:2]
            ed = None

            # if full NEON name == E/D name:
            if speciesname == speciesname2:
                #get evergreen/decid
                ed = row2[0].split()
                ed = ed[0]

            # if data genus == E/D genus:
            elif str(speciesname[0]) == str(speciesname2[0]):
                #print(speciesname[0], 'and', speciesname2[0])
                ed = row2[0].split()
                ed = ed[0]

            # if they don't match, dont append
            else:
                continue
            newrow.append(ed)

        newlist.append(newrow)
    
    # get unique leaf habit values
    final_l = []
    for l in newlist:
        newl = list(set(l))
        string = ','.join(newl)
        final_l.append(string)
        
    species_pft[ret_col_name] = final_l
    return species_pft


def join_to_checklist(unique_species, checklist, u_name, c_unofficial_name, 
                      c_official_name, mapping_name, habit):
    
    """
    Giant main function that iteratively tries to match a species name from
    the fcover data table to a species name in the akveg species checklist
    table. It strts by comparing the genus-species name to the accepted names
    in the checklist. Then, to the synonyms for accepted names. If still no
    match, it will compare the genus name to the accepted genus name, and 
    then the genus name to the synonym name. If no match is found, the habit
    is designated as NaN. If a match(es) is found, it will be the recorded
    habit(s).
    
    unique_species (dataframe): dataframe with fcover species names
    checklist      (dataframe): dataframe with checklist species names
    u_name            (string): column in unique_species dataframe that
                                contains the species names
    c_unofficial_name (string): column in checklist dataframe that contains
                                the possible synonyms for an accepted name
    c_official_name   (string): column in checklist dataframe that contains
                                the accepted species names
    mapping_name      (string): column name that both the unqiue_species and
                                checklist dataframes have (the genus-species
                                name used to join the two dataframes)
    habit             (string): column name from the checklist that contains
                                the habit (PFT) associated with a species
    """
    
    # assign habit to species
    def create_checklist_habits(checklist, mapping_name, habit):

        """
        Subroutine function used in the `join_to_checklist` function
        to generate a comma-separated list of "potential" PFT 
        (habit) names as a value in a pandas dataframe

        checklist (dataframe): the akveg species checklist dataframe
        mapping_name (string): the column name used to merge the 
                               checklist with the fcover dataframe
        habit        (string): the checklist column name containing the 
                               PFT designation for a species
        """

        # For every genus-species, create a list of potential habits
        checklist_merge = (checklist
                           .groupby(mapping_name)[habit]
                           .apply(set)
                           .reset_index())

        # Make lists strings, remove brackets, remove whitespace
        checklist_merge[habit] = (checklist_merge[habit]
                                  .astype(str)
                                  .str.strip('{}')
                                  .str.strip("''")
                                  .str.strip())

        # clean up the strings created above
        checklist_merge[habit] = checklist_merge[habit].apply(cleanlist)

        # Return dataframe
        return checklist_merge
    
    ######################################################################################
    # compare genus-species to official name genus-species
    ######################################################################################
    # extract first two words (genus & species) of official name
    unique_species[mapping_name] = unique_species[u_name].apply(get_substrings)
    checklist[mapping_name] = checklist[c_official_name].apply(get_substrings)

    # match unofficial name to blank habit species
    checklist1 = create_checklist_habits(checklist=checklist, 
                                         mapping_name=mapping_name, 
                                         habit=habit)

    # get habit for species based on unofficial name
    species_pft = (unique_species
                   .reset_index()
                   .merge(checklist1, how='left', on=mapping_name)
                   .set_index('index'))

    # check if species habit is blank
    species_pft1 = species_pft.replace('', np.nan)
    missinghabit1 = species_pft1.loc[species_pft1.isna().any(axis=1)]
    print(f'{len(missinghabit1)} species are missing habits.')
    
    ######################################################################################
    # compare genus-species to UNofficial name genus-species
    ######################################################################################
    # extract first two words (genus & species) of official name
    checklist2 = checklist.copy()
    checklist2[mapping_name] = checklist2[c_unofficial_name].apply(get_substrings)

    # match unofficial name to blank habit species
    checklist2 = create_checklist_habits(checklist=checklist2, 
                                         mapping_name=mapping_name, 
                                         habit=habit)

    # get habit for species based on unofficial name
    species_pft2 = (missinghabit1
                    .reset_index()
                    .merge(checklist2, how='left', on=mapping_name, suffixes=['1','2'])
                    .set_index('index'))

    # show species that are still missing habits
    newhabit2 = f'{habit}2'
    missinghabit2 = species_pft2[species_pft2[newhabit2].isnull()]
    print(f'{len(missinghabit2)} species still missing habits.')
    
    ######################################################################################
    # compare genus to official name genus
    ######################################################################################
    # extract genus of official name and species name
    checklist3 = checklist.copy()
    checklist3[mapping_name] = checklist3[mapping_name].apply(get_first_substring)
    missinghabit3 = missinghabit2.copy()
    missinghabit3[mapping_name] = missinghabit2[mapping_name].apply(get_first_substring)

    # match unofficial name to blank habit species
    checklist3 = create_checklist_habits(checklist=checklist3,
                                         mapping_name=mapping_name,
                                         habit=habit)

    # get habit for genus
    species_pft3 = (missinghabit3
                    .reset_index()
                    .merge(checklist3, how='left', on=mapping_name, suffixes=['2','3'])
                    .set_index('index'))

    # show species that are still missing habits
    newhabit3 = habit
    missinghabit3 = species_pft3[species_pft3[newhabit3].isnull()]
    print(f'{len(missinghabit3)} species still missing habits.')
    
    ######################################################################################
    # compare genus to unofficial name genus
    ######################################################################################
    # extract genus of unofficial name and species name
    checklist4 = checklist.copy()
    checklist4[mapping_name] = checklist4[c_unofficial_name].apply(get_first_substring)
    missinghabit4 = missinghabit3.copy()
    missinghabit4[mapping_name] = missinghabit3[mapping_name].apply(get_first_substring)

    # match unofficial name to blank habit species
    checklist4 = create_checklist_habits(checklist=checklist4,
                                         mapping_name=mapping_name,
                                         habit=habit)

    # get habit for genus
    species_pft4 = (missinghabit3
                    .reset_index()
                    .merge(checklist4, how='left', on=mapping_name, suffixes=['3','4'])
                    .set_index('index'))

    # show species that are still missing habits
    newhabit4 = f'{habit}4'
    missinghabit4 = species_pft4[species_pft4[newhabit4].isnull()]
    print(f'{len(missinghabit4)} species still missing habits.')
    
    ######################################################################################
    # fill missing habit names
    ######################################################################################
    # set up columns for filling
    fill1 = species_pft2[[newhabit2]]
    fill2 = species_pft3[[newhabit3]]
    fill2 = fill2.copy()
    fill2.rename(columns={newhabit3: f'{habit}3'}, inplace=True)
    fill3 = species_pft4[[newhabit4]]
    finalhabits = pd.concat([species_pft, fill1, fill2, fill3], axis=1)
    
    # fill
    finalhabits[habit] = finalhabits[habit].fillna(finalhabits[newhabit2])
    finalhabits[habit] = finalhabits[habit].fillna(finalhabits[f'{habit}3'])
    finalhabits[habit] = finalhabits[habit].fillna(finalhabits[newhabit4])
    finalhabits = finalhabits[[u_name, mapping_name, habit]]
    finalhabits.columns = [u_name, mapping_name, habit]
    
    # return dataframe
    return finalhabits


def add_standard_cols(df):
    
    """
    Main function that creates columns and fills them with NaN
    if they do not already exist in the provided dataframe. Used
    to standardize the dataset tables.
    
    df (dataframe): dataframe that contains the PFT-level fcover
                    data
    
    """
    
    # required columns
    necessary_cols = ['deciduous dwarf shrub cover (%)',
                      'deciduous dwarf to low shrub cover (%)',
                      'deciduous dwarf to tall shrub cover (%)',
                      'deciduous dwarf to tree cover (%)',
                      'deciduous tree cover (%)',
                      'evergreen dwarf shrub cover (%)',
                      'evergreen dwarf to low shrub cover (%)',
                      'evergreen dwarf to tall shrub cover (%)',
                      'evergreen dwarf to tree cover (%)',
                      'evergreen tree cover (%)',
                      'bryophyte cover (%)',
                      'forb cover (%)',
                      'graminoid cover (%)',
                      'lichen cover (%)']
    
    # add missing columns and fill with nan
    cols = df.columns.tolist()
    addcols = []
    for nc in necessary_cols:
        if nc not in cols:
            addcols.append(nc)
    df[addcols] = np.nan
    return df


def neon_plot_centroids(dfs, DIR):
    
    """
    Main function for NEON data that is used to extract
    the centroid coordinates for the 1-meter-level plots;
    this data has to be queried online. The coordinates
    provided in the .csv are for the larger 40-meter plot.
    This function is not generalizable at all, apologies.
    
    dfs   (list): list containing pandas dataframes with neon
                  fcover data (pandas dataframes created from
                  the TOOL.csv and BARR.csv
    DIR (string): path to the output .csv that combines TOOL
                  and BARR .csvs
    """
    
    # create name column to exract plot centroids
    df = pd.concat(dfs)
    df.reset_index(inplace=True, drop=True)
    df['name'] = df.namedLocation + '.' + df.subplotID
    
    # get subplot lat/lon from server
    requests_dict = {}
    plots = df['name'].to_list()
    url = 'http://data.neonscience.org/api/v0/locations/'
    locs = []

    for plot in plots:

        # only get response when the request is new
        response = None
        if plot not in list(requests_dict.keys()):
            response = requests.get(url + plot)
            requests_dict[plot] = response
        else:
            response = requests_dict[plot]
        # print(url + plot, end='\r')

        # extract lat/lon
        try:
            lat = response.json()['data']['locationDecimalLatitude']
            lon = response.json()['data']['locationDecimalLongitude']
        except Exception as e:
            # print(response.content)
            print(e)
            lat, lon = None, None
        locs.append([lat,lon])
        
    # add coordinate data to rows
    coords = pd.DataFrame(locs, columns=['subplot_lat','subplot_lon'])
    new_df = pd.concat([df, coords], axis=1)
    new_df.to_csv(DIR + '/NEON.D18.TOOLBARR.DP1.10058.001.div_1m2Data.2021.csv')
    

def leaf_retention_df(path):
    
    """
    Main function that reads and cleans the Macander 2022 leaf 
    retention supplementary table into a pandas dataframe.
    
    path (string): path to the Macander 2022 leaf retention table
    """
    
    df = pd.read_csv(path, header=None)
    df.columns = ['leafRetention', 'retentionSpeciesName']
    df.replace(to_replace='Deciduous Shrubs', value='deciduous', inplace=True)
    df.replace(to_replace='Evergreen Shrubs', value='evergreen', inplace=True)
    return df


def checklist_df(path):
    
    """
    Main function that reads and cleans the AKVEG species checklist
    table into a pandas dataframe.
    
    path (string): path to the AKVEG species checklist table
    """
    
    df = read_dataframe(path)
    df.rename(columns={'Code': 'nameCode',
                       'Name':'checklistSpeciesName',
                       'Status': 'nameStatus',
                       'Accepted Name': 'nameAccepted',
                       'Family': 'nameFamily',
                       'Name Source': 'acceptedNameSource',
                       'Level': 'nameLevel',
                       'Category': 'speciesForm',
                       'Habit': 'speciesHabit'},
              inplace=True)
    return df


def export_habit_files(habits_df, outdir, dataname, habitcol):
    
    """
    Main function used to group and export the species and
    designated PFT into three files: one with species that
    are shrubs, one with species that are not shrubs, and
    ones that did not successfully match with a habit in the
    `join_to_checklist` function.
    
    habits_df (dataframe): dataframe containing the species-habit data
    outdir       (string): path to where the .csvs will be exported
    dataname     (string): datasource name, e.g. 'ava'
    habitcol     (string): column name containing the PFTs
    """
    
    habits = habits_df.copy()
    # export all shrub species
    nonnull = habits[~habits[habitcol].isnull()]
    shrubs = nonnull[nonnull[habitcol].str.contains('shrub')]
    shrubs.to_csv(f'{outdir}/{dataname}_shrubs.csv')
    
    # export all non-shrub species
    nonshrubs = nonnull[~nonnull[habitcol].str.contains('shrub')]
    nonshrubs.to_csv(f'{outdir}/{dataname}_nonshrubs.csv')
    
    # get null habits
    null = habits[habits[habitcol].isnull()]
    null.to_csv(f'{outdir}/{dataname}_nullhabit.csv')
    
    return shrubs, nonshrubs, null


def add_standard_cols(df, pft_cols):
    
    """
    Main function that adds columns not present in a dataframe
    given a list of column names. Returns dataframe with new
    columns added and filled with NaN.
    
    df  (dataframe): dataframe to add columns to
    pft_cols (list): list of strings that are columns to add
                     to the dataframe
    """
    
    # add missing columns and fill with nan
    cols = df.columns.tolist()
    addcols = []
    for nc in pft_cols:
        if nc not in cols:
            addcols.append(nc)
    df[addcols] = np.nan
    return df

def add_geospatial_aux(df, paths, names, colnames, epsg):
    
    """
    Main function that, given a list of paths, reads a shapefile
    into a geodataframe. Then, it fixes any invalid geometry and
    finds the intersection between a provided dataframe of points 
    and the shapefile geodataframes. Geodataframes must be in the
    same projection (EPSG:4326 yields incorrect results; choose a
    projected EPSG).
    
    df  (dataframe): geodataframe of points to add intersections to
    paths    (list): list of paths to shapefiles of polygons
    names    (list): list of names for the shapefiles of polygons
    colnames (list): list of lists containing the column names to
                     keep during intersection
    epsg   (string): EPSG code indicating a shared projection 
                     between the df and shapefiles
    """
    
    new_df = df.copy()
    
    def fix_geometries(gdf):
        valid_geoms = gdf[gdf['geometry'].notna()].copy()
        none_geoms = gdf[gdf['geometry'].isna()].copy()
        valid_geoms['geometry'] = valid_geoms['geometry'].apply(make_valid)
        return valid_geoms
    
    def read_geospatial(path):
        gdf = gpd.read_file(path)
        gdf = gdf.to_crs(epsg)
        gdf = fix_geometries(gdf)
        return gdf
    
    for path, name, cnames in zip(paths, names, colnames):
        
        gdf = read_geospatial(path)
        gdf = fix_geometries(gdf)
        new_df = gpd.sjoin(new_df, gdf[cnames], 
                           how='left', predicate='intersects', rsuffix=name)
        
    return new_df

# populates a column with the indicies of duplicated
# information; e.g., duplicate coords or dates
def find_duplicates(df, subset, col_name):
    
    """
    Main function that adds columns to a dataframe indicating
    if a subset of columns are duplicated. E.g., if the same
    latitude and longitude are found in multiple rows, the
    indices of those duplicate rows will be recorded as a list
    in a column.
    
    df    (dataframe): dataframe to check for duplicates and add
                       columns to
    subset     (list): list of column names that will be checked
                       for duplicate values
    col_name (string): column to create that stores a list of
                       indices where values are duplicated
    """
    
    df = df.copy()
    if df.duplicated(subset=subset).any():
        print('duplicates found')
        dupes = df.duplicated(subset=subset, keep=False)
        dupe_groups = df[dupes].groupby(subset).apply(
            lambda x: list(x.index)).reset_index(name='indices')
        idx_to_dupes = {idx: indices for indices in dupe_groups['indices'] for idx in indices}
        df[col_name] = df.index.map(idx_to_dupes)
    else:
        print('no duplicates found')
    return df

    
##########################################################################################
# Pandas row-wise functions to use with .apply()
##########################################################################################

def get_substrings(row):

    # NOTE: DO NOT modify this code to change the source words in any way
    # because then you won't be able to match new names to originals later
    
    if not isinstance(row, str) or not row.strip():  # Handle empty or non-string inputs
        return None
    
    words = row.split()[:2] # assumed output: [genus, species]
    lowercase_words = [word.lower() for word in words]
    
    if len(words) == 0:  # edge case: empty list
        return None
    elif len(words) == 1 or 'species' in lowercase_words:  # Edge case: only one word present
        return words[0]
    elif lowercase_words[0] == 'unknown' and len(words) >1:  # Handling 'Unknown' case
        return words[1]
    else:
        return ' '.join(words[:2]) # return the words


# function get genus name only
def get_first_substring(row):
    
    # extract genus name
    words = row.split()[:1]
    string = ' '.join(words)
    return string


# get unique values in a list
def uniquelist(row):
    
    rowlist = list(row)
    unique = set(rowlist)
    return list(unique)


# clean list appearance
def cleanlist(row):
    
    new = row.strip().replace("'", '')
    newlist = new.split(',')
    newlist = list(set(newlist))
    return ','.join(newlist)


# simplify shrub names
def clean_shrub_habits(row):
    if isinstance(row, float):
        return np.nan
    if 'shrub' in row:
        return 'shrub'
    else:
        return row