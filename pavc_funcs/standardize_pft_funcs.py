import requests
import geopandas as gpd
import pandas as pd
import numpy as np
from datetime import date, timedelta
from pyogrio import read_dataframe
import glob
import geopandas as gpd
import os
import chardet
import tarfile
from urllib.request import urlretrieve
import regex as re
from shapely.validation import make_valid
import hashlib
from typing import get_args, get_origin, Union, List, Literal
import csv
from pydantic import ValidationError
from pyproj import Transformer
import warnings
import ast

from pavc_funcs.schemas import SCHEMAS

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

def _is_list_annotation(annotation) -> bool:
    """
    Returns True if the annotation is List[...] or Optional[List[...]].
    """
    origin = get_origin(annotation)
    if origin is list:
        return True
    if origin is Union:
        return any(get_origin(arg) is list for arg in get_args(annotation))
    return False

def validate_synthesized_species_fcover(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate that `df` has exactly the columns and pandas dtypes declared
    in the SynthesizedSpeciesFcover model.
    Raises a ValueError if any column is missing or any dtype mismatches.
    Returns df unchanged if all checks pass.
    """
    # 1) Grab the Pydantic model
    Model = SCHEMAS["synthesized_species_fcover"]

    # 2) Build expected pandas dtype map by inspecting Model.model_fields
    expected: dict[str, str] = {}
    for name, field in Model.model_fields.items():
        t      = field.annotation
        origin = get_origin(t)
        args   = get_args(t)

        # map Python types to pandas dtypes
        if t is str or (origin is Union and set(args) == {str, type(None)}):
            pd_type = "string"
        elif t is float or (origin is Union and set(args) == {float, type(None)}):
            pd_type = "float64"
        elif t is int or (origin is Union and set(args) == {int, type(None)}):
            pd_type = "Int64"
        else:
            # fall back to object for anything else (e.g. lists)
            pd_type = "object"

        expected[name] = pd_type

    # 3) Check for missing columns
    missing = set(expected) - set(df.columns)
    if missing:
        raise ValueError(f"synthesized_species_fcover missing columns: {missing}")

    # 4) Check dtype mismatches
    mismatches = []
    for col, exp in expected.items():
        actual = df[col].dtype.name
        # allow nullable Float64 alias for float64
        if exp == "float64" and actual in ("Float64", "float64"):
            continue
        if actual != exp:
            mismatches.append((col, actual, exp))

    if mismatches:
        msgs = [f"{c!r}: got {got!r}, expected {exp!r}" for c, got, exp in mismatches]
        raise ValueError("synthesized_species_fcover dtype mismatches:\n  " + "\n  ".join(msgs))

    return df


def validate_synthesized_aux(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates a DataFrame against the SynthesizedAux schema:
      1. Promotes index 'visit_id' into a column
      2. Checks for required columns & coerces dtypes
      3. Enforces visit_id uniqueness & prefix 'u'
      4. Reprojects coords into EPSG:4326 & checks bounds
      5. Whitelists survey_method values
      6. Normalizes any List[...] or Optional[List[...]] fields into real Python lists
         and drops any exploded '.0', '.1', etc. columns
      7. Runs each row through the SynthesizedAux Pydantic model

    Returns:
        A new DataFrame indexed by visit_id, fully validated.
    """
    df = df.copy()
    Model = SCHEMAS.get("synthesized_aux")
    if Model is None:
        raise KeyError("No schema found for key 'synthesized_aux'")

    # 1) Promote index to visit_id
    if df.index.name != "visit_id":
        raise ValueError(f"Expected index name 'visit_id', got {df.index.name!r}")
    df = df.reset_index()

    # 2) Check required columns
    required = set(Model.model_fields.keys())
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    # 2a) Coerce integer fields
    for col in {"survey_year", "survey_month", "survey_day"} & set(df.columns):
        df[col] = pd.to_numeric(df[col], errors="raise").astype("Int64")
    # 2b) Coerce float fields
    for col in {"plot_area", "latitude_y", "longitude_x", "georef_accuracy"} & set(df.columns):
        df[col] = pd.to_numeric(df[col], errors="raise").astype(float)
    # 2c) Coerce string fields
    str_cols = {name for name, fld in Model.model_fields.items() if fld.annotation is str}
    for col in str_cols & set(df.columns):
        df[col] = df[col].astype("string")

    # 3) visit_id constraints
    if not df["visit_id"].is_unique:
        dupes = df.loc[df["visit_id"].duplicated(), "visit_id"].tolist()
        raise ValueError(f"Duplicate visit_id values: {dupes}")
    bad_prefix = df.loc[~df["visit_id"].str.startswith("u"), "visit_id"].tolist()
    if bad_prefix:
        raise ValueError(f"visit_id must start with 'u': {bad_prefix}")

    # 4) Reproject coords into EPSG:4326 if needed, then check bounds
    transformer_cache = {}
    for idx, epsg in df["coord_epsg"].dropna().items():
        if epsg != "EPSG:4326":
            if epsg not in transformer_cache:
                transformer_cache[epsg] = Transformer.from_crs(epsg, "EPSG:4326", always_xy=True)
            tf = transformer_cache[epsg]
            lon, lat = tf.transform(df.at[idx, "longitude_x"], df.at[idx, "latitude_y"])
            df.at[idx, "longitude_x"] = lon
            df.at[idx, "latitude_y"]  = lat
            df.at[idx, "coord_epsg"]  = "EPSG:4326"

    lon_bad = df.loc[~df["longitude_x"].between(-180, 180), ["visit_id", "longitude_x"]]
    lat_bad = df.loc[~df["latitude_y"].between(-90, 90), ["visit_id", "latitude_y"]]
    if not lon_bad.empty or not lat_bad.empty:
        bad = pd.concat([lon_bad, lat_bad])
        raise ValueError(f"Coordinates out of EPSG:4326 bounds:\n{bad}")

    # 5) survey_method whitelist
    allowed = {
        "center-staked point-intercept along transect",
        "simple plot",
        "plot along transect",
    }
    bad_methods = df.loc[~df["survey_method"].isin(allowed), "survey_method"].unique().tolist()
    if bad_methods:
        raise ValueError(f"Invalid survey_method(s): {bad_methods}")

    # 6) Normalize list fields and drop any exploded .0/.1 columns
    list_fields = [
        name for name, fld in Model.model_fields.items()
        if _is_list_annotation(fld.annotation)
    ]
    for name in list_fields:
        # drop columns like 'fire_years.0', 'duplicated_coords.1', etc.
        pattern = re.compile(rf'^{re.escape(name)}\.\d+$')
        drop_cols = [col for col in df.columns if pattern.match(col)]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        # ensure each cell is a real list
        def _to_list(x):
            if isinstance(x, list):
                return x
            if pd.isna(x):
                return []
            if isinstance(x, str) and x.strip().startswith("["):
                try:
                    return ast.literal_eval(x)
                except Exception:
                    pass
            return [x]

        df[name] = df[name].apply(_to_list)

    # 7) Final Pydantic validation
    validated = []
    for rec in df.to_dict(orient="records"):
        filtered = {k: v for k, v in rec.items() if k in Model.model_fields}
        obj = Model(**filtered)
        validated.append(obj.dict())

    # here we explicitly pass only the schema’s field names,
    # so pandas will not create any `.0` columns
    cols = list(Model.model_fields.keys())
    out = pd.DataFrame.from_records(validated, columns=cols)
    return out.set_index("visit_id")


def validate_synthesized_pft_fcover(df: pd.DataFrame) -> pd.DataFrame:
    schema = SCHEMAS.get("synthesized_pft_fcover")
    if schema is None:
        raise ValueError("synthesized_pft_fcover schema not found in SCHEMAS")

    # 1) pull index into 'visit_id'
    df = df.reset_index()
    idx_col = df.columns[0]
    df = df.rename(columns={ idx_col: 'visit_id' })

    # 2) ensure all schema fields exist
    required = set(schema.__fields__.keys())
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # 3) Coerce & check numeric dtype for all cover fields except visit_id
    cover_cols = [c for c in required if c != 'visit_id']
    for col in cover_cols:
        # convert non-numeric to numeric or raise
        try:
            df[col] = pd.to_numeric(df[col], errors='raise')
        except Exception as e:
            raise ValueError(f"Column `{col}` must be numeric (error converting): {e}")

    # 4a) bryophyte + lichen == nonvascular_sum (treat NaN as 0)
    bryo = df['bryophyte_cover'].fillna(0)
    lich = df['lichen_cover'].fillna(0)
    total_nonvasc = df['nonvascular_sum_cover']
    bad = ~np.isclose(bryo + lich, total_nonvasc)

    if bad.any():
        rows = df.loc[bad, ['visit_id', 'bryophyte_cover', 'lichen_cover', 'nonvascular_sum_cover']]
        raise ValueError(f"Nonvascular sum mismatch:\n{rows.to_string(index=False)}")

    # 4b) water and bareground ≤ 100 → warning only
    too_much = df[(df['water_cover'] > 100) | (df['bareground_cover'] > 100)]
    if not too_much.empty:
        rows = too_much[['visit_id', 'water_cover', 'bareground_cover']]
        warnings.warn(
            f"`water_cover` or `bareground_cover` >100 on these rows:\n"
            f"{rows.to_string(index=False)}",
            UserWarning
        )

    # 5) warning if total <100
    total = df[cover_cols].sum(axis=1, skipna=True)
    for vid, tot in zip(df['visit_id'], total):
        if tot < 100:
            warnings.warn(f"visit_id={vid!r} has total cover {tot:.1f} < 100", UserWarning)

    # 6) validate via Pydantic
    for i, row in df.iterrows():
        data = row.to_dict()
        try:
            schema(**data)
        except ValidationError as e:
            print(f"\n🔍 Validation failed on visit_id={data['visit_id']!r}:")
            print(data)
            raise ValueError(f"Schema error for visit_id={data['visit_id']!r}:\n{e}")

    return df


def validate_species_pft_checklist(df: pd.DataFrame) -> pd.DataFrame:

    # Controlled vocabularies
    VALID_PFT = {
        'evergreen shrub', 'deciduous shrub', 'evergreen tree', 'deciduous tree',
        'graminoid', 'bryophyte', 'lichen', 'other', 'litter', 'water', 'bare ground', 'forb'
    }

    VALID_TAXON_RANK = {
        'type', 'family', 'genus', 'species', 'subspecies', 'variety'
    }

    # Load the correct schema
    schema = SCHEMAS.get("species_pft_checklist")
    if schema is None:
        raise ValueError("species_pft_checklist schema not found in SCHEMAS")

    # --- 1. Strip whitespace from string values, excluding dataset_species_name and non-strings ---
    for col in df.columns:
        if col != 'dataset_species_name' and df[col].dtype == 'object':
            df[col] = df[col].apply(
                lambda x: x.strip() if isinstance(x, str) else x
            )

    # --- 2. Check for invalid PFT and taxon_rank values ---
    invalid_pft = ~df['pft'].isin(VALID_PFT)
    if invalid_pft.any():
        raise ValueError(f"Invalid 'pft' values found:\n{df.loc[invalid_pft, ['dataset_species_name', 'pft']]}")

    invalid_rank = ~df['taxon_rank'].isin(VALID_TAXON_RANK)
    if invalid_rank.any():
        raise ValueError(f"Invalid 'taxon_rank' values found:\n{df.loc[invalid_rank, ['dataset_species_name', 'taxon_rank']]}")

    # --- 3. Ensure dataset_species_name is unique ---
    if not df['dataset_species_name'].is_unique:
        duplicates = df['dataset_species_name'][df['dataset_species_name'].duplicated()]
        raise ValueError(f"Duplicate dataset_species_name values found: {duplicates.tolist()}")

    # --- 4. Validate each row with the schema ---
    for i, row in df.iterrows():
        try:
            schema(**row.to_dict())
        except ValidationError as e:
            print("\nValidation failed on row", i)
            print("Offending values:")
            print(row.to_dict())
            raise ValueError(f"\nRow {i} failed schema validation:\n{e}")

    return df


def format_column_dtypes(df: pd.DataFrame, schema_key: str) -> pd.DataFrame:
    """
    Coerce DataFrame columns to the expected dtypes based on the Pydantic schema.

    Parameters:
        df (pd.DataFrame): DataFrame to format
        schema_key (str): Key from SCHEMAS registry

    Returns:
        pd.DataFrame: DataFrame with coerced column types
    """
    schema = SCHEMAS.get(schema_key)
    if not schema:
        raise ValueError(f"No schema found for key: {schema_key}")

    df = df.copy()
    for name, field in schema.model_fields.items():
        expected_type = field.annotation

        # Handle Optional[...] types
        if get_origin(expected_type) is Union:
            subtypes = [t for t in get_args(expected_type) if t is not type(None)]
            if len(subtypes) == 1:
                expected_type = subtypes[0]  # Use the actual inner type

        # Only format if the column exists
        if name not in df.columns:
            continue

        try:
            if expected_type == float:
                df[name] = pd.to_numeric(df[name], errors="coerce")
            elif expected_type == int:
                df[name] = pd.to_numeric(df[name], errors="coerce").astype("Int64")
            elif expected_type == str:
                df[name] = df[name].astype(str)
            elif expected_type == list or get_origin(expected_type) == list:
                df[name] = df[name].apply(lambda x: x if isinstance(x, list) else ([] if pd.isna(x) else [x]))
        except Exception as e:
            print(f"Warning: Could not coerce column '{name}' to {expected_type}: {e}")

    return df


def export_dataframe(df: pd.DataFrame, path: str, schema_key: str, index: bool = False, uid_col: str = "plotVisit"):
    """
    Validate and export a DataFrame safely using a registered Pydantic schema.

    Parameters:
        df (pd.DataFrame): DataFrame to validate and export
        path (str): Output CSV path
        schema_key (str): Key from SCHEMAS registry to select Pydantic schema
        index (bool): Whether to include index in export
        uid_col (str): Column name used as unique ID (default 'plotVisit')
    """

    schema = SCHEMAS.get(schema_key)
    if not schema:
        raise ValueError(f"No schema found for key: {schema_key}")

    df = df.copy()

    # Handle UID column in index
    if uid_col in df.index.names:
        df = df.reset_index()
        was_index = True
    else:
        was_index = False

    # Validate each row
    for i, row in df.iterrows():
        try:
            schema(**row.to_dict())
        except Exception as e:
            raise ValueError(f"Row {i} failed validation: {e} {row}")

    # Check UID uniqueness (except for species_fcover)
    if uid_col in df.columns and schema_key != "species_fcover":
        if df[uid_col].duplicated().any():
            duplicates = df[df[uid_col].duplicated(keep=False)]
            raise ValueError(
                f"Duplicate values found in UID column: {uid_col}. "
                f"Here are the duplicates:\n{duplicates[[uid_col]]}"
            )

    # Ensure all object-like columns are cast to string
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str)

    # Restore index if originally set and requested
    if was_index and index:
        df = df.set_index(uid_col)

    df.to_csv(path, index=index, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)


def assign_uid_column(
    df: pd.DataFrame,
    id_col: str = 'plotVisit',
    uid_col: str = 'UID',
    uid_length: int = 10,
    prefix: str = 'u'   # <-- single‐letter prefix
) -> pd.DataFrame:
    """
    Generate a stable hash‐based UID from a column, prepending a letter so
    Excel/pandas never misinterpret it as a number.

    uid_length includes the prefix, so by default you get 1 letter + 9 hex chars.
    """
    # make sure prefix is a single ascii‐letter
    assert len(prefix) == 1 and prefix.isalpha(), "prefix must be one letter"

    core_len = uid_length - 1
    def hash_val(val):
        # treat NaN or infinite as “missing”
        if pd.isna(val) or (isinstance(val, (float, np.floating)) and not np.isfinite(val)):
            return f"{prefix}missing"
        raw = hashlib.sha1(str(val).encode()).hexdigest()[:core_len]
        return f"{prefix}{raw}"

    out = df.copy()
    out[uid_col] = out[id_col].map(hash_val)
    return out


def replace_column_with_uid(
    df: pd.DataFrame,
    uid_map: pd.DataFrame,
    id_col: str = 'plotVisit',
    uid_col: str = 'UID',
    set_index: bool = False,
    schema_key: str = None
) -> pd.DataFrame:
    """
    Replaces a specified ID column in a DataFrame with a hashed UID from a mapping,
    ensuring the result is stored as a string. Allows non-unique ID values for
    certain schemas (e.g., species_fcover).

    Parameters:
        df (pd.DataFrame): DataFrame containing the ID column to replace.
        uid_map (pd.DataFrame): Mapping DataFrame with columns [id_col, uid_col].
        id_col (str): Name of the ID column to replace in df.
        uid_col (str): Name of the UID column in uid_map.
        set_index (bool): If True, set the hashed UID as index.
        schema_key (str): Optional schema key to allow UID exceptions (e.g., 'species_fcover').

    Returns:
        pd.DataFrame: Updated DataFrame with replaced and optionally reordered ID column.
    """

    # Check uniqueness unless allowed by schema
    if schema_key != "species_fcover":
        if not df[id_col].is_unique:
            raise Exception(f"There are non-unique values in {id_col}")
        if not uid_map[uid_col].is_unique:
            raise Exception(f"There are non-unique values in {uid_col}")

    # Ensure id_col is string type in both dataframes
    df[id_col] = df[id_col].astype(str)
    uid_map[id_col] = uid_map[id_col].astype(str)

    # Merge and replace old column
    df = df.merge(uid_map[[id_col, uid_col]], on=id_col, how='left')
    df = df.drop(columns=id_col).rename(columns={uid_col: id_col})

    # Reorder columns to keep ID column first
    cols = [id_col] + [c for c in df.columns if c != id_col]
    df = df[cols]

    # Optionally set index
    if set_index:
        df = df.set_index(id_col)

    return df


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

def neon_plot_centroids(dfs, file_paths, DIR):
    """
    Main function for NEON data that extracts the centroid coordinates 
    for the 1-meter-level plots. This data has to be queried online. 
    The coordinates provided in the .csv are for the larger 40-meter plot.

    dfs        (list): List containing pandas DataFrames with NEON fCover data.
    file_paths (list): List of file paths corresponding to the DataFrames.
    DIR       (string): Path to the output directory where the combined .csv will be saved.
    """
    
    # Combine all dataframes into one
    df = pd.concat(dfs)
    df.reset_index(inplace=True, drop=True)
    df['name'] = df.namedLocation + '.' + df.subplotID

    # Extract source names (e.g., TOOL, BARR) and years from file paths
    source_names = set()
    years = set()

    for filepath in file_paths:
        filename = os.path.basename(filepath)
        filename_parts = filename.split('.')
        
        # 'NEON', 'D18', 'BARR', 'DP1', '10058', '001', 'div_1m2Data', '2016-07', 'basic', '20241118T015606Z', 'csv'
        # Extract the site name (SITE)
        source_name = filename_parts[2]
        source_names.add(source_name)

        # Extract the year (YYYY-MM)
        year = filename_parts[7].split('-')[0]
        years.add(year)

    # Construct output filename
    source_str = "_".join(sorted(source_names)) if source_names else "NOSOURCE"
    year_range = f"{min(years)}-{max(years)}" if years else "NOYEAR"
    output_file = f"neon_cover_{source_str}_1m2Data_{year_range}.csv"

    # Get subplot lat/lon from server
    requests_dict = {}
    plots = df['name'].to_list()
    url = 'http://data.neonscience.org/api/v0/locations/'
    locs = []

    for plot in plots:
        response = None
        if plot not in requests_dict:
            print(url + plot)
            try:
                response = requests.get(url + plot)
                requests_dict[plot] = response
            except Exception as e:
                print('Exception occurred: ', e)
        else:
            response = requests_dict[plot]

        # Extract lat/lon
        try:
            lat = response.json()['data']['locationDecimalLatitude']
            lon = response.json()['data']['locationDecimalLongitude']
        except Exception as e:
            print('Exception occurred: ', e)
            lat, lon = None, None
        locs.append([lat, lon])
        
    # Add coordinate data to rows
    coords = pd.DataFrame(locs, columns=['subplot_lat','subplot_lon'])
    new_df = pd.concat([df, coords], axis=1)

    # Save output file
    new_df.to_csv(os.path.join(DIR, output_file), index=False)
    print(f"Saved file: {output_file}")
    

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


def add_geospatial_aux(
    points_gdf: gpd.GeoDataFrame,
    paths: List[str],
    orig_names: List[List[str]],
) -> gpd.GeoDataFrame:
    """
    Adds attributes from multiple polygon shapefiles to a point GeoDataFrame
    via spatial join. Maintains original index length and adds shapefile-derived
    columns with unique suffixes to avoid name collisions.

    Parameters:
    df          (GeoDataFrame): Point GeoDataFrame.
    idx_col              (str): Name to give the index column
    paths          (List[str]): Paths to shapefiles.
    orig_names (List[List[str]]): Column names to retain per shapefile (must include 'geometry').
    epsg                 (str): EPSG code for projection (must be projected, not WGS84).

    Returns:
    GeoDataFrame with additional attributes from shapefile intersections.
    """

    points_gdf = points_gdf.copy()
    aux_dataframes = []

    def fix_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        gdf = gdf[gdf.geometry.notna()].copy()
        gdf['geometry'] = gdf['geometry'].apply(make_valid)
        return gdf

    def read_and_prepare(path: str, columns: List[str]) -> gpd.GeoDataFrame:
        gdf = gpd.read_file(path)
        gdf = fix_geometries(gdf)
        return gdf[columns], gdf.crs

    for path, orig_cols in zip(paths, orig_names):
        poly_gdf, poly_crs = read_and_prepare(path, orig_cols)
        print(f"Finding spatial intersection for '{orig_cols}' in native CRS: {poly_crs}")

        # We only need the index and points data from the points_gdf
        projected_points = points_gdf[['geometry']].to_crs(poly_crs)

        # Perform join between projected_points (geometry) and poly_gdf[orig_names]
        joined = gpd.sjoin(projected_points, poly_gdf, how='left', predicate='intersects')

        # Group by index and aggregate (set, otherwise default first)
        agg_dict = {col: set for col in orig_cols}

        # Ensure cell values are strings or list of strings
        def normalize_cell(val):
            if not isinstance(val, set):
                return val  # leave untouched
            val = {v for v in val if pd.notna(v)}  # drop nan from set
            if not val:
                return None
            val = [str(v) for v in val]
            return val[0] if len(val) == 1 else val

        grouped = joined.groupby(level=0).agg(agg_dict)
        grouped = grouped.applymap(normalize_cell)
        aux_dataframes.append(grouped)

    new_cols = pd.concat(aux_dataframes, axis=1)
    merged = points_gdf.merge(new_cols, how='left', left_index=True, right_index=True)
    merged = merged.drop(columns=[col for col in merged.columns if 'geometry' in col])

    def clean_set_cell(val):
        if isinstance(val, set):
            if len(val) == 0:
                return None
            # Drop null-like items
            non_null = {v for v in val if not pd.isna(v)}
            if len(non_null) == 0:
                return None
            elif len(non_null) == 1:
                return next(iter(non_null))  # unwrap the single item
            else:
                return list(non_null)        # convert to list
        return val  # leave non-sets untouched

    merged = merged.applymap(clean_set_cell)
    return merged


# populates a column with the indicies of duplicated
# information; e.g., duplicate coords or dates
def find_duplicates(df, subset, col_name):
    """
    For each row, find other rows that have the same values in `subset`.
    Adds a new column that contains the list of matching indices (including itself),
    with indices converted to strings for schema compatibility.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to check for duplicates and add the result column to.
    subset : list of str
        Column names to check for matching values.
    col_name : str
        Name of the new column to create, which will contain lists of matching indices.

    Returns
    -------
    pandas.DataFrame
        A copy of df with `col_name` added. Each row contains a list of matching index values
        (as strings), or an empty list if no duplicates were found.
    """
    df = df.copy()

    # Create a group id based on the subset columns
    group_ids = df.groupby(subset, sort=False).grouper.group_info[0]

    # Map from group id to list of indices in that group
    from collections import defaultdict
    group_to_indices = defaultdict(list)
    for idx, group_id in zip(df.index, group_ids):
        if group_id != -1:
            group_to_indices[group_id].append(idx)

    # Now, for each row, assign the corresponding list of indices
    duplicated_list = []
    for idx, group_id in zip(df.index, group_ids):
        if group_id == -1:
            duplicated_list.append([])
        else:
            indices = group_to_indices[group_id]
            if len(indices) > 1:
                duplicated_list.append([str(i) for i in indices])  # Convert to strings here
            else:
                duplicated_list.append([])

    # Assign to the new column
    df[col_name] = duplicated_list

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