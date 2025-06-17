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

from alaska_pft_fcover_harmonization.pavc_funcs.file_validation_schemas import SCHEMAS

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