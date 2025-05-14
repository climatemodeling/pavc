# schemas.py
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Union

# --- Schema: species_fcover ---
class SpeciesFcoverRow(BaseModel):
    class Config:
        extra = "ignore"
    plotVisit: str = Field(..., min_length=10)
    datasetSpeciesName: str
    standardHabit: str
    nonstandardHabit: Optional[str]
    percentCover: Optional[float]

# --- Schema: pft_fcover ---
class PftFcoverRow(BaseModel):
    class Config:
        extra = "ignore"
    plotVisit: str = Field(..., min_length=10)
    deciduousShrubCover: Optional[float]
    deciduousTreeCover: Optional[float]
    evergreenShrubCover: Optional[float]
    evergreenTreeCover: Optional[float]
    forbCover: Optional[float]
    graminoidCover: Optional[float]
    nonvascularSumCover: Optional[float]
    bryophyteCover: Optional[float]
    lichenCover: Optional[float]
    litterCover: Optional[float]
    otherCover: Optional[float]
    baregroundCover: Optional[float]
    waterCover: Optional[float]

# --- Schema: pft_aux ---
class PftAuxRow(BaseModel):
    class Config:
        extra = "ignore"
    plotVisit: str = Field(..., min_length=10)
    surveyYear: Optional[int]
    surveyMonth: Optional[int]
    surveyDay: Optional[int]
    plotArea: Optional[float]
    plotShape: Optional[str]
    latitudeY: Optional[float]
    longitudeX: Optional[float]
    georefSource: Optional[str]
    georefAccuracy: Optional[float]
    coordEPSG: Optional[str]
    plotName: Optional[str]
    dataSubsource: Optional[str]
    dataSource: Optional[str]
    dataSourceHtml: Optional[str]
    dataSubsourceCitation: Optional[str]
    surveyMethod: Optional[str]
    fcoverScale: Optional[str]
    surveyPurpose: Optional[str]
    geometry: Optional[Union[str, dict]]  # Could be WKT or GeoJSON
    fireYears: Optional[List[int]]
    duplicatedCoords: Optional[List[str]]
    duplicatedDate: Optional[List[str]]

# --- Schema: synthesized pft_checklist ---
class SpeciesPftChecklist(BaseModel):
    class Config:
        extra = "ignore"
    dataset_species_name: str
    accepted_species_name: str
    accepted_species_name_author: Optional[str]
    visit_id: List[str]
    data_source: List[Optional[str]]
    data_subsource: List[Optional[str]]
    taxon_rank: str
    naming_authority: Optional[str]
    category: Optional[str]
    habit: Optional[str]
    pft: str
    nonstandard_pft: Optional[str]

# --- Schema: synthesized pft fcover ---
class SynthesizedPftFcover(BaseModel):
    class Config:
        extra = "ignore"
    visit_id: str
    deciduous_shrub_cover: Optional[float]
    deciduous_tree_cover: Optional[float]
    evergreen_shrub_cover: Optional[float]
    evergreen_tree_cover: Optional[float]
    forb_cover: Optional[float]
    graminoid_cover: Optional[float]
    nonvascular_sum_cover: Optional[float]
    bryophyte_cover: Optional[float]
    lichen_cover: Optional[float]
    litter_cover: Optional[float]
    bareground_cover: Optional[float]
    water_cover: Optional[float]
    other_cover: Optional[float]

# --- Schema: synthesized plot aux info ---
class SynthesizedAux(BaseModel):
    class Config:
        extra = "ignore"
    visit_id: str
    survey_year: Optional[int]
    survey_month: Optional[int]
    survey_day: Optional[int]
    plot_area: Optional[float]
    plot_shape: Optional[str]
    latitude_y: Optional[float]
    longitude_x: Optional[float]
    georef_source: Optional[str]
    georef_accuracy: Optional[float]
    coord_epsg: Optional[str]
    plot_name: str
    data_subsource: str
    data_source: str
    data_source_html: str
    data_subsource_citation: str
    survey_method: Optional[str]
    fcover_scale: str
    survey_purpose: str
    admin_unit: str
    admin_country: str
    fire_years: List[Optional[int]]
    bioclim_subzone: str
    duplicated_coords: Optional[List[str]]
    duplicated_date: Optional[List[str]]
    cavm_unit: str
    cavm_unit_description: str

# --- Schema: synthesized plot aux info ---
class SynthesizedSpeciesFcover(BaseModel):
    class Config:
        extra = "ignore"
    visit_id: str
    accepted_species_name: str
    fcover: Optional[float]


# --- Schema registry ---
SCHEMAS = {
    "species_fcover": SpeciesFcoverRow,
    "pft_fcover": PftFcoverRow,
    "pft_aux": PftAuxRow,
    "species_pft_checklist": SpeciesPftChecklist,
    "synthesized_pft_fcover": SynthesizedPftFcover,
    "synthesized_aux": SynthesizedAux,
    "synthesized_species_fcover": SynthesizedSpeciesFcover
}
