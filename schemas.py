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
    adminUnit: Optional[str]
    adminCountry: Optional[str]
    fireYears: Optional[List[int]]
    bioclimSubzone: Optional[str]
    duplicatedCoords: Optional[List[str]]
    duplicatedDate: Optional[List[str]]

# --- Schema registry ---
SCHEMAS = {
    "species_fcover": SpeciesFcoverRow,
    "pft_fcover": PftFcoverRow,
    "pft_aux": PftAuxRow
}
