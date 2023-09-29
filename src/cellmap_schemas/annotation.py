from __future__ import annotations
from datetime import date
from enum import Enum
from typing import Any, Dict, Generic, List, Literal, Mapping, Optional, Protocol, TypeVar, Union, runtime_checkable
from pydantic_zarr import GroupSpec, ArraySpec
from pydantic import BaseModel, Field, root_validator
from pydantic.generics import GenericModel


class StrictBase(BaseModel):
    class Config:
        extra = "forbid"


T = TypeVar("T")

class CellmapWrapper(StrictBase, GenericModel, Generic[T]):
    """
    A generic pydantic model that wraps the type `T` under the namespace "cellmap"

    Attributes
    ----------
    
    cellmap: T
        `T`, but accessed via the attribute `cellmap`.

    Examples
    --------

    ```python
    from pydantic import BaseModel
    
    class Foo(BaseModel):
        bar: int

    print(CellmapWrapper[Foo](cellmap={'bar': 10}).dict())
    # {'cellmap': {'bar': 10}}
    ```

    """
    cellmap: T


class AnnotationWrapper(StrictBase, GenericModel, Generic[T]):
    """
    A generic pydantic model that wraps the type `T` under the namespace "annotation"

    Attributes
    ----------
    
    annotation: T
        `T`, but accessed via the attribute `annotation`.

    Examples
    --------

    ```python
    from pydantic import BaseModel
    
    class Foo(BaseModel):
        bar: int

    print(AnnotationWrapper[Foo](annotation={'bar': 10}).dict())
    # {'annotation': {'bar': 10}}
    ```
    """
    annotation: T

def wrap_attributes(attributes: T) -> CellmapWrapper[AnnotationWrapper[T]]:
    return CellmapWrapper(
        cellmap=AnnotationWrapper(
            annotation=attributes
            ))

class InstanceName(StrictBase):
    long: str
    short: str


class Annotated(str, Enum):
    dense: str = "dense"
    sparse: str = "sparse"
    empty: str = "empty"


class AnnotationState(StrictBase):
    present: bool
    annotated: Annotated


class Label(StrictBase):
    value: int
    name: InstanceName
    annotationState: AnnotationState
    count: Optional[int]


class LabelList(StrictBase):
    labels: List[Label]
    annotation_type: AnnotationType = "semantic"


classNameDict = {
    1: InstanceName(short="ECS", long="Extracellular Space"),
    2: InstanceName(short="Plasma membrane", long="Plasma membrane"),
    3: InstanceName(short="Mito membrane", long="Mitochondrial membrane"),
    4: InstanceName(short="Mito lumen", long="Mitochondrial lumen"),
    5: InstanceName(short="Mito DNA", long="Mitochondrial DNA"),
    6: InstanceName(short="Golgi Membrane", long="Golgi apparatus membrane"),
    7: InstanceName(short="Golgi lumen", long="Golgi apparatus lumen"),
    8: InstanceName(short="Vesicle membrane", long="Vesicle membrane"),
    9: InstanceName(short="Vesicle lumen", long="VesicleLumen"),
    10: InstanceName(short="MVB membrane", long="Multivesicular body membrane"),
    11: InstanceName(short="MVB lumen", long="Multivesicular body lumen"),
    12: InstanceName(short="Lysosome membrane", long="Lysosome membrane"),
    13: InstanceName(short="Lysosome lumen", long="Lysosome membrane"),
    14: InstanceName(short="LD membrane", long="Lipid droplet membrane"),
    15: InstanceName(short="LD lumen", long="Lipid droplet lumen"),
    16: InstanceName(short="ER membrane", long="Endoplasmic reticulum membrane"),
    17: InstanceName(short="ER lumen", long="Endoplasmic reticulum membrane"),
    18: InstanceName(
        short="ERES membrane", long="Endoplasmic reticulum exit site membrane"
    ),
    19: InstanceName(short="ERES lumen", long="Endoplasmic reticulum exit site lumen"),
    20: InstanceName(short="NE membrane", long="Nuclear envelope membrane"),
    21: InstanceName(short="NE lumen", long="Nuclear envelope lumen"),
    22: InstanceName(short="Nuclear pore out", long="Nuclear pore out"),
    23: InstanceName(short="Nuclear pore in", long="Nuclear pore in"),
    24: InstanceName(short="HChrom", long="Heterochromatin"),
    25: InstanceName(short="NHChrom", long="Nuclear heterochromatin"),
    26: InstanceName(short="EChrom", long="Euchromatin"),
    27: InstanceName(short="NEChrom", long="Nuclear euchromatin"),
    28: InstanceName(short="Nucleoplasm", long="Nucleoplasm"),
    29: InstanceName(short="Nucleolus", long="Nucleolus"),
    30: InstanceName(short="Microtubules out", long="Microtubules out"),
    31: InstanceName(short="Centrosome", long="Centrosome"),
    32: InstanceName(short="Distal appendages", long="Distal appendages"),
    33: InstanceName(short="Subdistal appendages", long="Subdistal appendages"),
    34: InstanceName(short="Ribosomes", long="Ribsoomes"),
    35: InstanceName(short="Cytosol", long="Cytosol"),
    36: InstanceName(short="Microtubules in", long="Microtubules in"),
    37: InstanceName(short="Nucleus combined", long="Nucleus combined"),
    38: InstanceName(short="Vimentin", long="Vimentin"),
    39: InstanceName(short="Glycogen", long="Glycogen"),
    40: InstanceName(short="Cardiac neurons", long="Cardiac neurons"),
    41: InstanceName(short="Endothelial cells", long="Endothelial cells"),
    42: InstanceName(short="Cardiomyocytes", long="Cardiomyocytes"),
    43: InstanceName(short="Epicardial cells", long="Epicardial cells"),
    44: InstanceName(
        short="Parietal pericardial cells", long="Parietal pericardial cells"
    ),
    45: InstanceName(short="Red blood cells", long="Red blood cells"),
    46: InstanceName(short="White blood cells", long="White blood cells"),
    47: InstanceName(short="Peroxisome membrane", long="Peroxisome membrane"),
    48: InstanceName(short="Peroxisome lumen", long="Peroxisome lumen"),
}

Possibility = Literal["unknown", "absent"]


class SemanticSegmentation(StrictBase):
    """
    Metadata for a semantic segmentation, i.e. a segmentation where unique
    numerical values represent separate semantic classes.

    Attributes
    ----------

    type: Literal["semantic_segmentation"]
        Must be the string 'semantic_segmentation'.
    encoding: Dict[Union[Possibility, Literal["present"]], int]
        This dict represents the mapping from possibilities to numeric values. The keys
        must be strings in the set `{'unknown', 'absent', 'present'}`, and the values
        must be numeric values contained in the array described by this metadata.

        For example, if an annotator produces an array where 0 represents `unknown` and
        1 represents the presence of class X then `encoding` would take the value
        `{'unknown': 0, 'present': 1}`

    """

    type: Literal["semantic_segmentation"] = "semantic_segmentation"
    encoding: Dict[Union[Possibility, Literal["present"]], int]


class InstanceSegmentation(StrictBase):
    """
    Metadata for instance segmentation, i.e. a segmentation where unique numerical 
    values represent distinct occurrences of the same semantic class.

    Attributes
    ----------

    type: Literal["instance_segmentation"]
        Must be the string "instance_segmentation"
    encoding: Dict[Possibility, int]
        This dict represents the mapping from possibilities to numeric values. The keys
        must be strings from the set `{'unknown', 'absent'}`, and the values
        must be numeric values contained in the array described by this metadata.

        For example, if an annotator produces an array where 0 represents 'unknown' and
        the values 1...N represent instances of some class, then `encoding` would take 
        the value {'unknown': 0}. The meaning of the non-zero values (i.e., that they 
        represent distinct instances of a class) can be inferred from the fact that 
        this is instance segmentation, and thus these values do not appear as keys in 
        `encoding`.
    """
    type: Literal["instance_segmentation"] = "instance_segmentation"
    encoding: Dict[Possibility, int]


AnnotationType = Union[SemanticSegmentation, InstanceSegmentation]

TName = TypeVar("TName", bound=str)

class AnnotationArrayAttrs(GenericModel, Generic[TName]):
    """
    The metadata for an array of annotated values.

    Attributes
    ----------

    class_name: str
        The name of the semantic class annotated in this array.
    histogram: Dict[Possibility, int]
        The frequency of 'absent' and / or 'missing' values in the array data.
        The total number of elements in the array that represent "positive" examples can
        be calculated from this histogram -- take the number of elements in the array
        minus the sum of the values in the histogram.
    annotation_type: SemanticSegmentation | InstanceSegmentation
        The type of the annotation. Must be either an instance of SemanticSegmentation
        or an instance of InstanceSegmentation.
    """

    class_name: TName
    # a mapping from values to frequencies
    histogram: Dict[Possibility, int]
    # a mapping from class names to values
    # this is array metadata because labels might disappear during downsampling
    annotation_type: AnnotationType

    @root_validator()
    def check_encoding(cls, values):
        if (typ := values.get("type", False)) and (
            hist := values.get("histogram", False)
        ):
            # check that everything in the histogram is encoded
            assert set(typ.encoding.keys()).issuperset((hist.keys())), "Oh no"

        return values


class AnnotationGroupAttrs(GenericModel, Generic[TName]):
    """
    The metadata for an individual annotated semantic class.
    In a storage hierarchy like zarr or hdf5, this metadata is associated with a
    group-like container that contains a collection of arrays that contain the
    annotation data in a multiscale representation.

    Attributes
    ----------

    class_name: str
        The name of the semantic class annotated by the data in this group.
    annotation_type: AnnotationType
        The type of annotation represented by the data in this group.
    """

    class_name: TName
    annotation_type: AnnotationType


class CropGroupAttrs(GenericModel, Generic[TName]):
    """
    The metadata for all annotations in zarr group representing a single crop.

    Attributes
    ----------
    version: str
        The version of this collection of metadata. Must be the string '0.1.0'.
    name: Optional[str]
        The name of the crop. Optional.
    description: Optional[str]
        A description of the crop. Optional.
    created_by: list[str]
        The people or entities responsible for creating the annotations in the crop. If 
        unknown, use an empty list.
    created_with: list[str]
        The tool(s) used to create the annotations in the crop. If unknown, 
        use an empty list.
    start_date: Optional[datetime.date]
        The calendar date when the crop was started. Optional.
    end_date: Optional[datetime.date]
        The calendar date when the crop was completed. Optional.
    duration_days: Optional[int]
        The number of days spent annotating the crop. Optional.
    protocol_uri: Optional[str]
        A URI pointing to a description of the annotation protocol used to produce the
        annotations. Optional.
    class_names: list[str]
        The names of the classes that are annotated in this crop. Each element from 
        `class_names` should also be the name of a Zarr group stored under the Zarr
        group that contains this metadata.
    """
    
    class Config:
        frozen = True
        validate_assignment = True
    
    version: Literal['0.1.0'] = Field('0.1.0', allow_mutation=False)
    name: Optional[str]
    description: Optional[str]
    created_by: list[str]
    created_with: list[str]
    start_date: Optional[date]
    end_date: Optional[date]
    duration_days: Optional[int]
    protocol_uri: Optional[str]
    class_names: list[TName]

class AnnotationArray(ArraySpec):
    """
    The specification of a zarr array that contains data from an annotation, e.g. a 
    semantic segmentation or an instance segmentation.

    Attributes
    ----------
    attrs : CellmapWrapper[AnnotationWrapper[AnnotationArrayAttrs]]
        Metadata describing the annotated class, which is nested
        under two outer dicts that define the namespace of this metadata, 
        i.e. `{'cellmap': {'annotation': {...}}}`.
        See [AnnotationGroupAttrs][cellmap_schemas.annotation.AnnotationArrayAttrs] for
        details of the wrapped metadata.
    """
    attrs: CellmapWrapper[AnnotationWrapper[AnnotationArrayAttrs]]

class AnnotationGroup(GroupSpec):
    """
    The specification of a multiscale group that contains a segmentation of a single 
    class.

    Attributes
    ----------
    attrs : CellmapWrapper[AnnotationWrapper[AnnotationGroupAttrs]]
        A dict describing the annotation, which is nested
        under two outer dicts that define the namespace of this metadata, 
        i.e. `{'cellmap': {'annotation': {...}}}`.
        See [AnnotationGroupAttrs][cellmap_schemas.annotation.AnnotationGroupAttrs] for 
        details of the wrapped metadata.
    """
    attrs: CellmapWrapper[AnnotationWrapper[AnnotationGroupAttrs]]
    members: Dict[str, AnnotationArray]


class CropGroup(GroupSpec):
    """
    The specification of a crop group. Conventionally, a crop is a subset of a 
    larger imaging volume that has been annotated by a human annotator. A crop may 
    contain multiple semantic classes, which might be annotated via semantic 
    segmentation or instance segmentation.
    
    Attributes
    ----------
    attrs : CellmapWrapper[AnnotationWrapper[CropGroupAttrs]]
        A dict describing all annotations contained within the group, which is nested
         under two outer dicts that define the namespace of this metadata, 
         i.e. `{'cellmap': {'annotation': {...}}}`. See 
        [CropGroupAttrs][cellmap_schemas.annotation.CropGroupAttrs] for details 
        of the structure of the metadata.
    members : Mapping[str, AnnotationGroup]
        A dict with keys that are strings and values that are instances of 
        [AnnotationGroup][cellmap_schemas.annotation.AnnotationGroup].

    """
    attrs: CellmapWrapper[AnnotationWrapper[CropGroupAttrs]]
    members: Mapping[str, AnnotationGroup]

def json_schema():
    print(CropGroup.schema_json(indent=2))