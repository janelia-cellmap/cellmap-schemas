"""
The hierarchy described in this module exists to facilitate training machine learning models with manually curated 
contiguous subsets of raw data, called "crops". Conventionally, crops are annotated densely, resulting in images where 
each sample of the image has been given a semantic label. Crops may contain many separate label values. This "dense" 
representation is convenient when generating annotations, but the process of training machine learning models sometimes 
benefits from a more sparse representation, e.g. one where the values for each semantic class are stored in separate arrays.

This module defines a convention for representing a dense crop as a collection of multiscale images. Each multiscale image
should comply with the version 0.4 of the OME-NGFF specification.
"""
from __future__ import annotations
from datetime import date
from enum import Enum
from typing import (
    Any,
    Generic,
    List,
    Literal,
    Mapping,
    Optional,
    TypeVar,
    Union,
)
from pydantic_zarr.v2 import GroupSpec, ArraySpec
from pydantic import BaseModel, model_validator, field_serializer
import zarr

T = TypeVar("T")


class CellmapWrapper(BaseModel, Generic[T]):
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

    print(CellmapWrapper[Foo](cellmap={'bar': 10}).model_dump())
    # {'cellmap': {'bar': 10}}
    ```

    """

    cellmap: T


class AnnotationWrapper(BaseModel, Generic[T]):
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

    print(AnnotationWrapper[Foo](annotation={'bar': 10}).model_dump())
    # {'annotation': {'bar': 10}}
    ```
    """

    annotation: T


def wrap_attributes(attributes: T) -> CellmapWrapper[AnnotationWrapper[T]]:
    return CellmapWrapper(cellmap=AnnotationWrapper(annotation=attributes))


class InstanceName(BaseModel, extra="forbid"):
    long: str
    short: str


class Annotated(str, Enum):
    dense: str = "dense"
    sparse: str = "sparse"
    empty: str = "empty"


class AnnotationState(BaseModel, extra="forbid"):
    present: bool
    annotated: Annotated


class Label(BaseModel, extra="forbid"):
    value: int
    name: InstanceName
    annotationState: AnnotationState
    count: Optional[int]


class LabelList(BaseModel, extra="forbid"):
    labels: List[Label]
    annotation_type: AnnotationType = "semantic"


classNamedict = {
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
    18: InstanceName(short="ERES membrane", long="Endoplasmic reticulum exit site membrane"),
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
    44: InstanceName(short="Parietal pericardial cells", long="Parietal pericardial cells"),
    45: InstanceName(short="Red blood cells", long="Red blood cells"),
    46: InstanceName(short="White blood cells", long="White blood cells"),
    47: InstanceName(short="Peroxisome membrane", long="Peroxisome membrane"),
    48: InstanceName(short="Peroxisome lumen", long="Peroxisome lumen"),
}

Possibility = Literal["unknown", "absent"]


class SemanticSegmentation(BaseModel, extra="forbid"):
    """
    Metadata for a semantic segmentation, i.e. a segmentation where unique
    numerical values represent separate semantic classes.

    Attributes
    ----------

    type: Literal["semantic_segmentation"]
        Must be the string 'semantic_segmentation'.
    encoding: dict[Union[Possibility, Literal["present"]], int]
        This dict represents the mapping from possibilities to numeric values. The keys
        must be strings in the set `{'unknown', 'absent', 'present'}`, and the values
        must be numeric values contained in the array described by this metadata.

        For example, if an annotator produces an array where 0 represents `unknown` and
        1 represents the presence of class X then `encoding` would take the value
        `{'unknown': 0, 'present': 1}`

    """

    type: Literal["semantic_segmentation"] = "semantic_segmentation"
    encoding: dict[Union[Possibility, Literal["present"]], int]


class InstanceSegmentation(BaseModel, extra="forbid"):
    """
    Metadata for instance segmentation, i.e. a segmentation where unique numerical
    values represent distinct occurrences of the same semantic class.

    Attributes
    ----------

    type: Literal["instance_segmentation"]
        Must be the string "instance_segmentation"
    encoding: dict[Possibility, int]
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
    encoding: dict[Possibility, int]


AnnotationType = Union[SemanticSegmentation, InstanceSegmentation]

TName = TypeVar("TName", bound=str)


class AnnotationArrayAttrs(BaseModel, Generic[TName]):
    """
    The metadata for an array of annotated values.

    Attributes
    ----------

    class_name: str
        The name of the semantic class annotated in this array.
    complement_counts: Optional[dict[Possibility, int]]
        The frequency of 'absent' and / or 'missing' values in the array data.
        The total number of elements in the array that represent "positive" examples can
        be calculated from these counts -- take the number of elements in the array
        minus the sum of the values in this partial histogram.
    annotation_type: SemanticSegmentation | InstanceSegmentation
        The type of the annotation. Must be either an instance of SemanticSegmentation
        or an instance of InstanceSegmentation.
    """

    class_name: TName
    # a mapping from values to frequencies
    complement_counts: Optional[dict[Possibility, int]]
    # a mapping from class names to values
    # this is array metadata because labels might disappear during downsampling
    annotation_type: AnnotationType

    @model_validator(mode="after")
    def check_encoding(self: "AnnotationArrayAttrs"):
        assert set(self.annotation_type.encoding.keys()).issuperset((self.complement_counts.keys()))
        return self


class AnnotationGroupAttrs(BaseModel, Generic[TName]):
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


def serialize_date(value: date) -> str:
    return value.isoformat()


class CropGroupAttrs(BaseModel, Generic[TName], validate_assignment=True):
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
    class_names: list[TName]
        The names of the classes that are annotated in this crop. Each element from
        `class_names` should also be the name of a Zarr group stored under the Zarr
        group that contains this metadata.
    """

    version: Literal["0.1.1"] = "0.1.1"
    name: Optional[str]
    description: Optional[str]
    created_by: list[str]
    created_with: list[str]
    start_date: Optional[date]
    end_date: Optional[date]
    duration_days: Optional[int]
    protocol_uri: Optional[str]
    class_names: list[TName]

    @field_serializer("start_date")
    def ser_end_date(value: date):
        return serialize_date(value)

    @field_serializer("end_date")
    def ser_start_date(value: date):
        return serialize_date(value)


class AnnotationArray(ArraySpec):
    """
    The specification of a zarr array that contains data from an annotation, e.g. a
    semantic segmentation or an instance segmentation.

    Attributes
    ----------
    attributes : CellmapWrapper[AnnotationWrapper[AnnotationArrayAttrs]]
        Metadata describing the annotated class, which is nested
        under two outer dicts that define the namespace of this metadata,
        i.e. `{'cellmap': {'annotation': {...}}}`.
        See [AnnotationGroupAttrs][cellmap_schemas.annotation.AnnotationArrayAttrs] for
        details of the wrapped metadata.
    """

    attributes: CellmapWrapper[AnnotationWrapper[AnnotationArrayAttrs]]


class AnnotationGroup(GroupSpec):
    """
    The specification of a multiscale group that contains a segmentation of a single
    class.

    Attributes
    ----------
    attributes : CellmapWrapper[AnnotationWrapper[AnnotationGroupAttrs]]
        A dict describing the annotation, which is nested
        under two outer dicts that define the namespace of this metadata,
        i.e. `{'cellmap': {'annotation': {...}}}`.
        See [AnnotationGroupAttrs][cellmap_schemas.annotation.AnnotationGroupAttrs] for
        details of the wrapped metadata.
    """

    attributes: CellmapWrapper[AnnotationWrapper[AnnotationGroupAttrs]]
    members: dict[str, AnnotationArray]


class CropGroup(GroupSpec):
    """
    The specification of a crop group. Conventionally, a crop is a subset of a
    larger imaging volume that has been annotated by a human annotator. A crop may
    contain multiple semantic classes, which might be annotated via semantic
    segmentation or instance segmentation.

    Attributes
    ----------
    attributes : CellmapWrapper[AnnotationWrapper[CropGroupAttrs]]
        A dict describing all annotations contained within the group, which is nested
         under two outer dicts that define the namespace of this metadata,
         i.e. `{'cellmap': {'annotation': {...}}}`. See
        [CropGroupAttrs][cellmap_schemas.annotation.CropGroupAttrs] for details
        of the structure of the metadata.
    members : Mapping[str, AnnotationGroup]
        A dict with keys that are strings and values that are instances of
        [AnnotationGroup][cellmap_schemas.annotation.AnnotationGroup].

    """

    attributes: CellmapWrapper[AnnotationWrapper[CropGroupAttrs]]
    members: Mapping[str, AnnotationGroup]

    @classmethod
    def from_zarr(cls, group: zarr.Group):
        """
        Generate a `CropGroup` from a `zarr.Group`. Sub-groups that do not pass validation as
        `AnnotationGroup` will be ignored, as will sub-arrays.

        Parameters
        ----------
        group : zarr.Group
            A Zarr group that implements the `CropGroup` layout.

        Returns
        -------
        CropGroup

        """

        untyped_group = GroupSpec[CellmapWrapper[AnnotationWrapper[CropGroupAttrs]], Any].from_zarr(
            group
        )
        class_names = untyped_group.attributes.cellmap.annotation.class_names
        keep = {}
        for name in class_names:
            try:
                keep[name] = untyped_group.members[name]
            except KeyError as e:
                msg = (
                    f"Expected to find a group named {name} in {group.store}://{group.name}. "
                    "No group was found with that name."
                )
                raise ValueError(msg) from e
        return cls(attributes=untyped_group.attributes, members=keep)
