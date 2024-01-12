"""
This module contains Pydantic models for "COSEM-flavored" multiscale images. These images
are  stored in the [N5](https://github.com/saalfeldlab/n5) format, and use a layout / metadata
that's compatible with the [Neuroglancer](https://github.com/google/neuroglancer) visualization tool.

Note that the hierarchy convention modeled here will likely be superceded by conventions defined
in the [OME-NGFF](https://ngff.openmicroscopy.org/) specification.
"""

from typing import Annotated, Dict, List, Literal, Optional, Sequence, Tuple
import numpy as np
from pydantic_zarr.v2 import GroupSpec, ArraySpec
from pydantic import BaseModel, Field, model_validator
from cellmap_schemas.multiscale import neuroglancer_n5
from cellmap_schemas.multiscale.neuroglancer_n5 import PixelResolution


class STTransform(BaseModel):
    """
    Representation of an N-dimensional scaling -> translation transform for labelled
    axes with units.

    This metadata was created within COSEM/Cellmap at a time when existing
    spatial metadata conventions for images stored in chunked file formats
    could not express a name, translation, or unit per axis.

    The creation of `STTransform` metadata preceded version 0.4 of OME-NGFF,
    which can express the same information. `STTransform` metadata should not be used if
    the OME-NGFF metadata is available.

    Attributes
    ----------
    axes: Sequence[str]
        Names for the axes of the data.
    units: Sequence[str]
        Units for the axes of the data.
    translate: Sequence[float]
        The location of the origin of the data, in units given specified by the `units`
        attribute.
    scale: Sequence[float]
        The difference between adjacent coordinates of the data, in units specified by
        the `units` attribute. Note that when converting an array index into a
        coordinate, the scaling should be applied before translation.
    order:
        Defines the array indexing convention assumed by the `axes` attribute (and,
        by extension, the other attributes). `order` must be "C", which denotes
        C-ordered (lexicographic) indexing, or "F", which denotes F-ordered
        (colexicographic) indexing. This attribute exists because tools in the N5
        ecosystem enumerate dimenions in "F" order, while tools in the numpy ecosystem
        enumerate dimensions in  "C" order. This attribute allows an N5-based tool to
        express a scaling + translation in the axis order that is native to that
        ecosystem, while retaining compatibility with numpy-based tools.

        The default is "C". If `order` is missing, it should be assumed to be "C".
    """

    order: Optional[Literal["C", "F"]] = "C"
    axes: Tuple[str, ...]
    units: Tuple[str, ...]
    translate: Tuple[float, ...]
    scale: Tuple[float, ...]

    @model_validator(mode="after")
    def validate_argument_length(self: "STTransform"):
        if not len(self.axes) == len(self.units) == len(self.translate) == len(self.scale):
            raise ValueError(
                "The length of all arguments must match. "
                f"len(axes) = {len(self.axes)}, "
                f"len(units) = {len(self.units)}, "
                f"len(translate) = {len(self.translate)}, "
                f"len(scale) = {len(self.scale)}."
            )
        return self


class ArrayMetadata(BaseModel):
    """
    Metadata for an array in a multiscale group.

    Attributes
    ----------
    transform: STTransform
        A description of axis names, units, scaling, and translation for this array.
    pixelResolution: neuroglancer_n5.PixelResolution
        A description of the scaling and unit for this array. This metadata redundantly
        expresses a strict subset of the information expressed by `transform`,
        but it is necessary for a family of visualization tools (e.g., Neuroglancer).
    """

    pixelResolution: neuroglancer_n5.PixelResolution
    transform: STTransform

    @classmethod
    def from_transform(cls, transform: STTransform):
        """
        Generate an instance of ArrayMetadata from an STTtransform`
        """
        if transform.order == "C":
            pixr = PixelResolution(dimensions=transform.scale[::-1], unit=transform.units[-1])
        else:
            pixr = PixelResolution(dimensions=transform.scale, unit=transform.units[0])
        return cls(transform=transform, pixelResolution=pixr)

    @model_validator(mode="after")
    def check_dimensionality(self: "ArrayMetadata"):
        """
        Check that `pixelResolution` and `transform` are consistent.
        """

        pixr = self.pixelResolution
        tx = self.transform

        if not all(pixr.unit == u for u in tx.units):
            msg = (
                f"The `pixelResolution` and `transform` attributes are incompatible: "
                f"the `unit` attribute of `pixelResolution` ({pixr.unit}) was not "
                f"identical to every element in `transform.units` ({tx.units})."
            )
            raise ValueError(msg)

        if tx.order == "C":
            if pixr.dimensions != tx.scale[::-1]:
                msg = (
                    "The `pixelResolution` and `transform` attributes are incompatible: "
                    f"the `pixelResolution.dimensions` attribute ({pixr.dimensions}) "
                    "should match the reversed `transform.scale` attribute "
                    f"({tx.scale[::-1]})."
                )
                raise ValueError(msg)
        else:
            if pixr.dimensions != tx.scale:
                msg = (
                    "The `pixelResolution` and `transform` attributes are incompatible: "
                    f"the `pixelResolution.dimensions` attribute ({pixr.dimensions}) "
                    "should match the `transform.scale` attribute "
                    f"({tx.scale})."
                )
                raise ValueError(msg)
        return self


class ScaleMetadata(BaseModel):
    """
    Metadata for an entry in `MultiscaleMetadata.datasets`, which is group metadata that
    contains a list of references to arrays. Structurally, `ScaleMetadata` is the same
    as [`ArrayMetadata`][cellmap_schemas.multiscale.cosem.ArrayMetadata],
    but with an additional field, `path`.

    Attributes
    ----------
    transform: STTransform
        A description of axis names, units, scaling, and translation for the array
        referenced by this metadata.
    path: str
        The path to the array referenced by this metadata, relative to the group that
        contains this metadata.
    """

    transform: STTransform
    path: str


class MultiscaleMetadata(BaseModel):
    """
    Multiscale metadata used by COSEM/Cellmap for datasets published on OpenOrganelle.
    Inspired by [this discussion](https://github.com/zarr-developers/zarr-specs/issues/50).

    This metadata should be present in the attributes of an N5 group under the key
    `multiscales`.

    Attributes
    ----------
    name: Optional[str]
        A name for this multiscale group. Rarely used.
    datasets: Sequence[ScaleMetadata]
        A sequence of [`ScaleMetadata`][cellmap_schemas.multiscale.cosem.ScaleMetadata] elements
            that refer to the arrays contained inside the group bearing this metadata.
            Each element of `MultiscaleMetadata.datasets` references an array contained
            within the group that bears this metadata. These references contain
        the name of the array, under the `ScaleMeta.path` attribute, and the coordinate
        metadata for the array, under the `ScaleMeta.transform` attribute.
    """

    name: Optional[str] = None
    datasets: Sequence[ScaleMetadata]

    @classmethod
    def from_transforms(cls, transforms: Dict[str, STTransform], *, name: Optional[str] = None):
        """
        Create an instance of `MultiscaleMetadata` from a dict of `STTransform` and an optional name.
        """
        return cls(
            name=name,
            datasets=list(ScaleMetadata(path=key, transform=tx) for key, tx in transforms.items()),
        )


class GroupMetadata(neuroglancer_n5.GroupMetadata):
    """
    Multiscale metadata used by COSEM/Cellmap for multiscale datasets saved in N5 groups.

    Note that this class inherits attributes from
    [`neuroglancer_n5.GroupMetadata`][cellmap_schemas.multiscale.neuroglancer_n5.GroupMetadata].
    Those attributes are necessary to ensure that the N5 group can be displayed properly
    by the Neuroglancer visualization tool.

    Additional attributes are added by this class in to express properties of the
    multiscale group that cannot be expressed by [`neuroglancer_n5.GroupMetadata`][cellmap_schemas.multiscale.neuroglancer_n5.GroupMetadata]. However,
    this results in some redundancy, as the total metadata describes several properties
    of the data multiple times (e.g., the resolution of the images is conveyed
    redundantly, as are the axis names).

    Attributes
    ----------
    multiscales: List[MultiscaleMetadata]
        This metadata identifies the group as a multiscale group, i.e. a collection of
        images at different levels of detail.
    """

    multiscales: Annotated[List[MultiscaleMetadata], Field(..., max_length=1, min_length=1)]

    @classmethod
    def from_transforms(cls, transforms: Dict[str, STTransform], *, name: Optional[str] = None):
        """
        Create `MultiscaleMetadata` from a dict of `STTransform` and an optional name.
        """
        s0 = transforms["s0"]
        if s0.order in ("C", None):
            reorder = slice(None, None, -1)
        else:
            reorder = slice(0, None, 1)
        scales = []

        for scale in [a.scale for a in transforms.values()]:
            scales.append([s // b for s, b in zip(scale, s0.scale)])

        axes = s0.axes[reorder]
        units = s0.units[reorder]
        pixr = PixelResolution(dimensions=s0.scale[reorder], unit=units[0])

        multiscales = MultiscaleMetadata.from_transforms(transforms, name=name)

        return cls(
            pixelResolution=pixr, axes=axes, units=units, multiscales=[multiscales], scales=scales
        )


class Array(ArraySpec):
    """
    The metadata for a single scale level of a multiscale group.

    Attributes
    ----------
    attributes: ArrayMetadata

    """

    attributes: ArrayMetadata

    @model_validator(mode="after")
    def check_consistent_transform(self: "Array"):
        """
        Check that the spatial metadata in the attributes of this array are consistent
        with the properties of the array.
        """

        if not len(self.shape) == len(self.attributes.transform.axes):
            msg = (
                "The `shape` and `attributes.transform` attributes are incompatible: "
                f"The length of `shape` ({len(self.shape)}) must match the length of "
                f"`transform.axes` ({len(self.attributes.transform.axes)}) "
            )
            raise ValueError(msg)

        return self


class Group(neuroglancer_n5.Group):
    """
    A model of a multiscale N5 group used by COSEM/Cellmap for data presented on
    OpenOrganelle.

    Attributes
    ----------
    attributes: GroupMetadata
        Metadata that conveys that this is a multiscale group, and the coordinate
            information of the arrays it contains.
    members: dict[str, MultiscaleArray]
        The members of this group must be instances of
            [`MultiscaleArray`][cellmap_schemas.multiscale.cosem.Array]

    """

    attributes: GroupMetadata
    members: dict[str, Array]

    @model_validator(mode="after")
    def check_arrays_consistent(self: "Group"):
        """
        Check that the arrays referenced by `GroupMetadata` are consist with the
        arrays in `members`.
        """

        axes = self.attributes.axes
        multiscales = self.attributes.multiscales[0]
        members = self.members
        for idx, element in enumerate(multiscales.datasets):
            if element.path not in members:
                msg = (
                    "The `attributes` and `members` attributes are incompatible: "
                    f"`attributes.multiscales[0].datasets[{idx}].path` refers to an array "
                    f"named {element.path} that does not exist in `members`."
                )
                raise ValueError(msg)
            else:
                if isinstance(members[element.path], GroupSpec):
                    msg = (
                        "The `attributes` and `members` attributes are incompatible: "
                        f"`attributes.multiscales[0].datasets[{idx}].path` refers to an array "
                        f"named {element.path}, but `members[{element.path}]` "
                        "describes a group."
                    )
                    raise ValueError(msg)
                else:
                    # check that the array has a transform that matches the one in
                    # multiscale metadata
                    member_array: Array = members[element.path]
                    if member_array.attributes.transform != element.transform:
                        msg = (
                            "The `attributes` and `members` attributes are incompatible: "
                            f"`attributes.multiscales[0].datasets[{idx}].transform`: "
                            f"({member_array.attributes.transform}) "
                            "does not match the `attributes.transform` attribute of the "
                            f"correspdonding array described in members[{element.path}]: "
                            f"({element.transform})"
                        )
                        raise ValueError(msg)
                    if element.transform.order == "F":
                        if element.transform.axes != axes:
                            msg = (
                                "The `attributes` and `members` attributes are incompatible: "
                                f"`attributes.multiscales[0].datasets[{idx}].transform.axes`, "
                                f"indexed according to `attributes.multiscales[0].datasets[{idx}].transform.order` ({element.transform.order}) "
                                f"is {element.transform.axes},  which does not match `attributes.axes` ({axes})."
                            )
                            raise ValueError(msg)
                    else:
                        if element.transform.axes != axes[::-1]:
                            msg = (
                                "The `attributes` and `members` attributes are incompatible: "
                                f"`attributes.multiscales[0].datasets[{idx}].transform.axes`, "
                                f"indexed according to `attributes.multiscales[0].datasets[{idx}].transform.order` ({element.transform.order}) "
                                f"is {element.transform.axes[::-1]},  which does not match `attributes.axes` ({axes})."
                            )
                            raise ValueError(msg)

        return self

    @classmethod
    def from_arrays(cls, arrays: Dict[str, Array], *, name: Optional[str] = None):
        """
        Create a `Group` from a dict of `Array` instances
        """

        metadata = GroupMetadata.from_transforms(
            {key: array.attributes.transform for key, array in arrays.items()}, name=name
        )
        return cls(attributes=metadata, members=arrays)


def new_sttransform(
    meta: STTransform,
    *,
    scale: Optional[Tuple[float, ...]] = None,
    axes: Optional[Tuple[str, ...]] = None,
    order: Optional[Literal["C", "F"]] = None,
    translate: Optional[Tuple[float, ...]] = None,
    units: Optional[Tuple[str, ...]] = None,
) -> STTransform:
    if scale is None:
        scale = meta.scale
    if axes is None:
        axes = meta.axes
    if order is None:
        order = meta.order
    if translate is None:
        translate = meta.translate
    if units is None:
        units = meta.units

    return STTransform(scale=scale, order=order, axes=axes, translate=translate, units=units)


def new_multiscalemetadata(
    meta: MultiscaleMetadata,
    *,
    scale: Optional[Tuple[float, ...]] = None,
    axes: Optional[Tuple[str, ...]] = None,
    order: Optional[Literal["C", "F"]] = None,
    translate: Optional[Tuple[float, ...]] = None,
    units: Optional[Tuple[str, ...]] = None,
) -> MultiscaleMetadata:
    new_datasets = []
    for dataset in meta.datasets:
        new_transform = new_sttransform(
            dataset.transform, scale=scale, axes=axes, order=order, translate=translate, units=units
        )
        new_datasets.append(ScaleMetadata(path=dataset.path, transform=new_transform))

    return MultiscaleMetadata(name=meta.name, datasets=new_datasets)


def new_groupmetadata(
    meta: GroupMetadata,
    *,
    scale: Optional[Tuple[float, ...]] = None,
    axes: Optional[Tuple[str, ...]] = None,
    order: Optional[Literal["C", "F"]] = None,
    translate: Optional[Tuple[float, ...]] = None,
    units: Optional[Tuple[str, ...]] = None,
) -> GroupMetadata:
    new_multiscales = [
        new_multiscalemetadata(
            meta=meta.multiscales[0],
            scale=scale,
            axes=axes,
            order=order,
            translate=translate,
            units=units,
        )
        for m in meta.multiscales
    ]

    if axes is None:
        axes = meta.axes
    else:
        if order == "C" or order is None:
            axes = axes[::-1]

    if units is None:
        units = meta.units
    else:
        if order == "C" or order is None:
            # reverse the order, because the top-level units attribute is F-ordered
            units = units[::-1]

    if scale is None:
        pixelResolution = meta.pixelResolution
    else:
        if order == "C" or order is None:
            # reverse the order, because the top-level units attribute is F-ordered
            pixelResolution = PixelResolution(unit=units[0], dimensions=scale[::-1])
        else:
            pixelResolution = PixelResolution(unit=units[0], dimensions=scale)

    return GroupMetadata(
        axes=axes,
        units=units,
        scales=meta.scales,
        pixelResolution=pixelResolution,
        multiscales=new_multiscales,
    )


def change_coordinates(
    group: Group,
    *,
    scale: Optional[Tuple[float, ...]] = None,
    axes: Optional[Tuple[str, ...]] = None,
    order: Optional[Literal["C", "F"]] = None,
    translate: Optional[Tuple[float, ...]] = None,
    units: Optional[Tuple[str, ...]] = None,
):
    """
    Return a Group with new coordinates, i.e., new scale, axes, order, translate, or units. If any of these
    paramters are set to None (the default), then the old values are used.

    The `units`, `axes`, and `order` attributes can be changed globally.
    The `scale` and `translation` changes will be applied to the largest image, then a scaling / translation vector will be calculated to express this change, and that
    vector will be multiplied / added to the `scale` / `translation` attributes of all the other images.
    """
    # sort the group members by array size
    members_sorted = {
        key: array
        for key, array in sorted(
            group.members.items(), reverse=True, key=lambda v: np.prod(v[1].shape)
        )
    }
    base_member = tuple(members_sorted.values())[0]
    base_transform = base_member.attributes.transform
    new_base_transform = new_sttransform(
        base_transform, scale=scale, axes=axes, order=order, translate=translate, units=units
    )
    # if we started at 4 and end up at 8, then the scale_diff should be 2
    scale_diff = [
        a / b for a, b in zip(new_base_transform.scale, base_transform.scale, strict=True)
    ]
    # if we started at 0 and we are moved to 10, then the translation_diff should be +10
    translate_diff = [
        a - b for a, b in zip(new_base_transform.translate, base_transform.translate, strict=True)
    ]
    new_transforms = {}

    for key, member in members_sorted.items():
        old_tx = member.attributes.transform
        new_scale = [a * b for a, b in zip(old_tx.scale, scale_diff, strict=True)]
        new_trans = [a + b for a, b in zip(old_tx.translate, translate_diff, strict=True)]
        new_transforms[key] = new_sttransform(
            old_tx, scale=new_scale, translate=new_trans, order=order, units=units, axes=axes
        )

    new_groupmeta = GroupMetadata.from_transforms(
        new_transforms, name=group.attributes.multiscales[0].name
    )
    new_members = {
        key: member.model_copy(
            deep=True, update={"attributes": ArrayMetadata.from_transform(new_transforms[key])}
        )
        for key, member in members_sorted.items()
    }
    new_group = Group(attributes=new_groupmeta, members=new_members)
    return new_group
