"""
This module contains Pydantic models for Neuroglancer-compatible multiscale images stored 
in N5 groups. See Neuroglancer's [N5-specific documentation](https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/n5) 
for details on what Neuroglancer expects when it reads N5 data.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from typing_extensions import Self, Type, Literal
    from numcodecs.abc import Codec
from pydantic import BaseModel, PositiveInt, model_validator
from pydantic_zarr.v2 import ArraySpec, GroupSpec

from cellmap_schemas.n5_wrap import N5GroupSpec
from numpy.typing import NDArray


class PixelResolution(BaseModel):
    """
    This attribute is used by N5-fluent tools like Neuroglancer to convey the
    spacing between points on a N-dimensional coordinate grid, and the unit of measure
    for the coordinate grid, for data stored in the N5 format.
    Conventionally, it is used to express the resolution and unit of measure for the axes of
    an image.

     - If this metadata is stored in the `attributes.json` file of an N5 dataset,
    then the visualization tool Neuroglancer can use this metadata to infer coordinates
    for the data in the array.

    - If this metadata is stored in the `attributes.json` file
    of an N5 group that contains a collection of N5 datasets, then Neuroglancer will use
    this metadata to infer coordinates for that collection of datasets, assuming that
    they represent a multiscale pyramid, and provided that other required metadata is present (see [`GroupMetadata`][cellmap_schemas.multiscale.neuroglancer_n5.GroupMetadata]).

    See the Neuroglancer [N5-specific documentation](https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/n5)
    for details.

    Indexing note:

    The `pixelResolution.dimensions` attribute maps on to the dimensions of the array described by this
    metadata in *colexicographic* order, i.e., the first array index corresponds to the
    smallest stride of the stored representation of the array, and subsequent array
    indices are ordered by increasing stride size. This is the array indexing order used
    by Neuroglancer, and by the N5 ecosystem of tools.

    Be advised that Numpy uses *lexicographic* order for array indexing -- the first
    array index in Numpy corresponds to the largest stride of the stored representation
    of the array, and subsequent array indices are ordered by decreasing stride size.
    So Numpy users must reverse the order of the `dimensions` attribute when
    mapping it to the axes of a Numpy-like array.

    See the external [documentation](https://github.com/saalfeldlab/n5-viewer#readme) for this metadata for more information.

    Attributes
    ----------

    dimensions: Sequence[float]
            The distance, in units given by the `unit` attribute, between samples, per axis.
            Conventionally referred to as the "resolution" of the image.
    unit: str
            The physical unit for the `dimensions` attribute.
    """  # noqa: E501

    dimensions: tuple[float, ...]
    unit: str


class ArrayMetadata(BaseModel):
    pixelResolution: PixelResolution


Array = ArraySpec[ArrayMetadata]


# todo: validate argument lengths
class GroupMetadata(BaseModel):
    """
    Metadata to enable displaying an N5 group containing several datasets
    as a multiresolution image in neuroglancer, based on
    [this comment](https://github.com/google/neuroglancer/issues/176#issuecomment-553027775).

    Indexing note:

    The attributes `axes`, `units`, and the elements of `scales` should be indexed
    colexicographically, i.e. the opposite order from Numpy-standard order,
    which is lexicographic.

    Attributes
    ----------
    axes : tuple[str]
            The names of the axes of the data
    units : tuple[str]
            The units for the axes of the data
    scales : tuple[tuple[PositiveInt]]
            The relative scales of each axis. E.g., if this metadata describes a 3D dataset
            that was downsampled isotropically by 2 along each axis over two iterations,
            resulting in 3 total arrays, then `scales` would be the list
            `[[1,1,1], [2,2,2], [4,4,4]]`.
    pixelResolution: PixelResolution
            An instance of `PixelResolution` that specifies the interval, in physical units,
            between adjacent samples, per axis. Note that `PixelResolution` also defines a
            `unit` attribute, which is redundant with the `units` attribute defined on this
            class.
    """  # noqa: E501

    axes: tuple[str, ...]
    units: tuple[str, ...]
    scales: tuple[tuple[PositiveInt, ...], ...]
    pixelResolution: PixelResolution

    @model_validator(mode="after")
    def check_dimensionality(self: "GroupMetadata"):
        if not len(self.axes) == len(self.units):
            msg = (
                f"The number of elements in `axes` ({len(self.axes)}) does not"
                f"match the number of elements in `units` ({len(self.units)})."
            )
            raise ValueError(msg)
        if not all(s == 1 for s in self.scales[0]):
            msg = f"The first element of `scales` must be all 1s. Got {self.scales[0]} instead."
            raise ValueError(msg)
        for idx, scale in tuple(enumerate(self.scales)):
            if not len(scale) == len(self.axes):
                msg = (
                    f"The number of elements in `axes` ({len(self.axes)}) does not"
                    f"match the number of elements in the {idx}th element in `scales`"
                    f"({len(self.units)})."
                )
                raise ValueError(msg)

        if not len(self.pixelResolution.dimensions) == len(self.axes):
            raise ValueError(
                f"The number of elements in `axes` ({len(self.axes)})"
                "does not match the number of elements in `pixelResolution.dimensions`"
                f"({len(self.pixelResolution.dimensions)})"
            )

        for idx, u in enumerate(self.units):
            if not u == self.pixelResolution.unit:
                msg = (
                    f"The {idx}th element of `units` ({u}) does not "
                    f"match `pixelResolution.unit` ({self.pixelResolution.unit})"
                )
                raise ValueError(msg)
        return self


class Group(N5GroupSpec):
    """
    A `GroupSpec` representing the structure of a N5 group with
    neuroglancer-compatible structure and metadata.

    Attributes
    ----------

    attributes : GroupMetadata
            The metadata required to convey to neuroglancer that this group represents a
            multiscale image.
    members : Dict[str, ArraySpec | GroupSpec]
            The members of this group. Arrays must be consistent with the `scales` attribute
            in `attributes`.

    """

    attributes: GroupMetadata
    members: dict[str, Array | GroupSpec]

    @classmethod
    def from_arrays(
        cls: Type[Self],
        *,
        arrays: Sequence[NDArray[Any]],
        paths: Sequence[str],
        axes: Sequence[str],
        scales: Sequence[Sequence[float]],
        units: Sequence[str],
        chunks: tuple[int, ...] | tuple[tuple[int, ...]] | Literal["auto"] = "auto",
        compressor: Codec | None | Literal["auto"] = "auto",
        dimension_order: Literal["C", "F"],
    ) -> Self:
        if dimension_order == "C":
            # this will flip the axes, units, and scales before writing metadata
            indexer = slice(-1, None, -1)
        else:
            # this preserves the input order of dimensional attributes
            indexer = slice(None)

        axes_parsed = tuple(axes[indexer])
        scales_parsed = tuple(tuple(s)[indexer] for s in scales)
        units_parsed = tuple(units[indexer])

        scale_ratios = tuple(
            tuple(int(a // b) for a, b in zip(scale, scales_parsed[0])) for scale in scales_parsed
        )

        attributes = GroupMetadata(
            axes=axes_parsed,
            units=units_parsed,
            scales=scale_ratios,
            pixelResolution=PixelResolution(unit=units_parsed[0], dimensions=scales_parsed[0]),
        )
        members = {
            path: Array.from_array(
                attributes=ArrayMetadata(
                    pixelResolution=PixelResolution(dimensions=scale, unit=units_parsed[0])
                ),
                array=array,
                chunks=chunks,
                compressor=compressor,
            )
            for path, array, scale in zip(paths, arrays, scales_parsed)
        }
        return cls(members=members, attributes=attributes)

    @model_validator(mode="after")
    def check_scales(self: "Group"):
        scales = self.attributes.scales
        members = self.members

        for level in range(len(scales)):
            name_expected = f"s{level}"
            if name_expected not in members:
                raise ValueError(
                    f"Expected an array called {name_expected} in `members`, but it is missing. "
                )
            if not isinstance(members[name_expected], ArraySpec):
                raise ValueError(
                    f"Expected an array called {name_expected} in `members`. Got {type(members[name_expected])} instead."
                )

        return self
