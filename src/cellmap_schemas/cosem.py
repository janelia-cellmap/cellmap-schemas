from typing import Dict, Literal, Optional, Sequence, Union
from pydantic_zarr import GroupSpec, ArraySpec
from pydantic import BaseModel, root_validator
from cellmap_schemas.neuroglancer_n5 import NeuroglancerN5GroupMetadata, PixelResolution


class STTransform(BaseModel):
	"""
	Representation of an N-dimensional scaling -> translation transform for labelled
	axes with units.

	This metadata was created within Cellmap (then called COSEM) because existing
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

	order: Optional[Literal['C', 'F']] = 'C'
	axes: Sequence[str]
	units: Sequence[str]
	translate: Sequence[float]
	scale: Sequence[float]

	@root_validator
	def validate_argument_length(cls, values: Dict[str, Union[Sequence[str], Sequence[float]]]):
		scale = values.get('scale')
		axes = values.get('axes')
		units = values.get('units')
		translate = values.get('translate')
		if not len(axes) == len(units) == len(translate) == len(scale):
			msg = (
				'The length of all arguments must match. '
				f'len(axes) = {len(axes)}, '
				f'len(units) = {len(units)}, '
				f'len(translate) = {len(translate)}, '
				f'len(scale) = {len(scale)}.'
			)
			raise ValueError(msg)
		return values


class ArrayMetadata(BaseModel):
	"""
	Metadata for an array in a multiscale group.

	Attributes
	----------
	transform: STTransform
	    A description of axis names, units, scaling, and translation for this array.
	pixelResolution: PixelResolution
	    A description of the scaling and unit for this array. This metadata redundantly
	    expresses a strict subset of the information expressed by `transform`,
	    but it is necessary for a family of visualization tools (e.g., Neuroglancer).
	"""

	pixelResolution: PixelResolution
	transform: STTransform


class ScaleMetadata(BaseModel):
	"""
	Metadata for an entry in `MultiscaleMetadata.datasets`, which is group metadata that
	contains a list of references to arrays. Structurally, `ScaleMetadata` is the same
	as `ArrayMetadata`, but with an additional field, `path`.

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
	Multiscale metadata used by Cellmap for datasets published on OpenOrganelle.
	Inspired by this discussion: https://github.com/zarr-developers/zarr-specs/issues/50

	Attributes
	----------

	name: Optional[str]
	    A name for this multiscale group. Rarely used.
	datasets: Sequence[ScaleMetadata]
	    A sequence of `ScaleMetadata` elements that refer to the arrays contained inside
	    the group bearing this metadata. Each element of `datasets` references an array
	    contained within the group that bears this metadata. These references contain
	    the name of the array, under the `ScaleMeta.path` attribute, and the coordinate
	    metadata for the array, under the `ScaleMeta.transform` attribute.
	"""

	name: Optional[str]
	datasets: Sequence[ScaleMetadata]


class GroupMetadata(NeuroglancerN5GroupMetadata):
	"""
	Multiscale metadata used by Cellmap for multiscale datasets saved in N5 groups.

	Note that this class inherits attributes from `NeuroglancerN5GroupMetadata`.
	Those attributes are necessary to ensure that the N5 group can be displayed properly
	by the Neuroglancer visualization tool.

	Additional attributes are added by this class in to express properties of the
	multiscale group that cannot be expressed by `NeuroglancerN5Metadata`. However,
	this results in some redundancy, as the total metadata describes several properties
	of the data multiple times (e.g., the resolution of the images is conveyed
	redundantly, as are the axis names).

	Attributes
	----------
	multiscales: List[MultiscaleMetadata]
	    This metadata identifies the group as a multiscale group, i.e. a collection of
	    images at different levels of detail.
	"""

	multiscales: list[MultiscaleMetadata]


class MultiscaleArray(ArraySpec):
	"""
	The metadata for a single scale level of a multiscale group.

	Attributes
	----------
	attrs: ArrayMetadata

	"""

	attrs: ArrayMetadata


class COSEMMultiscaleGroup(GroupSpec):
	"""
	A model of a multiscale N5 group used by COSEM/Cellmap for data on OpenOrganelle.

	Attributes
	----------


	"""

	attrs: GroupMetadata
	members: dict[str, MultiscaleArray]
