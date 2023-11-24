from typing import Sequence
from pydantic import BaseModel, PositiveInt, validator
from pydantic_zarr.core import GroupSpec, ArraySpec
import zarr


class PixelResolution(BaseModel):
	"""
	This attribute is used by N5-fluent tools like Neuroglancer to convey the
	spacing between points on a N-dimensional coordinate grid, and the unit of measure
	for the coordinate grid, for data stored in the N5 format.
	Conventionally, it is used to express the resolution, in world coordinates unit for
	an image. If this metadata is stored in the `attributes.json` file of an N5 dataset,
	then the visualization tool Neuroglancer can use this metadata to infer coordinates
	for the data in the array. If this metadata is stored in the `attributes.json` file
	of an N5 group that contains a collection of N5 datasets, then Neuroglancer will use
	this metadata to infer coordinates for that collection of datasets, assuming that
	they represent a multiscale pyramid, and provided that other required metadata is present (see `NeuroglancerN5GroupMetadata`).

	See the [Neuroglancer N5 documentation](https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/n5)
	for details.

	Indexing note:

	The `dimensions` attribute maps on to the dimensions of the array described by this
	metadata in colexicographic order, i.e., the first array index corresponds to the
	smallest stride of the stored representation of the array, and subsequent array
	indices are ordered by increasing stride size.

	Be advised that numpy uses lexicographic order for array indexing, i.e., the first
	array index in numpy corresponds to the largest stride of the stored representation
	of the array, and subsequent array indices are ordered by decreasing stride size.
	So numpy users will need to reverse the order of the `dimensions` attribute when
	mapping it to the axes of a numpy-like array.

	See documentation for this metadata [here](https://github.com/saalfeldlab/n5-viewer#readme).

	Attributes
	----------

	dimensions: Sequence[float]
	    The distance, in units given by the `unit` attribute, between samples, per axis.
	    Conventionally referred to as the "resolution" of the image.
	unit: str
	    The physical unit for the `dimensions` attribute.
	"""  # noqa: E501

	dimensions: Sequence[float]
	unit: str


# todo: validate argument lengths
class NeuroglancerN5GroupMetadata(BaseModel):
	"""
	Metadata to enable displaying an N5 group containing several datasets
	as a multiresolution dataset in neuroglancer, based on
	[this comment](https://github.com/google/neuroglancer/issues/176#issuecomment-553027775).

	Indexing note:

	The attributes `axes`, `units`, and the elements of `scales` should be indexed
	colexicographically, i.e. the opposite order from numpy-standard order,
	which is lexicographic.

	Attributes
	----------
	axes : Sequence[str]
	    The names of the axes of the data
	units : Sequence[str]
	    The units for the axes of the data
	scales : Sequence[Sequence[PositiveInt]]
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

	axes: Sequence[str]
	units: Sequence[str]
	scales: Sequence[Sequence[PositiveInt]]
	pixelResolution: PixelResolution


class NeuroglancerN5Group(GroupSpec):
	"""
	A `GroupSpec` representing the structure of a N5 group with
	neuroglancer-compatible structure and metadata.

	Attributes
	----------

	attrs : NeuroglancerN5GroupMetadata
	    The metadata required to convey to neuroglancer that this group represents a
	    multiscale image.
	members : Dict[str, Union[GroupSpec, ArraySpec]]
	    The members of this group. Arrays must be consistent with the `scales` attribute
	    in `attrs`.

	"""

	attrs: NeuroglancerN5GroupMetadata

	@validator('members')
	def validate_members(cls, v: dict[str, ArraySpec]) -> dict[str, ArraySpec]:
		# check that the names of the arrays are s0, s1, s2, etc
		for key, spec in v.items():
			assert check_scale_level_name(key)
		assert len(set(a.dtype for a in v.values())) == 1
		return v

	@classmethod
	def from_zarr(cls, node: zarr.Group):
		"""
		Create an instance of `NeuroglancerN5Group` from a Zarr group. This method will
		raise an exception if the Zarr group is not backed by one of the N5-compatible
		stores (`zarr.N5Store`, `zarr.N5FSStore`).

		Parameters
		----------

		node: zarr.Group
		    A Zarr group with attributes and members compatible with
		    `NeuroglancerN5Group`.

		Returns
		-------

		An instance of `NeuroglancerN5Group`.
		"""
		if not isinstance(node.store, (zarr.N5FSStore, zarr.N5Store)):
			msg = (
				f'{cls.__name__} must be using an N5-compatible storage backend, '
				f'namely zarr.N5FSStore or zarr.N5Store. Got {node.store.__class__} '
				'instead'
			)
			raise ValueError(msg)
		else:
			base = GroupSpec.from_zarr(node).dict()
			return cls(**base)


def check_scale_level_name(name: str) -> bool:
	"""
	Check if the input follows the pattern `s$INT`, i.e., check if the input is
	a string that starts with "s", followed by a string representation of an integer,
	and nothing else.

	If validation passes, this function returns `True`.
	If validation fails, it returns `False`.

	Parameters
	----------

	name: str
	    The name to check.

	Returns
	-------

	`True` if `name` is valid, `False` otherwise.
	"""
	valid = True
	if name.startswith('s'):
		try:
			int(name.split('s')[-1])
		except ValueError:
			valid = False
	else:
		valid = False
	return valid