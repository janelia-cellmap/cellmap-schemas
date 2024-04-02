from __future__ import annotations
import zarr
from numcodecs import GZip
from pydantic_zarr.v2 import GroupSpec

from cellmap_schemas.n5_wrap import n5_array_unwrapper, n5_spec_unwrapper, n5_spec_wrapper


def test_n5_wrapping(tmpdir: str) -> None:
    n5_store = zarr.N5FSStore(str(tmpdir))
    group = zarr.group(n5_store, path="group1")
    group.attrs.put({"group": "true"})
    compressor = GZip(-1)
    arr = group.create_dataset(
        name="array", shape=(10, 10, 10), compressor=compressor, dimension_separator="."
    )
    arr.attrs.put({"array": True})

    spec_n5 = GroupSpec.from_zarr(group)
    assert spec_n5.members["array"].dimension_separator == "."

    arr_unwrapped = n5_array_unwrapper(spec_n5.members["array"])

    assert arr_unwrapped.dimension_separator == "/"
    assert arr_unwrapped.compressor == compressor.get_config()

    spec_unwrapped = n5_spec_unwrapper(spec_n5)
    assert spec_unwrapped.attributes == spec_n5.attributes
    assert spec_unwrapped.members["array"] == arr_unwrapped

    spec_wrapped = n5_spec_wrapper(spec_unwrapped)
    group2 = spec_wrapped.to_zarr(group.store, path="group2")
    assert GroupSpec.from_zarr(group2) == spec_n5

    """     
    # test with N5 metadata
    test_data = np.zeros((10, 10, 10))
    multi = multiscale(test_data, windowed_mean, (2, 2, 2))
    n5_neuroglancer_spec = NeuroglancerN5Group.from_xarrays(multi, chunks="auto")
    assert n5_spec_wrapper(n5_neuroglancer_spec) """
