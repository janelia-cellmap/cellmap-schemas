from __future__ import annotations
from typing import TYPE_CHECKING
from numcodecs import GZip, Zstd

if TYPE_CHECKING:
    from typing import Literal

import numpy as np
from pydantic import ValidationError
import pytest
from zarr.util import guess_chunks
from cellmap_schemas.multiscale.neuroglancer_n5 import GroupMetadata, PixelResolution, Group


def test_pixelresolution() -> None:
    data = {"dimensions": [1.0, 1.0, 1.0], "unit": "nm"}
    PixelResolution.model_validate(data)


@pytest.mark.parametrize("ndim", (2, 3, 4))
@pytest.mark.parametrize(
    "broken_attribute", (None, "axes", "units", "scales[0]", "pixelResolution.dimensions")
)
def test_group_metadata_attribute_lengths(ndim: int, broken_attribute: str | None) -> None:
    data = {
        "axes": list(map(str, range(ndim))),
        "units": ["nm"] * ndim,
        "scales": [[1] * ndim, [2] * ndim],
        "pixelResolution": {"dimensions": [1.0] * ndim, "unit": "nm"},
    }
    if broken_attribute is None:
        GroupMetadata.model_validate(data)
        return
    else:
        if broken_attribute == "pixelResolution.dimensions":
            data["pixelResolution"]["dimensions"] = data["pixelResolution"]["dimensions"][:-1]
        elif broken_attribute == "scales[0]":
            data["scales"][0] = data["scales"][0][:-1]
        else:
            data[broken_attribute] = data[broken_attribute][:-1]
        with pytest.raises(ValidationError):
            GroupMetadata.model_validate(data)


@pytest.mark.parametrize("ndim", (2, 3, 4))
def test_group_metadata_wrong_scale_0(ndim: int):
    """
    First element in `scales` should be all 1s
    """
    data_wrong_axes = {
        "axes": list(map(str, range(ndim))),
        "units": ["nm"] * ndim,
        "scales": [[2] * ndim, [2] * ndim],
        "pixelResolution": {"dimensions": [1.0] * ndim, "unit": "nm"},
    }
    with pytest.raises(ValidationError):
        GroupMetadata.model_validate(data_wrong_axes)


@pytest.mark.parametrize("dimension_order", ("C", "F"))
@pytest.mark.parametrize("chunks", ("auto", ((2, 2, 2))))
@pytest.mark.parametrize("compressor", (Zstd(3), GZip(-1)))
def test_from_arrays(
    dimension_order: Literal["C", "F"],
    chunks: Literal["auto"] | int | tuple[int, ...] | tuple[tuple[int, ...], ...],
    compressor: Zstd | GZip,
) -> None:
    arrays = (np.zeros((10, 10, 10)), np.zeros((5, 5, 5)), np.zeros((2, 2, 2)))
    scales = ((1.0, 1.0, 1.5), (2.0, 2.0, 2.0), (4.0, 4.0, 4.0))
    axes = ("z", "y", "x")
    paths = ("s0", "s1", "s2")
    units = ("nm", "nm", "nm")

    group = Group.from_arrays(
        arrays=arrays,
        paths=paths,
        scales=scales,
        axes=axes,
        units=units,
        compressor=compressor,
        chunks=chunks,
        dimension_order=dimension_order,
    )

    if dimension_order == "C":
        indexer = slice(-1, None, -1)
    else:
        indexer = slice(None)

    assert set(group.members.keys()) == set(paths)
    assert group.attributes.pixelResolution == PixelResolution(
        dimensions=scales[0][indexer], unit=units[indexer][0]
    )
    for idx in range(len(arrays)):
        obs = group.members[paths[idx]]
        exp = arrays[idx]
        if chunks == "auto":
            chunks_expected = guess_chunks(exp.shape, exp.dtype.itemsize)
        else:
            chunks_expected = chunks
        assert obs.chunks == chunks_expected
        assert obs.attributes.pixelResolution == PixelResolution(
            dimensions=scales[idx][indexer], unit=units[indexer][0]
        )
        assert obs.compressor == compressor.get_config()
