from typing import Optional
from pydantic import ValidationError
import pytest
from cellmap_schemas.multiscale.neuroglancer_n5 import GroupMetadata, PixelResolution


def test_pixelresolution():
    data = {"dimensions": [1.0, 1.0, 1.0], "unit": "nm"}
    PixelResolution.model_validate(data)


@pytest.mark.parametrize("ndim", (2, 3, 4))
@pytest.mark.parametrize(
    "broken_attribute", (None, "axes", "units", "scales[0]", "pixelResolution.dimensions")
)
def test_group_metadata_attribute_lengths(ndim: int, broken_attribute: Optional[str]):
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
