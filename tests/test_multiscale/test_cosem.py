from functools import reduce
import operator
from typing import Literal, Optional, Tuple
from pydantic import ValidationError
import pytest
from cellmap_schemas.multiscale.cosem import (
    ArrayMetadata,
    GroupMetadata,
    MultiscaleMetadata,
    STTransform,
    Group,
    Array,
    ScaleMetadata,
)
from cellmap_schemas.multiscale.neuroglancer_n5 import PixelResolution


@pytest.mark.parametrize("ndim", (2, 3, 4))
@pytest.mark.parametrize("broken_attribute", (None, "axes", "scale", "translate", "units"))
def test_sttransform_attribute_lengths(ndim: int, broken_attribute: Optional[str]):
    data = {
        "axes": list(map(str, range(ndim))),
        "scale": [1] * ndim,
        "translate": [1] * ndim,
        "units": ["nm"] * ndim,
    }
    if broken_attribute is None:
        # ensure that normal validation works correctly
        STTransform.parse_obj(data)
        return
    else:
        # chop the end off of a dimensional attribute to break validation
        data[broken_attribute] = data[broken_attribute][:-1]
        with pytest.raises(ValidationError):
            STTransform.parse_obj(data)
        return


@pytest.mark.parametrize("ndim", (2, 3, 4))
@pytest.mark.parametrize("order", ("C", "F"))
@pytest.mark.parametrize(
    "broken_attribute",
    (
        None,
        "pixelResolution.dimensions",
        "transform.axes",
        "transform.scale",
        "transform.translate",
        "transform.units",
    ),
)
def test_multiscale_array_attribute_length(
    ndim: int, order: Literal["C", "F"], broken_attribute: Optional[str]
):
    transform = STTransform(
        axes=list(map(str, range(ndim))),
        scale=[2] * ndim,
        translate=[1] * ndim,
        units=["nm"] * ndim,
        order=order,
    )
    if transform.order == "C":
        pixr = PixelResolution(dimensions=transform.scale[::-1], unit=transform.units[0])
    else:
        pixr = PixelResolution(dimensions=transform.scale, unit=transform.units[0])
    data = {"transform": transform.dict(), "pixelResolution": pixr.dict()}

    if broken_attribute is None:
        # ensure that normal validation works correctly
        ArrayMetadata.parse_obj(data)
        return

    elif broken_attribute == "pixelResolution.dimensions":
        data["pixelResolution"]["dimensions"] = data["pixelResolution"]["dimensions"][:-1]

    elif broken_attribute.startswith("transform"):
        _, attribute = broken_attribute.split(".")
        data["transform"][attribute] = data["transform"][attribute][:-1]

    with pytest.raises(ValidationError):
        ArrayMetadata.parse_obj(data)


def test_multiscale_array_consistent_units():
    transform = STTransform(
        axes=["a", "b"],
        scale=[2, 2],
        translate=[1, 1],
        units=["km", "m"],
    )
    pixr = PixelResolution(dimensions=transform.scale[::-1], unit="km")
    data = {"transform": transform.dict(), "pixelResolution": pixr.dict()}

    with pytest.raises(ValidationError):
        ArrayMetadata.parse_obj(data)


@pytest.mark.parametrize("order", ("C", "F"))
def test_multiscale_array_c_f_order(order: Literal["C", "F"]):
    transform = STTransform(
        axes=["a", "b"], scale=[2, 2], translate=[1, 1], units=["km", "m"], order=order
    )
    if order == "C":
        pixr = PixelResolution(dimensions=transform.scale, unit="km")
    else:
        pixr = PixelResolution(dimensions=transform.scale[::-1], unit="km")
    data = {"transform": transform, "pixelResolution": pixr}

    with pytest.raises(ValidationError):
        ArrayMetadata.parse_obj(data)


@pytest.mark.parametrize("shape", ((16, 16), (64, 64, 64), (128, 128, 128)))
@pytest.mark.parametrize("order", ("C", "F"))
@pytest.mark.parametrize("broken_attribute", (None, "axes", "scales", "units"))
def test_multiscale_group(
    shape: Tuple[int, ...], order: Literal["C", "F"], broken_attribute: Optional[str]
):
    shapes = tuple(tuple(s // (2**idx) for s in shape) for idx in range(3))
    scales: list[list[int]] = []
    members: dict[str, Array] = {}
    ndim = len(shape)
    for idx, shape in enumerate(shapes):
        name = f"s{idx}"
        scales.append([2**idx] * ndim)
        transform = STTransform(
            axes=list(map(str, range(ndim))),
            scale=[1.0 * (2**idx)] * ndim,
            translate=[reduce(operator.add, (0, *(2 ** (i - 1) for i in range(idx))))] * ndim,
            units=["nm"] * ndim,
            order=order,
        )
        if transform.order == "C":
            pixr = PixelResolution(dimensions=transform.scale[::-1], unit=transform.units[0])
        else:
            pixr = PixelResolution(dimensions=transform.scale, unit=transform.units[0])
        attrs = ArrayMetadata(transform=transform, pixelResolution=pixr)
        members[name] = Array(shape=shape, chunks=shape, attrs=attrs, dtype="uint8")

    # normal validation
    multiscale_meta = MultiscaleMetadata(
        datasets=[
            ScaleMetadata(path=key, transform=value.attrs.transform)
            for key, value in members.items()
        ]
    )
    if order == "C":
        axes = members["s0"].attrs.transform.axes[::-1]
    else:
        axes = members["s0"].attrs.transform.axes
    attrs = GroupMetadata(
        scales=scales,
        units=["nm"] * ndim,
        pixelResolution=members["s0"].attrs.pixelResolution,
        axes=axes,
        multiscales=[multiscale_meta],
    ).dict()
    members_dict = {key: value.dict() for key, value in members.items()}
    if broken_attribute is None:
        Group(members=members_dict, attrs=attrs)
        return
    elif broken_attribute == "axes":
        attrs["axes"] = attrs["axes"][::-1]
        with pytest.raises(ValidationError):
            Group(members=members_dict, attrs=attrs)
