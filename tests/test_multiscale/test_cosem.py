from __future__ import annotations
from typing import TYPE_CHECKING

from cellmap_schemas.base import normalize_chunks

if TYPE_CHECKING:
    from typing import Literal

from numcodecs import GZip, Zstd
from functools import reduce
import operator
import numpy as np
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
    change_coordinates,
)
from cellmap_schemas.multiscale.neuroglancer_n5 import PixelResolution


@pytest.mark.parametrize("ndim", (2, 3, 4))
@pytest.mark.parametrize("broken_attribute", (None, "axes", "scale", "translate", "units"))
def test_sttransform_attribute_lengths(ndim: int, broken_attribute: str | None):
    data = {
        "axes": list(map(str, range(ndim))),
        "scale": [1] * ndim,
        "translate": [1] * ndim,
        "units": ["nm"] * ndim,
    }
    if broken_attribute is None:
        # ensure that normal validation works correctly
        STTransform.model_validate(data)
        return
    else:
        # chop the end off of a dimensional attribute to break validation
        data[broken_attribute] = data[broken_attribute][:-1]
        with pytest.raises(ValidationError):
            STTransform.model_validate(data)
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
    ndim: int, order: Literal["C", "F"], broken_attribute: str | None
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
    data = {"transform": transform.model_dump(), "pixelResolution": pixr.model_dump()}

    if broken_attribute is None:
        # ensure that normal validation works correctly
        ArrayMetadata.model_validate(data)
        return

    elif broken_attribute == "pixelResolution.dimensions":
        data["pixelResolution"]["dimensions"] = data["pixelResolution"]["dimensions"][:-1]

    elif broken_attribute.startswith("transform"):
        _, attribute = broken_attribute.split(".")
        data["transform"][attribute] = data["transform"][attribute][:-1]

    with pytest.raises(ValidationError):
        ArrayMetadata.model_validate(data)


def test_multiscale_array_consistent_units():
    transform = STTransform(
        axes=["a", "b"],
        scale=[2, 2],
        translate=[1, 1],
        units=["km", "m"],
    )
    pixr = PixelResolution(dimensions=transform.scale[::-1], unit="km")
    data = {"transform": transform.model_dump(), "pixelResolution": pixr.model_dump()}

    with pytest.raises(ValidationError):
        ArrayMetadata.model_validate(data)


@pytest.mark.parametrize("order", ["C", "F"])
def test_array_meta_from_transform(order: Literal["C", "F"]):
    tx = STTransform(
        order=order, scale=(3, 2, 1), translate=(1,) * 3, units=("nm",) * 3, axes=("z", "y", "x")
    )
    if order == "C":
        expected = ArrayMetadata(
            pixelResolution=PixelResolution(unit=tx.units[0], dimensions=tx.scale[::-1]),
            transform=tx,
        )
    else:
        expected = ArrayMetadata(
            pixelResolution=PixelResolution(unit=tx.units[-1], dimensions=tx.scale), transform=tx
        )
    assert ArrayMetadata.from_transform(tx) == expected


@pytest.mark.parametrize("name", [None, "foo"])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("scale", [(1, 2, 3), (2, 4, 6)])
@pytest.mark.parametrize("axes", [("a", "b", "c"), ("z", "y", "x")])
def test_multiscale_meta_from_transforms(
    name: str | None, order: Literal["C", "F"], scale: tuple[int, ...], axes: tuple[str, ...]
):
    transforms = {
        "s0": STTransform(
            order=order, scale=scale, translate=(1,) * 3, units=("nm",) * 3, axes=axes
        ),
        "s1": STTransform(
            order=order,
            scale=tuple(2 * s for s in scale),
            translate=(1.5,) * 3,
            units=("nm",) * 3,
            axes=axes,
        ),
    }

    expected = MultiscaleMetadata(
        name=name,
        datasets=tuple(ScaleMetadata(path=key, transform=tx) for key, tx in transforms.items()),
    )
    assert MultiscaleMetadata.from_transforms(transforms=transforms, name=name) == expected


@pytest.mark.parametrize("name", [None, "bar"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_groupmetadata_from_transforms(order: Literal["C", "F"], name: str | None):
    units = ("nm",) * 3
    axes = ("z", "y", "x")
    scale = (3, 2, 1)

    if order == "C":
        reorder = slice(None, None, -1)
    else:
        reorder = slice(0, None, 1)

    transforms = {}
    transforms["s0"] = STTransform(
        order=order, scale=scale, translate=(1,) * 3, units=("nm",) * 3, axes=axes
    )
    transforms["s1"] = transforms["s0"].model_copy(
        deep=True,
        update={
            "scale": list(2 * s for s in scale),
            "translate": [
                1.5,
            ]
            * 3,
        },
    )

    expected = GroupMetadata(
        axes=axes[reorder],
        scales=[[1, 1, 1], [2, 2, 2]],
        units=units[reorder],
        pixelResolution=PixelResolution(unit=units[reorder][0], dimensions=scale[reorder]),
        multiscales=[MultiscaleMetadata.from_transforms(transforms=transforms, name=name)],
    )

    observed = GroupMetadata.from_transforms(transforms, name=name)
    assert expected == observed


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
        ArrayMetadata.model_validate(data)


@pytest.mark.parametrize("shape", ((16, 16), (64, 64, 64), (128, 128, 128)))
@pytest.mark.parametrize("order", ("C", "F"))
@pytest.mark.parametrize("broken_attribute", (None, "axes", "scales", "units"))
def test_multiscale_group(
    shape: tuple[int, ...], order: Literal["C", "F"], broken_attribute: str | None
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
        members[name] = Array(shape=shape, chunks=shape, attributes=attrs, dtype="uint8")

    # normal validation
    multiscale_meta = MultiscaleMetadata(
        datasets=[
            ScaleMetadata(path=key, transform=value.attributes.transform)
            for key, value in members.items()
        ]
    )
    if order == "C":
        axes = members["s0"].attributes.transform.axes[::-1]
    else:
        axes = members["s0"].attributes.transform.axes
    attrs = GroupMetadata(
        scales=scales,
        units=["nm"] * ndim,
        pixelResolution=members["s0"].attributes.pixelResolution,
        axes=axes,
        multiscales=[multiscale_meta],
    ).model_dump()
    members_dict = {key: value.model_dump() for key, value in members.items()}
    if broken_attribute is None:
        Group(members=members_dict, attributes=attrs)
        return
    elif broken_attribute == "axes":
        attrs["axes"] = attrs["axes"][::-1]
        with pytest.raises(ValidationError):
            Group(members=members_dict, attributes=attrs)


@pytest.mark.parametrize("name", [None, "bar"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_multiscale_group_from_arrays(order: Literal["C", "F"], name: str | None):
    units = ("nm",) * 3
    axes = ("z", "y", "x")
    scale = (3, 2, 1)

    if order == "C":
        reorder = slice(None, None, -1)
    else:
        reorder = slice(0, None, 1)

    transforms = {
        "s0": STTransform(order=order, scale=scale, translate=(1,) * 3, units=units, axes=axes),
        "s1": STTransform(
            order=order,
            scale=tuple(2 * s for s in scale),
            translate=(1.5,) * 3,
            units=units,
            axes=axes,
        ),
    }

    arrays = {"s0": np.zeros((10, 10, 10)), "s1": np.zeros((5, 5, 5))}

    groupMeta = GroupMetadata(
        axes=axes[reorder],
        scales=[[1, 1, 1], [2, 2, 2]],
        units=units[reorder],
        pixelResolution=PixelResolution(unit=units[reorder][0], dimensions=scale[reorder]),
        multiscales=[MultiscaleMetadata.from_transforms(transforms=transforms)],
    )

    expected = Group(
        attributes=groupMeta,
        members={
            key: Array.from_array(
                array=array, attributes=ArrayMetadata.from_transform(transform=transforms[key])
            )
            for key, array in arrays.items()
        },
    )
    observed = Group.from_arrays(
        arrays=arrays.values(), paths=arrays.keys(), transforms=transforms.values()
    )
    assert expected == observed


@pytest.mark.parametrize("axes", [None, ("z", "y", "x")])
@pytest.mark.parametrize(
    "translate",
    [
        None,
        (0, 1, 2),
    ],
)
@pytest.mark.parametrize(
    "scale",
    [
        None,
        (1, 2, 3),
    ],
)
@pytest.mark.parametrize("units", [None, ("m", "m", "m")])
@pytest.mark.parametrize("order", [None, "C", "F"])
def test_change_coordinates_3d(
    axes: str | None,
    translate: tuple[int, ...] | None,
    scale: tuple[int, ...] | None,
    units: tuple[str, ...] | None,
    order: Literal["C", "F"] | None,
):
    base_units_c = ("micron", "micron", "micron")
    base_axes_c = ("a", "b", "c")
    base_scale_c = (4, 5, 6)
    base_translate_c = (-1, -2, -3)
    base_order = "C"

    if axes is None:
        axes_expected = base_axes_c
    else:
        axes_expected = axes

    if scale is None:
        scale_expected = base_scale_c
    else:
        scale_expected = scale

    if translate is None:
        translate_expected = base_translate_c
    else:
        translate_expected = translate

    if units is None:
        units_expected = base_units_c
    else:
        units_expected = units

    if order is None:
        order_expected = base_order
    else:
        order_expected = order

    tx_s0_expected = STTransform(
        order=order_expected,
        scale=scale_expected,
        translate=translate_expected,
        units=units_expected,
        axes=axes_expected,
    )

    tx_s1_expected = tx_s0_expected.model_copy(
        deep=True,
        update={
            "scale": tuple(2 * x for x in tx_s0_expected.scale),
            "translate": tuple(x + 0.5 for x in tx_s0_expected.translate),
        },
    )

    transforms_expected = {"s0": tx_s0_expected, "s1": tx_s1_expected}

    arrays = {"s0": np.zeros((10, 10, 10)), "s1": np.zeros((5, 5, 5))}

    source_spec = Group.from_arrays(
        arrays=arrays.values(), paths=arrays.keys(), transforms=transforms_expected.values()
    )

    new_spec = change_coordinates(
        source_spec, axes=axes, order=order, translate=translate, scale=scale, units=units
    )
    new_attrs = new_spec.attributes

    if order in ("C", None):
        reorder = slice(None, None, -1)
    else:
        reorder = slice(0, None, 1)

    assert new_attrs.axes == axes_expected[reorder]
    assert new_attrs.pixelResolution.dimensions == scale_expected[reorder]
    assert new_attrs.pixelResolution.unit == units_expected[reorder][0]
    assert new_attrs.scales == source_spec.attributes.scales
    assert new_attrs.multiscales[0].datasets[0] == ScaleMetadata(
        path="s0", transform=transforms_expected["s0"]
    )
    assert new_attrs.multiscales[0].datasets[1] == ScaleMetadata(
        path="s1", transform=transforms_expected["s1"]
    )
    assert new_spec.members["s0"].attributes.transform == transforms_expected["s0"]
    assert new_spec.members["s1"].attributes.transform == transforms_expected["s1"]


@pytest.mark.parametrize("dimension_order", ("C", "F"))
@pytest.mark.parametrize("chunks", ("auto", (2, 2, 2), ((1, 1, 1), (2, 2, 2), (3, 3, 3))))
@pytest.mark.parametrize("compressor", (Zstd(3), GZip(-1)))
def test_from_arrays(
    dimension_order: Literal["C", "F"],
    chunks: Literal["auto"] | int | tuple[int, ...] | tuple[tuple[int, ...], ...],
    compressor: Zstd | GZip,
) -> None:
    arrays = (np.zeros((10, 10, 10)), np.zeros((5, 5, 5)), np.zeros((2, 2, 2)))
    scales = ((1.0, 1.0, 1.5), (2.0, 2.0, 2.0), (4.0, 4.0, 4.0))
    translates = ((0.0, 0.0, 1.5), (0.6, 0.6, 2.0), (3.0, 4.0, 5.0))
    axes = ("z", "y", "x")
    paths = ("s0", "s1", "s2")
    units = ("nm", "nm", "nm")
    transforms = tuple(
        STTransform(axes=axes, scale=scale, translate=translate, units=units, order=dimension_order)
        for scale, translate in zip(scales, translates)
    )

    group = Group.from_arrays(
        arrays=arrays, paths=paths, transforms=transforms, compressor=compressor, chunks=chunks
    )

    assert set(group.members.keys()) == set(paths)

    if dimension_order == "C":
        indexer = slice(-1, None, -1)
    else:
        indexer = slice(None)

    assert group.attributes.pixelResolution == PixelResolution(
        dimensions=scales[0][indexer], unit=units[indexer][0]
    )
    chunks_expected = normalize_chunks(chunks, arrays)
    for idx in range(len(arrays)):
        obs = group.members[paths[idx]]
        assert obs.chunks == chunks_expected[idx]

        assert obs.attributes.pixelResolution == PixelResolution(
            dimensions=scales[idx][indexer], unit=units[indexer][0]
        )
        assert obs.compressor == compressor.get_config()
