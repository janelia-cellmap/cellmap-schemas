from __future__ import annotations
from typing import Literal

import pytest
from cellmap_schemas.annotation import (
    AnnotationArray,
    AnnotationArrayAttrs,
    AnnotationGroup,
    AnnotationGroupAttrs,
    CropGroup,
    CropGroupAttrs,
    InstancePossibility,
    InstanceSegmentation,
    SemanticPossibility,
    SemanticSegmentation,
    wrap_attributes,
)
import numpy as np
import zarr


@pytest.mark.parametrize(
    "annotation_cls, encoding",
    [
        (SemanticSegmentation, {"absent": 0}),
        (SemanticSegmentation, {"absent": 0, "present": 1}),
        (SemanticSegmentation, {"absent": 0, "present": 1, "unknown": 3}),
        (InstanceSegmentation, {"absent": 0}),
        (InstanceSegmentation, {"absent": 0, "unknown": 3}),
    ],
)
def test_cropgroup(
    annotation_cls: type[SemanticSegmentation] | type[InstanceSegmentation],
    encoding: dict[SemanticPossibility, int] | dict[InstancePossibility, int],
):
    ClassNamesT = Literal["foo", "bar"]
    class_names: list[Literal["foo", "bar"]] = ["foo", "bar"]
    ann_type = annotation_cls(encoding=encoding)  # type: ignore
    arrays = [np.zeros(10) for class_name in class_names]
    crop_group_attrs = CropGroupAttrs[ClassNamesT](
        name="foo",
        description="a description",
        created_by=["person_a", "person_b"],
        created_with=["amira"],
        start_date="2022-03-01",
        end_date="2022-04-01",
        duration_days=4,
        protocol_uri=None,
        class_names=class_names,
    )

    ann_groups = {}

    for array, class_name in zip(arrays, class_names):
        ann_array_attrs = AnnotationArrayAttrs[ClassNamesT](
            class_name=class_name, complement_counts={"absent": 0}, annotation_type=ann_type
        )
        ann_array = AnnotationArray.from_array(
            array, attributes=wrap_attributes(ann_array_attrs).model_dump()
        )

        ann_group_attrs = AnnotationGroupAttrs[ClassNamesT](
            class_name=class_name, annotation_type=ann_type
        )

        # check that an extra attribute is OK
        ann_group = AnnotationGroup(
            members={"s0": ann_array},
            attributes={**wrap_attributes(ann_group_attrs).model_dump(), **{"foo": 10}},
        )
        ann_groups[class_name] = ann_group

    crop_group = CropGroup(
        members=ann_groups, attributes=wrap_attributes(crop_group_attrs).model_dump()
    )

    stored = crop_group.to_zarr(zarr.MemoryStore(), path="test")
    observed = CropGroup.from_zarr(stored)
    assert observed == crop_group

    # insert an extra group called "all"
    extra_group_name = "all"
    stored.create_group(extra_group_name)
    observed_2 = CropGroup.from_zarr(stored)
    assert extra_group_name not in observed_2.members
    assert observed_2 == observed

    # remove a labelled group
    to_remove = class_names[0]
    del stored[to_remove]

    match = (
        f"Expected to find a group named {to_remove} in {stored.store}://{stored.name}. "
        "No group was found with that name."
    )
    with pytest.raises(ValueError, match=match):
        CropGroup.from_zarr(stored)


@pytest.mark.parametrize(
    "annotation_cls, encoding",
    [
        (SemanticSegmentation, {"absent": 0}),
        (SemanticSegmentation, {"absent": 0, "present": 1}),
        (SemanticSegmentation, {"absent": 0, "present": 1, "unknown": 3}),
        (InstanceSegmentation, {"absent": 0}),
        (InstanceSegmentation, {"absent": 0, "unknown": 3}),
    ],
)
def test_annotation_attrs_from_array(annotation_cls, encoding) -> None:
    array = np.array([0, 1, 0, 3])
    attrs = AnnotationArrayAttrs.from_array(
        array,
        class_name="foo",
        annotation_type=annotation_cls(encoding=encoding),
        complement_counts="auto",
    )
    assert attrs.complement_counts["absent"] == 2
