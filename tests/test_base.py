from __future__ import annotations
from cellmap_schemas.base import structure_equal


from pydantic_zarr.v2 import ArraySpec, GroupSpec


def test_structure_equal():
    array_a = ArraySpec(
        shape=(10, 10),
        chunks=(10, 10),
        dtype="uint8",
        compressor=None,
        filters=None,
        fill_value=0,
        attributes={},
    )
    array_b = array_a.model_copy(deep=True, update={"shape": (20, 20)})

    assert structure_equal(array_a, array_a)
    assert not structure_equal(array_a, array_b)

    group_a = GroupSpec(attributes={"foo": 10}, members={"array": array_a})
    group_b = group_a.model_copy(deep=True)
    assert structure_equal(group_a, group_b)

    group_c = group_a.model_copy(deep=True, update={"attributes": {"foo": 1000}})
    assert group_a != group_c
    assert structure_equal(group_a, group_c)

    group_d = group_a.model_copy(deep=True, update={"members": {"array_new": array_a}})
    assert not structure_equal(group_a, group_d)

    group_e = group_a.model_copy(deep=True, update={"members": {"array": array_b}})
    assert not structure_equal(group_a, group_e)
