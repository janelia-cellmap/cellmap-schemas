from __future__ import annotations
import re
from typing import Literal

import numpy as np
import pytest
from cellmap_schemas.base import normalize_chunks, structure_equal
from pydantic_zarr.v2 import auto_chunks
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


@pytest.mark.parametrize("chunks", ("auto", (1, 2, 3), ((1, 2, 3), (1, 2, 3), (2, 3, 4))))
def test_normalize_chunks(chunks: Literal["auto"] | tuple[int, ...] | tuple[tuple[int, ...], ...]):
    data = (np.zeros((10, 10, 10)),) * 3
    if chunks == "auto":
        expected = (auto_chunks(data[0]),) * len(data)
    elif isinstance(chunks[0], int):
        expected = (chunks,) * len(data)
    else:
        expected = chunks
    observed = normalize_chunks(chunks, data)
    assert observed == expected


def test_normalize_chunks_wrong_length():
    arrays = (np.zeros((1, 1, 1)),) * 2
    chunks = ((1, 1, 1),)
    match = (
        f"The number of chunks ({len(chunks)}) does not match the number of "
        f"arrays ({len(arrays)})"
    )
    with pytest.raises(ValueError, match=re.escape(match)):
        normalize_chunks(chunks, arrays)
