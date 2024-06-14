from __future__ import annotations
import numpy.typing as npt
from typing import Any, Iterable, Literal
from pydantic_zarr.v2 import ArraySpec, GroupSpec
from pydantic_zarr.v2 import auto_chunks


def structure_group_equal(spec_a: GroupSpec, spec_b: GroupSpec) -> bool:
    keys_equal = spec_a.members.keys() == spec_b.members.keys()
    if keys_equal:
        for member_a, member_b in zip(spec_a.members.values(), spec_b.members.values()):
            if isinstance(member_a, ArraySpec):
                member_eq = member_a.like(member_b, exclude={"attributes"})
            else:
                member_eq = structure_group_equal(member_a, member_b)
            if member_eq is False:
                return False
        return True
    else:
        return False


def structure_equal(spec_a: ArraySpec | GroupSpec, spec_b: ArraySpec | GroupSpec) -> bool:
    """
    Check that two GroupSpec / ArraySpec instances are identical up to their attributes
    """
    if not (
        isinstance(spec_a, (ArraySpec, GroupSpec)) and isinstance(spec_b, (ArraySpec, GroupSpec))
    ):
        raise TypeError(
            f"Cannot apply this function to objects with types ({type(spec_a)}, {type(spec_b)})"
        )
    if isinstance(spec_a, ArraySpec) and isinstance(spec_b, ArraySpec):
        return spec_a.like(spec_b, exclude="attributes")
    elif isinstance(spec_a, GroupSpec) and isinstance(spec_b, GroupSpec):
        return structure_group_equal(spec_a, spec_b)
    else:
        return False


def normalize_chunks(
    chunks: Literal["auto"] | tuple[int, ...] | tuple[tuple[int, ...]],
    arrays: Iterable[npt.NDArray[Any]],
) -> tuple[tuple[int, ...], ...]:
    """
    Normalize a specification of chunks against a collection of arrays. Returns a tuple of tuples of ints.

    This handles 3 cases:
    - If `chunks` is the string "auto", then for each array in `arrays`, the `auto_chunks` routine
    from `pydantic-zarr` is used to estimate a chunk size for the first (largest) array, and that chunk size
    is used for all arrays
    - If `chunks` is a tuple of integers, then that is used as the chunk size for all arrays
    - If `chunks` is a tuple of tuples of integers, then that is returned after some minor
    validation.
    """
    arrays_tuple = tuple(arrays)

    if chunks == "auto":
        return (auto_chunks(arrays_tuple[0]),) * len(arrays)
    elif all(isinstance(x, int) for x in chunks):
        return (chunks,) * len(arrays_tuple)
    elif all(all(isinstance(x, int) for x in t) for t in chunks):
        result = tuple(map(tuple, chunks))
        if len(result) != len(arrays_tuple):
            msg = (
                f"The number of chunks ({len(chunks)}) does not match the number of "
                f"arrays ({len(arrays_tuple)})"
            )
            raise ValueError(msg)
        return result
    else:
        msg = (
            f'Invalid chunks: {chunks}. Expected the string "auto"'
            "a tuple of ints, or a tuple of tuples of ints."
        )
        raise ValueError(msg)
