from pydantic_zarr.v2 import ArraySpec, GroupSpec


def structure_array_equal(spec_a: ArraySpec, spec_b: ArraySpec) -> bool:
    return spec_a.model_dump(exclude=["attributes"]) == spec_b.model_dump(exclude=["attributes"])


def structure_group_equal(spec_a: GroupSpec, spec_b: GroupSpec) -> bool:
    keys_equal = spec_a.members.keys() == spec_b.members.keys()
    if keys_equal:
        for member_a, member_b in zip(spec_a.members.values(), spec_b.members.values()):
            if isinstance(member_a, ArraySpec):
                member_eq = structure_array_equal(member_a, member_b)
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
        return structure_array_equal(spec_a, spec_b)
    elif isinstance(spec_a, GroupSpec) and isinstance(spec_b, GroupSpec):
        return structure_group_equal(spec_a, spec_b)
    else:
        return False
