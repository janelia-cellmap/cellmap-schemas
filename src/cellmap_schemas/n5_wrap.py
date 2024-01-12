from pydantic_zarr.v2 import ArraySpec, GroupSpec
from typing import overload, Union

from zarr.core import Array
from zarr.storage import BaseStore
import zarr

from cellmap_schemas.base import structure_equal


@overload
def n5_spec_wrapper(spec: ArraySpec) -> ArraySpec:
    ...


@overload
def n5_spec_wrapper(spec: GroupSpec) -> GroupSpec:
    ...


def n5_spec_wrapper(spec: Union[GroupSpec, ArraySpec]) -> Union[GroupSpec, ArraySpec]:
    """
    Convert an instance of GroupSpec into one that can be materialized
    via `zarr.N5FSStore`. This requires changing array compressor metadata
    and checking that the `dimension_separator` attribute is compatible wih N5FSStore.

    Parameters
    ----------
    spec: Union[GroupSpec, ArraySpec]
        The spec to transform. An n5-compatible version of this spec will be generated.

    Returns
    -------
    Union[GroupSpec, ArraySpec]
    """
    if isinstance(spec, ArraySpec):
        return n5_array_wrapper(spec)
    else:
        return n5_group_wrapper(spec)


def n5_group_wrapper(spec: GroupSpec) -> GroupSpec:
    """
    Transform a GroupSpec to make it compatible with N5FSStore. This function
    recursively applies itself `n5_spec_wrapper` on its members to produce an
    n5-compatible spec.

    Parameters
    ----------
    spec: GroupSpec
        The spec to transform. Only array descendants of this spec will actually be
        altered after the transformation.

    Returns
    -------
    GroupSpec
    """
    new_members = {}
    for key, member in spec.members.items():
        if hasattr(member, "shape"):
            new_members[key] = n5_array_wrapper(member)
        else:
            new_members[key] = n5_group_wrapper(member)

    return spec.__class__(attributes=spec.attributes, members=new_members)


def n5_array_wrapper(spec: ArraySpec) -> ArraySpec:
    """
    Transform an ArraySpec into one that is compatible with `zarr.N5FSStore`. This function
     ensures that the `dimension_separator` of the ArraySpec is ".".

    Parameters
    ----------
    spec: ArraySpec
        ArraySpec instance to be transformed.

    Returns
    -------
    ArraySpec
    """
    return spec.__class__(**(spec.model_dump() | dict(dimension_separator=".")))


def n5_array_unwrapper(spec: ArraySpec) -> ArraySpec:
    """
    Transform an ArraySpec from one parsed from an array stored in N5. This function
    applies two changes: First, the `dimension_separator` of the ArraySpec is set to
    "/", and second, the `compressor` field has some N5-specific wrapping removed.

    Parameters
    ----------
    spec: ArraySpec
        The ArraySpec to be transformed.

    Returns
    -------
    ArraySpec
    """
    new_metadata = dict(compressor=spec.compressor["compressor_config"], dimension_separator="/")
    return spec.__class__(**(spec.model_dump() | new_metadata))


def n5_group_unwrapper(spec: GroupSpec) -> GroupSpec:
    """
    Transform a GroupSpec to remove the N5-specific attributes. Used when generating
    GroupSpec instances from Zarr groups that are stored using `zarr.N5FSStore`.
    This function will be applied recursively to subgroups; subarrays will be
    transformed with `n5_array_unwrapper`.

    Parameters
    ----------
    spec: GroupSpec
        The spec to be transformed.

    Returns
    -------
    GroupSpec
    """
    new_members = {}
    for key, member in spec.members.items():
        if isinstance(member, ArraySpec):
            new_members[key] = n5_array_unwrapper(member)
        else:
            new_members[key] = n5_group_unwrapper(member)
    return spec.__class__(attributes=spec.attributes, members=new_members)


@overload
def n5_spec_unwrapper(spec: ArraySpec) -> ArraySpec:
    ...


@overload
def n5_spec_unwrapper(spec: GroupSpec) -> GroupSpec:
    ...


def n5_spec_unwrapper(spec: Union[GroupSpec, ArraySpec]) -> Union[GroupSpec, ArraySpec]:
    """
    Transform a GroupSpec or ArraySpec to remove the N5-specific attributes.
    Used when generating GroupSpec or ArraySpec instances from Zarr groups that are
    stored using N5FSStore. If the input is an instance of GroupSpec, this
    function will be applied recursively to subgroups; subarrays will be transformed
    via `n5_array_unwrapper`. If the input is an ArraySpec, it will be transformed with
    `n5_array_unwrapper`.

    Parameters
    ----------
    spec: Union[GroupSpec, ArraySpec]
        The spec to be transformed.

    Returns
    -------
    Union[GroupSpec, ArraySpec]
    """
    if isinstance(spec, ArraySpec):
        return n5_array_unwrapper(spec)
    else:
        return n5_group_unwrapper(spec)


class N5ArraySpec(ArraySpec):
    def to_zarr(self, store: BaseStore, path: str, overwrite: bool = False) -> zarr.Array:
        """
        Create a Zarr array from an `N5ArraySpec`.

        Parameters
        ----------

        store: zarr.N5FSStore
            The storage backend to use for storage. Must be an instance of `zarr.N5FSStore`.
        path: str
            The location within the storage backend for the Zarr array.
        overwrite: bool
            If `True`, any Zarr arrays or groups at `path` will be removed.
            If `False` (default), Zarr arrays or groups at `path` will result in an error.

        Returns
        -------

        zarr.Array
        """
        wrapped = ArraySpec(**n5_array_wrapper(self).model_dump())
        return wrapped.to_zarr(store, path, overwrite)

    @classmethod
    def from_zarr(cls, zarray: Array):
        """
        Create an instance of this class from a Zarr array. This method will
        raise an exception if the Zarr group is not backed by `zarr.N5FSStore`.

        Parameters
        ----------

        node: zarr.Array
            A Zarr array. The `store` attribute of this array must be an instance of `N5FSStore`.

        Returns
        -------

        An instance of `N5Array`.
        """

        result = n5_array_unwrapper(super().from_zarr(zarray))
        return cls(**result.model_dump())


class N5GroupSpec(GroupSpec):
    def to_zarr(self, store: zarr.N5FSStore, path: str, overwrite: bool = False) -> zarr.Group:
        """
        Create a Zarr group from an `N5GroupSpec`.

        Parameters
        ----------

        store: zarr.N5FSStore
            The storage backend to use for storage. Must be an instance of `zarr.N5FSStore`.
        path: str
            The location within the storage backend for the Zarr Group.
        overwrite: bool
            If `True`, any Zarr arrays or groups at `path` will be removed.
            If `False` (default), Zarr arrays or groups at `path` will result in an error.

        Returns
        -------

        zarr.Group
        """
        if not isinstance(store, zarr.N5FSStore):
            msg = (
                f"Instances of {self.__class__} must use an N5-compatible storage backend, "
                f"namely `zarr.N5FSStore`. Got {store.__class__} instead"
            )
            raise TypeError(msg)
        wrapped = GroupSpec(**n5_group_wrapper(self).model_dump())
        return wrapped.to_zarr(store, path, overwrite=overwrite)

    @classmethod
    def from_zarr(cls, node: zarr.Group) -> "N5GroupSpec":
        """
        Create an instance of this class from a Zarr group. This method will
        raise an exception if the Zarr group is not backed by `zarr.N5FSStore`.

        Parameters
        ----------

        node: zarr.Group
            A Zarr group. The `store` attribute of this group must be an instance of `N5FSStore`.

        Returns
        -------

        An instance of `N5Group`.
        """
        if not isinstance(node.store, zarr.N5FSStore):
            msg = (
                f"Instances of {cls.__name__} must be use an N5-compatible storage backend, "
                f"namely `zarr.N5FSStore`. Got {node.store.__class__} instead"
            )
            raise TypeError(msg)
        else:
            base = n5_group_unwrapper(super().from_zarr(node)).model_dump()
            return cls(**base)


@overload
def from_zarr(element: zarr.Group) -> N5GroupSpec:
    ...


@overload
def from_zarr(element: zarr.Array) -> N5ArraySpec:
    ...


def from_zarr(element: Union[zarr.Array, zarr.Group]) -> Union[N5ArraySpec, N5GroupSpec]:
    """
    Recursively parse a Zarr group or Zarr array into an untyped ArraySpec or GroupSpec.

    Parameters
    ---------
    element : Union[zarr.Array, zarr.Group]

    Returns
    -------
    An instance of GroupSpec or ArraySpec that represents the
    structure of the zarr group or array.
    """

    if isinstance(element, zarr.Array):
        result = N5ArraySpec.from_zarr(element)
    elif isinstance(element, zarr.Group):
        members = {}
        for name, member in element.items():
            if isinstance(member, zarr.Array):
                _item = N5ArraySpec.from_zarr(member)
            elif isinstance(member, zarr.Group):
                _item = N5GroupSpec.from_zarr(member)
            else:
                msg = (
                    f"Unparseable object encountered: {type(member)}. Expected "
                    "zarr.Array or zarr.Group.",
                )
                raise ValueError(msg)
            members[name] = _item

        result = N5GroupSpec(attributes=element.attrs.asdict(), members=members)
        return result
    else:
        msg = (
            f"Object of type {type(element)} cannot be processed by this function. "
            "This function can only parse zarr.Group or zarr.Array"
        )
        raise TypeError(msg)
    return result


def update_n5_attrs(node: Union[zarr.Group, zarr.Array], new_spec: N5GroupSpec):
    """
    Update the attributes of an N5-flavored zarr group and its contents
    """

    # first check that the two groups are structurally compatible
    if not structure_equal(from_zarr(node), new_spec):
        msg = (
            f"The node {node} is not structurally equivalent with the new spec. "
            "This means that some of the non-attributes properties differ between the existing and new spec. "
            "This function may only be applied when the the second argument defines a zarr group that is structurally "
            "identical to the existing zarr group."
        )
        raise ValueError(msg)

    node.attrs.update(new_spec.attributes.model_dump())

    if isinstance(node, zarr.Group):
        for name, member in new_spec.members.items():
            update_n5_attrs(node[name], member)

    return node
