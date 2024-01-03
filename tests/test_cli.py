from cellmap_schemas.cli import parse_url
import pytest


@pytest.mark.parametrize(
    "data, expected",
    [
        ("/foo.zarr/bar/baz", ("/foo", ".zarr", "/bar/baz")),
        ("/foo.n5/bar/baz", ("/foo", ".n5", "/bar/baz")),
    ],
)
def test_parse_url(data, expected):
    parsed = parse_url(data)
    assert parsed == expected


def test_broken_url():
    with pytest.raises(ValueError):
        parse_url("foo.n5/bar.zarr")
