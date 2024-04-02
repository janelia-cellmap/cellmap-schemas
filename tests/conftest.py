from __future__ import annotations
import pytest
from zarr import N5Store, N5FSStore
from zarr.storage import FSStore, NestedDirectoryStore, MemoryStore


@pytest.fixture(scope="function")
def n5store(tmpdir):
    return N5Store(str(tmpdir))


@pytest.fixture(scope="function")
def n5fsstore_local(tmpdir):
    return N5FSStore(str(tmpdir))


@pytest.fixture(scope="function")
def fsstore_local(tmpdir):
    return FSStore(str(tmpdir))


@pytest.fixture(scope="function")
def nested_directory_store(tmpdir):
    return NestedDirectoryStore(str(tmpdir))


@pytest.fixture(scope="function")
def memory_store():
    return MemoryStore()
