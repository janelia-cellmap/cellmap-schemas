# Welcome to cellmap-schemas

This project contains a python library and JSON schema documents that formalize some of the data structures developed by the [Cellmap](https://www.janelia.org/project-team/cellmap) project team at [Janelia Research Campus](https://www.janelia.org/).

## Model checking

Cellmap works with large (multi-TB) imaging datasets. We store our images with chunked array formats like [N5](https://github.com/saalfeldlab/n5) and [Zarr](https://zarr.readthedocs.io/en/stable/), because these formats support performant I/O operations at the terabyte scale, and also because these file formats give developers a wide range of freedom in how arrays are organized, and what metadata is present. But with that freedom comes the need for applications to check whether the N5 or Zarr hierarchies they consume are correctly structured. To address this problem, this library provides [Pydantic models](https://docs.pydantic.dev/latest/) of Cellmap-specific N5 / Zarr hierarchies which can be used for validating N5 / Zarr data.

# Installation

`pip install -U cellmap-schemas`

# Contributing

Raise issues on our [issue tracker](https://github.com/janelia-cellmap/cellmap-schemas/issues). For local development, see the [developer guide](./development.md)