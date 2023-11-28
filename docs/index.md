# Welcome to cellmap-schemas

This project contains a [`pydantic`](https://docs.pydantic.dev/latest/)-based python library that formalizes some of the data structures developed by the [Cellmap](https://www.janelia.org/project-team/cellmap) project team at [Janelia Research Campus](https://www.janelia.org/).

## Model checking

Cellmap works with large (multi-TB) imaging datasets. We store our images with chunked array formats like [N5](https://github.com/saalfeldlab/n5) and [Zarr](https://zarr.readthedocs.io/en/stable/), because these formats support performant I/O operations at the terabyte scale, and also because these file formats give developers a wide range of freedom in how arrays are organized, and what metadata is present. But with that freedom comes the need for applications to check whether the N5 or Zarr hierarchies they consume are correctly structured. To address this problem, this library provides [Pydantic models](https://docs.pydantic.dev/latest/) of Cellmap-specific N5 / Zarr hierarchies which can be used for validating N5 / Zarr data.

### Creating a Neuroglancer-compatible N5 group

In this example, we create an N5 hierarchy that complies with the [Neuroglancer N5 convention](https://github.com/google/neuroglancer/issues/176#issuecomment-553027775):

```python
from pydantic_zarr import ArraySpec
from cellmap_schemas.multiscale.neuroglancer_n5 import Group, PixelResolution, GroupMetadata
import numpy as np

# define a toy multiscale image
data = np.arange(16).reshape(4,4)
data_ds = data[::2, ::2]
arrays = {
    's0': ArraySpec.from_array(data),
    's1': ArraySpec.from_array(data_ds)
    }
ngroup = Group(
    members=arrays, 
    attrs=GroupMetadata(
        scales=[[1,1], [2,2]], 
        axes=['x','y'],
        units=['nm','nm'],
        pixelResolution= PixelResolution(dimensions=[4,4], unit='nm')))

# prepare the hiearchy for writing data by calling 
# stored_group = ngroup.to_zarr(
#   zarr.N5FSStore('path/to/n5/root.n5'), 
#   path='foo')
# then write data, e.g.
# stored_group['s0'][:] = data
# stored_group['s1'][:] = data_ds

```



# Installation

`pip install -U cellmap-schemas`

# Contributing

Raise issues on our [issue tracker](https://github.com/janelia-cellmap/cellmap-schemas/issues). For local development, see the [developer guide](./development.md)