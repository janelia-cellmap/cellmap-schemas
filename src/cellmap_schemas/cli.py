from typing import Literal, Tuple, Union
import click
from pydantic_zarr import ArraySpec, GroupSpec, from_zarr
from pydantic import ValidationError
from cellmap_schemas.multiscale import cosem, neuroglancer_n5
import zarr
from rich.console import Console
from rich.traceback import install
from rich import print_json

install()

GROUP_TYPES = ('multiscale.cosem.Group', 'multiscale.neuroglancer_n5.Group')


@click.group
def cli():
	pass


def validate(
	cls: Union[GroupSpec, ArraySpec], node: Union[zarr.Array, zarr.Group], console: Console
):
	class_name = f'{cls.__module__}.{cls.__name__}'
	protocol = 'file'
	if hasattr(node.store, 'fs'):
		if isinstance(node.store.fs.protocol, tuple):
			protocol = node.store.fs.protocol[0]
	node_path = protocol + '://' + node.store.path + '/' + node.path
	try:
		cls.from_zarr(node)
		console.print(f'[bold]{class_name}: [blue]{node_path}[/blue] validated ' 'successfully.')
	except ValidationError as e:
		console.print(
			f'[bold]{class_name}[/bold]: [blue]{node_path}[/blue] failed '
			'validation due to the following validation errors:'
		)
		console.print(str(e))


@cli.command
@click.argument('url', type=click.STRING)
@click.argument('group_type', type=click.STRING)
def check(url: str, group_type: str):
	"""
	Check whether a Zarr / N5 hierarchy, specified by a url, is consistent with a model,
	specified by a class name.

	Parameters\n
	----------

	url: A url referring to a Zarr / N5 group or dataset.

	group_type: A class name for one of the models defined in `cellmap-schemas`.\n
	        Avaialble options: \n
	                multiscale.cosem.Group\n
	                multiscale.neuroglancer_n5.Group
	"""
	store_stem, prefix, component_path = parse_url(url)
	store = guess_store(store_stem=store_stem, prefix=prefix)
	node = zarr.open(store, path=component_path, mode='r')

	console = Console()
	if group_type == 'multiscale.cosem.Group':
		validate(cosem.Group, node, console)
	elif group_type == 'multiscale.neuroglancer_n5.Group':
		validate(neuroglancer_n5.Group, node, console)
	else:
		console.print(
			f'`group_type` parameter "{group_type}" was not recognized.\n'
			'Choose from one of the following options:'
		)
		for gt in GROUP_TYPES:
			console.print('\t' + gt)


@cli.command
@click.argument('url', type=click.STRING)
def inspect(url: str):
	store_stem, prefix, component_path = parse_url(url)
	store = guess_store(store_stem=store_stem, prefix=prefix)
	node = zarr.open(store, path=component_path, mode='r')
	print_json(from_zarr(node).json(indent=2))


def guess_store(
	store_stem: str, prefix: Literal['.zarr', '.n5']
) -> Union[zarr.N5FSStore, zarr.storage.BaseStore]:
	if prefix == '.n5':
		store = zarr.N5FSStore(store_stem + prefix)
	else:
		store = zarr.storage.FSStore(store_stem + prefix)
	return store


def parse_url(url: str) -> Tuple[str, Literal['.zarr', '.n5'], str]:
	"""
	Given the suffixes '.n5' and '.zarr', this function parses a string into a
	pre-suffix component, a suffix, and a post-suffix component.

	Parameters
	----------

	url: str
	        An FSSpec-compatible URL, e.g. s3://bucket/path.zarr/foo or http://domain.com/path.n5/bar

	Returns
	-------
	Tuple[str, Literal['.zarr', '.n5'], str]
	"""

	# try zarr
	if '.n5' not in url and '.zarr' not in url:
		raise ValueError(
			f'Invalid url parameter. Got {url}, but expected a string containing a at '
			'least one instance of *.n5 or *.zarr. Are you sure this url refers to '
			'Zarr or N5 storage?'
		)
	if '.n5' and '.zarr' in url:
		raise ValueError(
			'Invalid url parameter. Because this url contains both the .n5 and .zarr '
			'substrings, it is ambiguous. Valid urls must contain only one instance of '
			'.zarr or one instance of .n5'
		)
	if '.n5' in url:
		return url.partition('.n5')
	else:
		return url.partition('.zarr')
