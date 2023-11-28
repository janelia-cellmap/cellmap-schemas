from typing import Union
import click
from pydantic_zarr import ArraySpec, GroupSpec, from_zarr
from pydantic import ValidationError
from cellmap_schemas.multiscale import cosem, neuroglancer_n5
import zarr
from rich.console import Console
from rich.traceback import install
from rich import print_json

install()


def validate(
	cls: Union[GroupSpec, ArraySpec], node: Union[zarr.Array, zarr.Group], console: Console
):
	protocol = 'file'
	if hasattr(node.store, 'fs'):
		if isinstance(node.store.fs.protocol, tuple):
			protocol = node.store.fs.protocol[0]
	node_path = protocol + '://' + node.store.path + '/' + node.path
	try:
		cls.from_zarr(node)
		console.print(
			f'[bold]{cls.__name__}: Node [blue]{node_path}[/blue] validated ' 'successfully.'
		)
	except ValidationError as e:
		console.print(
			f'[bold]{cls.__name__}[/bold]: Node [blue]{node_path}[/blue] failed '
			'validation due to the following validation errors:'
		)
		console.print(str(e))


@click.command
@click.argument('store_path', type=click.STRING)
@click.argument('node_path', type=click.STRING)
@click.argument('group_type', type=click.STRING)
def validate_cli(store_path: str, node_path: str, group_type: str):
	console = Console()
	group_types = ('multiscale.cosem.Group', 'multiscale.neuroglancer.Group')
	if group_type == 'multiscale.cosem.Group':
		store = zarr.N5FSStore(store_path)
		node = zarr.open(store, path=node_path, mode='r')
		validate(cosem.Group, node, console)
	elif group_type == 'multiscale.neuroglancer.Group':
		store = zarr.N5FSStore(store_path)
		node = zarr.open(store, path=node_path, mode='r')
		validate(neuroglancer_n5.Group, node, console)
	else:
		msg = f'group_type parameter "{group_type}" was not recognized.\n' f'Available options:'
		console.print(msg)
		for gt in group_types:
			console.print('\t' + gt)


@click.command
@click.argument('store_path', type=click.STRING)
@click.argument('node_path', type=click.STRING)
def inspect_cli(store_path: str, node_path: str):
	if store_path.endswith('.n5') or store_path.endswith('.n5/'):
		store = zarr.N5FSStore(store_path)
	else:
		store = store_path
	node = zarr.open(store, path=node_path, mode='r')
	print_json(from_zarr(node).json(indent=2))
