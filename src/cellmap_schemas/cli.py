from typing import Union
import click
from pydantic_zarr import ArraySpec, GroupSpec
from pydantic import ValidationError
from cellmap_schemas.cosem import MultiscaleGroup
import zarr
from rich.console import Console
from rich.traceback import install

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
		msg = f'[bold]{cls.__name__}: Node [blue]{node_path}[/blue] validated successfully.'
		console.print()
	except ValidationError as e:
		msg = f'[bold]{cls.__name__}[/bold]: Node [blue]{node_path}[/blue] failed validation due to the following validation errors:'
		console.print(msg)
		console.print(str(e))


@click.command
@click.argument('store_path', type=click.STRING)
@click.argument('node_path', type=click.STRING)
@click.argument('group_type', type=click.STRING)
def validate_cli(store_path: str, node_path: str, group_type: str):
	console = Console()
	group_types = ('cosem_multiscale_group',)
	if group_type == 'cosem_multiscale_group':
		store = zarr.N5FSStore(store_path)
		node = zarr.open(store, path=node_path, mode='r')
		validate(MultiscaleGroup, node, console)
	else:
		msg = f'group_type parameter "{group_type}" was not recognized.\n' f'Available options:'
		console.print(msg)
		[console.print('\t' + gt) for gt in group_types]
