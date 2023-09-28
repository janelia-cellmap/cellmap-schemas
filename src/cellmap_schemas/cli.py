import click
from .annotation import CropGroup

@click.command
def schematize():
    click.echo(CropGroup.schema_json(indent=2))