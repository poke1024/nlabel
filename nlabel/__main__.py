import click
from nlabel.io.common import open_archive
from pathlib import Path


@click.group(chain=True)
def cli():
    pass


@cli.command('convert')
@click.argument('path', type=click.Path(exists=True))
@click.option('--to', default='bahia', help='archive format to be saved')
def convert(path, to):
    new_path = Path(path).with_suffix(f".{to}.nlabel")

    print(f"converting {path}...", flush=True)

    with open_archive(path, mode="r") as archive:
        archive.save(new_path, engine=to, progress=True)


cli()
