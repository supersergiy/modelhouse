import click
from modelhouse.storage import Storage
from modelhouse.storage import get_files_and_contents, put_files_and_contents


@click.command()
@click.option('--src_path', '-s', nargs=1, type=str, required=True)
@click.option('--dst_path', '-d', nargs=1, type=str, required=True)
def create(src_path, dst_path):
    files_and_contents = get_files_and_contents(src_path)

    put_files_and_contents(dst_path, files_and_contents)
    print ("Model successfully created")

@click.command()
@click.option('--src_path', '-s', nargs=1, type=str)
def delete(src_path):
    store = Storage(src_path)
    files = store.list_files(flat=False)
    store.delete_files(files)

