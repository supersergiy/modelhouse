import click
from modelhouse.storage import Storage

@click.command()
@click.option('--model_materials_dir', '-m', type=click.Path(exists=True))
@click.option('--destination_path', '-p', nargs=1, type=str, default=None)
def create(model_materials_dir, destination_path):

    uplodable_files = []
    for dir_path, _, files in os.walk(root_dir):
        rel_dir = os.path.relpath(dir_path, model_materials_dir)

        for file_abspath in files:
            file_name = os.path.basename(file_name)
            file_relpath = os.path.join(rel_dir, file_abspath)
            with open(file_abspath, 'r') as f:
                file_contents = f.read()
            uploadable_files.append((file_relpath, file_contents))

    store = Storage(destination_path)
    store.put_files(
            uploadable_files,
            content_type='application/octet-stream',
            compress='gzip',
            cache_control='no-cache'
    )
    print ("Model successfully created")

@click.command()
@click.option('--model_path', '-p', nargs=1, type=str)
def delete(model_path):
    store = Storage(model_path)
    files = store.list_files(flat=False)
    store.delete_files(files)

