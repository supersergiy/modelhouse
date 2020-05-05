import click

from modelhouse.storage import secret_manager


@click.command()
@click.option('--storage_folder', '-c', type=click.Path(exists=True))
def set_secrets_folder(secrets_folder):
    secret_manager.set_secrets_folder(secrets_folder)
