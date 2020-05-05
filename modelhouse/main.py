import click

from modelhouse.cli.models import create, delete
from modelhouse.cli.storage import set_secrets_folder
from modelhouse.cli.version import version

@click.group()
@click.option('-v', '--verbose', count=True, help='Turn on debug logging')
def cli(verbose):
    #configure_logger(verbose)
    pass

def add_commands(cli):
    cli.add_command(create)
    cli.add_command(delete)
    cli.add_command(set_secrets_folder)
    cli.add_command(version)

add_commands(cli)
