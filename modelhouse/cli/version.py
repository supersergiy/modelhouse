import click

@click.command()
def version():
    """
    View the current version of the CLI.
    """
    import pkg_resources
    version = pkg_resources.require(PROJECT_NAME)[0].version
    print (version)
