import click

@click.command()
@click.argument('env_name', default='hello')
def print_env(env_name):
    print(env_name)

print_env()