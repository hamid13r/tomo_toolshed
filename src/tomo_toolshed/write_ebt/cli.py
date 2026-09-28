"""``tomo_toolshed write-ebt`` -- build an etomo batchruntomo ``.ebt`` project.

Scans a directory laid out as one subdirectory per tilt series
(``<name>/<name>.st``) and writes a batchruntomo ``.ebt`` file: a dataset-wide
header followed by one row per tilt series, each pointing at that series' ``.st``
stack. ``--platform`` selects how the ``.st`` paths are spelled and should match
the machine that will *open* the file in etomo, not the one writing it -- the
whole reason there used to be two scripts. The heavy lifting lives in ``core.py``.
"""

import click

from .core import DEFAULT_ROOT_NAME, build_ebt, list_series_dirs


@click.command(name="write-ebt")
@click.option('--o', prompt='Enter the name of the ebt file',
              help='Enter the name of the ebt file')
@click.option('--platform', type=click.Choice(['linux', 'windows']),
              prompt='Target platform for the .st paths',
              help='Path style for the .st stacks: match the machine that will '
                   'OPEN the .ebt in etomo, not the one writing it.')
@click.option('--root', '-r', default='.', show_default=True,
              type=click.Path(exists=True, file_okay=False),
              help='Directory to scan for tilt-series subdirectories.')
@click.option('--root-name', default=DEFAULT_ROOT_NAME, show_default=True,
              help='Value written into meta.RootName.')
def write_ebt(o, platform, root, root_name):
    """Write an etomo batchruntomo .ebt project file.

    Each subdirectory of --root is treated as one tilt series and becomes a row
    pointing at <root>/<name>/<name>.st. Loose files in --root (including the
    .ebt being written) are ignored. --platform picks the .st path style and
    should match the machine that will open the file in etomo.

    \b
    example:
    tomo_toolshed write-ebt --o batch.ebt --platform linux
    tomo_toolshed write-ebt --o batch.ebt --platform windows --root ts_data
    """
    directories = list_series_dirs(root)
    with open(o, 'w') as f:
        f.write(build_ebt(directories, root, platform, root_name=root_name))

    click.echo(f"wrote {o}: {len(directories)} tilt-series row(s) "
               f"({platform} paths)")
    for name in directories:
        click.echo(f"  {name}")


if __name__ == '__main__':
    write_ebt()
