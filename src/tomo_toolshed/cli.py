"""Top-level ``tomo_toolshed`` command group.

Every tool in the toolshed is registered here as a subcommand. To add a new
tool: create a subpackage under ``src/tomo_toolshed/`` exposing a click command,
import it below, and register it with :func:`tomo_toolshed.add_command`.
"""

import click

from .skipped_views.cli import skipped_views
from .segmentation_curator.cli import main as curate


# Declare the group name explicitly with underscores so click does not rewrite
# it to "tomo-toolshed"; the installed entry point is `tomo_toolshed`.
@click.group(name="tomo_toolshed")
@click.version_option(package_name="tomo-toolshed")
def tomo_toolshed():
    """Lightweight cryo-ET tools, each available as a subcommand.

    \b
    skipped-views  Prune skipped etomo views from WarpTools tilt-series XML.
    curate         Curate a 3D segmentation over a tomogram and export a mask.
    """


tomo_toolshed.add_command(skipped_views, name="skipped-views")
tomo_toolshed.add_command(curate, name="curate")


if __name__ == "__main__":
    tomo_toolshed()
