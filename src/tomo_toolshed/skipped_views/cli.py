"""``tomo_toolshed skipped-views`` — prune skipped etomo views from WarpTools XML.

Updates the ``UseTilt`` flags in WarpTools tilt-series XML files from the etomo
``taSolution.log`` produced during fine alignment (with optional dose- or
tilt-angle-based selection), backing up the originals first.
"""

import click

from .core import process_xml_files


@click.command(name="skipped-views")
@click.option('--xml-dir', default='./', help='Directory containing XML files')
@click.option('--xml-pattern', default='*.xml', help='Pattern to match XML files')
@click.option('--backup-dir', default='backup_xml', help='Directory to store XML backups')
@click.option('--tiltstack-dir', default='tiltstack', help='Base directory for tiltstack logs')
@click.option('--all-true', is_flag=True, default=False, help='Set all UseTilt values to True')
@click.option('--n-tilts', default=0, help='Number of tilts by dose to keep and discard the rest')
@click.option('--max-tilt', default=0, help='Maximum tilt angle (calculated from the minimum dose) to keep, others set to False')
def skipped_views(xml_dir, xml_pattern, backup_dir, tiltstack_dir, all_true, n_tilts, max_tilt):
    """Prune skipped etomo views from WarpTools tilt-series XML files.

    Updates UseTilt values based on taSolution.log; supports dose-based
    (--n-tilts) and tilt-angle-based (--max-tilt) selection, or resetting
    everything to True (--all-true). Originals are backed up first.

    example:
    tomo_toolshed skipped-views --xml-dir ./ --xml-pattern '*.xml' --backup-dir backup_xml --tiltstack-dir tiltstack
    """
    process_xml_files(xml_dir, xml_pattern, backup_dir, tiltstack_dir, all_true, n_tilts, max_tilt)


if __name__ == '__main__':
    skipped_views()
