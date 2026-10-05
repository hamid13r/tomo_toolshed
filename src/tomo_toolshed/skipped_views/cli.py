"""``tomo_toolshed skipped-views`` — prune skipped etomo views from WarpTools XML.

Two modes:

  * **flip** (default) — update the ``UseTilt`` flags in WarpTools tilt-series
    XML from the etomo ``taSolution.log`` produced during fine alignment (with
    optional dose- or tilt-angle-based selection), backing up the originals.
  * **delete** (``--delete``) — physically remove the excluded tilts from both
    the XML and the matching ``.tomostar`` file.
"""

import click

from .core import process_xml_files


@click.command(name="skipped-views")
@click.option('--xml-dir', default='./', help='Directory containing XML files')
@click.option('--xml-pattern', default='*.xml', help='Pattern to match XML files')
@click.option('--backup-dir', default='backup_xml',
              help='Directory to store XML backups (flip mode)')
@click.option('--tiltstack-dir', default='tiltstack',
              help='Base directory for tiltstack logs')
@click.option('--tomostar-dir', default='../tomostar',
              help='Directory containing .tomostar files (delete mode)')
@click.option('--all-true', is_flag=True, default=False,
              help='Set all UseTilt values to True')
@click.option('--n-tilts', default=0,
              help='Keep the N lowest-dose tilts (that survived alignment), '
                   'discard the rest')
@click.option('--max-tilt', default=0,
              help='Maximum tilt angle (from the minimum-dose view) to keep, '
                   'others discarded')
@click.option('--delete', 'do_delete', is_flag=True, default=False,
              help='Physically remove excluded tilts from the XML and .tomostar '
                   'instead of setting UseTilt=False')
@click.option('--dry-run', is_flag=True, default=False,
              help='Report what would change and write nothing')
def skipped_views(xml_dir, xml_pattern, backup_dir, tiltstack_dir, tomostar_dir,
                  all_true, n_tilts, max_tilt, do_delete, dry_run):
    """Prune skipped etomo views from WarpTools tilt-series XML files.

    By default (flip mode) this updates UseTilt True/False from taSolution.log;
    with --delete it physically removes the excluded tilts from both the XML and
    the matching .tomostar file. Selection supports dose-based (--n-tilts),
    tilt-angle-based (--max-tilt), or reset-all (--all-true). Originals are
    backed up first; use --dry-run to preview.

    \b
    examples:
      tomo_toolshed skipped-views --xml-dir ./ --tiltstack-dir tiltstack
      tomo_toolshed skipped-views --delete --tomostar-dir ../tomostar --dry-run
    """
    process_xml_files(xml_dir, xml_pattern, backup_dir, tiltstack_dir,
                      tomostar_dir, all_true, n_tilts, max_tilt, do_delete,
                      dry_run)


if __name__ == '__main__':
    skipped_views()
