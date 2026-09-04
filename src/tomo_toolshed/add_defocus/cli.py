"""``tomo_toolshed add-defocus`` — fill in IsoNet ``_rlnDefocus`` from Warp XML.

Reads the per-tilt ``GridCTF`` values of each tilt series' Warp XML file,
averages them, and writes the result into the ``_rlnDefocus`` column of an
IsoNet star file produced by ``isonet.py prepare_star``.
"""

import click

from .core import (
    DEFAULT_TOMONAME_REGEX,
    DEFAULT_XML_SUFFIX,
    StarFormatError,
    update_star_defocus,
)


@click.command(name="add-defocus")
@click.option('--star', 'star_file', required=True,
              type=click.Path(exists=True, dir_okay=False),
              help='IsoNet star file to update (from isonet.py prepare_star)')
@click.option('--xml-dir', default='xml', show_default=True,
              type=click.Path(exists=True, file_okay=False),
              help='Directory containing the Warp XML files')
@click.option('--output', '-o', default=None, type=click.Path(dir_okay=False),
              help='Write the result here instead of editing --star in place')
@click.option('--xml-suffix', default=DEFAULT_XML_SUFFIX, show_default=True,
              help='Suffix appended to the tomogram base name to find its XML')
@click.option('--tomoname-regex', default=DEFAULT_TOMONAME_REGEX, show_default=True,
              help='Regex whose first group extracts the base name from _rlnTomoName')
@click.option('--strict', is_flag=True, default=False,
              help='Fail on the first unparseable row or missing XML instead of skipping it')
@click.option('--dry-run', is_flag=True, default=False,
              help='Report the defocus values without writing anything')
@click.option('--quiet', '-q', is_flag=True, default=False,
              help='Only print the final summary')
def add_defocus(star_file, xml_dir, output, xml_suffix, tomoname_regex,
                strict, dry_run, quiet):
    """Add per-tomogram average defocus to an IsoNet star file.

    Each _rlnTomoName in the star file is mapped to a Warp XML file in
    --xml-dir, whose per-tilt GridCTF values are averaged (microns -> Angstroms)
    and written into _rlnDefocus. The star file is edited in place unless
    --output is given. Rows with an unparseable name or a missing XML are
    skipped with a warning, or abort the run under --strict.

    \b
    example:
    tomo_toolshed add-defocus --star tomos.star --xml-dir xml
    """
    try:
        updated, skipped = update_star_defocus(
            star_file,
            xml_dir,
            output=output,
            strict=strict,
            tomoname_regex=tomoname_regex,
            xml_suffix=xml_suffix,
            dry_run=dry_run,
            log=None if quiet else click.echo,
        )
    except (StarFormatError, ValueError) as exc:
        raise click.ClickException(str(exc))

    destination = output or star_file
    if dry_run:
        click.echo(f"dry run: {len(updated)} row(s) would be updated, "
                   f"{len(skipped)} skipped; {destination} not written")
    else:
        click.echo(f"updated {len(updated)} row(s), skipped {len(skipped)}; "
                   f"wrote {destination}")

    if skipped:
        click.secho(f"{len(skipped)} row(s) left unchanged:", fg='yellow', err=True)
        for tomoname, reason in skipped:
            click.secho(f"  {tomoname or '<row>'}: {reason}", fg='yellow', err=True)


if __name__ == '__main__':
    add_defocus()
