import os
import shutil
import glob
import io
import tempfile
from collections import defaultdict

import pandas as pd
import click
from lxml import etree

# Warp 2 tilt-series XML files are UTF-8 *with a BOM* and start with this exact
# declaration. lxml drops the BOM and single-quotes the declaration, so we emit
# both ourselves and only let lxml serialize the body (see serialize_xml).
XML_BOM = b"\xef\xbb\xbf"
XML_DECL = b'<?xml version="1.0" encoding="utf-8"?>\n'


# --------------------------------------------------------------------------- #
# XML I/O helpers
# --------------------------------------------------------------------------- #
def parse_xml(path):
    """Parse a Warp XML preserving all whitespace exactly."""
    parser = etree.XMLParser(remove_blank_text=False, strip_cdata=False)
    return etree.parse(path, parser)


def serialize_xml(tree):
    """Serialize a tree back to Warp's exact on-disk byte format.

    lxml preserves element .text/.tail verbatim (so tabs, newlines and the
    "first value on the open tag line" layout survive), but it (a) drops the
    UTF-8 BOM, (b) writes the declaration with single quotes, and (c) writes
    self-closing tags as `"/>` where Warp writes `" />`. We fix all three so an
    unedited parse round-trips byte-for-byte (verified by the round-trip test).
    """
    body = etree.tostring(tree, xml_declaration=False, encoding="utf-8")
    body = body.replace(b'"/>', b'" />')
    return XML_BOM + XML_DECL + body


def atomic_write_bytes(path, data):
    """Write bytes to a temp file in the same dir, then atomically replace."""
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def make_backup(src, backup_base):
    """Copy ``src`` to ``backup_base``; if that exists, try ``backup_base.1``,
    ``.2`` ... and use the first free name. Returns the backup path used."""
    candidate = backup_base
    i = 1
    while os.path.exists(candidate):
        candidate = f"{backup_base}.{i}"
        i += 1
    shutil.copy2(src, candidate)
    return candidate


# --------------------------------------------------------------------------- #
# Schema helpers -- everything is derived from the file, nothing hardcoded.
# --------------------------------------------------------------------------- #
def split_lines(text):
    """Split a per-tilt text block into its entries.

    Warp writes these with no leading/trailing newline, but we defensively drop
    a single trailing empty entry in case a file has one.
    """
    parts = (text or "").split("\n")
    if parts and parts[-1] == "":
        parts = parts[:-1]
    return parts


def get_n_tilts(root):
    """Number of tilts N == number of UseTilt entries."""
    ut = root.find("UseTilt")
    if ut is None or ut.text is None:
        return None
    return len(split_lines(ut.text))


def per_tilt_text_elems(root, n):
    """Top-level elements whose text is a newline list of length N.

    Enumerated by counting lines, not by a hardcoded name list. Elements with
    child nodes (grids) or an ID attribute (per-tilt indexed elements, handled
    separately) are excluded.
    """
    out = []
    for el in root:
        if len(el) > 0 or "ID" in el.attrib or el.text is None:
            continue
        if len(split_lines(el.text)) == n and n > 0:
            out.append(el)
    return out


def per_tilt_indexed(root, n):
    """Groups of sibling elements carrying an ID attribute of 0..N-1
    (e.g. <TiltPS1D ID="k">, <TiltSimulatedScale ID="k">)."""
    groups = defaultdict(list)
    for el in root:
        if "ID" in el.attrib:
            groups[el.tag].append(el)
    result = {}
    for tag, els in groups.items():
        if len(els) == n:
            ids = sorted(int(e.get("ID")) for e in els)
            if ids == list(range(n)):
                result[tag] = els
    return result


def per_tilt_grids(root, n):
    """Grid elements (Width/Height/Depth + <Node>) whose Depth == N, i.e. Z
    indexes tilts. Grids with Depth != N are global and left untouched."""
    out = []
    for el in root:
        depth = el.get("Depth")
        if depth is not None and el.find("Node") is not None and int(depth) == n:
            out.append(el)
    return out


def parse_dose(root):
    """Per-tilt accumulated dose as floats.

    NOTE: the original code did ``root.findall('Dose')`` and then
    ``float(elem.text.strip())`` -- but <Dose> is a single newline-separated
    block, so that raised ValueError and the --max-tilt mode never worked.
    Here we split the block correctly.
    """
    dose_el = root.find("Dose")
    return [float(x) for x in split_lines(dose_el.text)]


# --------------------------------------------------------------------------- #
# Selection: one keep-mask consumed by both the flip and delete paths.
# --------------------------------------------------------------------------- #
def compute_keep_mask(root, views_in_log, tilts_in_log, all_true, max_tilt):
    """Return a list[bool] of length N; True == keep this tilt.

    View numbers in taSolution.log are 1-based (tilt index i -> view i+1).
    """
    n = get_n_tilts(root)
    if all_true:
        return [True] * n
    if max_tilt == 0:
        return [(i + 1) in views_in_log for i in range(n)]
    # max_tilt > 0
    dose = parse_dose(root)
    min_dose_idx = dose.index(min(dose))
    threshold = abs(tilts_in_log[min_dose_idx]) + max_tilt
    return [
        ((i + 1) in views_in_log) and (abs(tilts_in_log[i]) <= threshold)
        for i in range(n)
    ]


# --------------------------------------------------------------------------- #
# Log reading
# --------------------------------------------------------------------------- #
def read_log(log_path):
    """Return (views_in_log:set, tilts_in_log:list) from a taSolution.log, or
    None if the file is missing / has no data table."""
    if not os.path.exists(log_path):
        return None
    with open(log_path, "r") as f:
        lines = f.readlines()
    view_line_idx = next(
        (idx for idx, line in enumerate(lines) if "view" in line), None
    )
    if view_line_idx is None:
        return None
    data_str = "".join(lines[view_line_idx:])
    df = pd.read_csv(io.StringIO(data_str), sep=r"\s+")
    return set(df["view"].unique()), df["tilt"].tolist()


# --------------------------------------------------------------------------- #
# Structure validation (pre-edit abort and post-edit assertion)
# --------------------------------------------------------------------------- #
def validate_structure(root, n):
    """Check every per-tilt structure is consistent at count ``n``.

    Returns a list of error strings; empty means the file is safe to edit.
    Aborting on any mismatch guarantees we never write a partially-edited file.
    """
    errors = []
    for el in root:
        if len(el) > 0 or "ID" in el.attrib or el.text is None:
            continue
        lines = split_lines(el.text)
        if len(lines) > 1 and len(lines) != n:
            errors.append(f"<{el.tag}> has {len(lines)} entries, expected {n}")

    # Indexed groups (TiltPS1D, TiltSimulatedScale, ...) must be exactly 0..N-1.
    groups = defaultdict(list)
    for el in root:
        if "ID" in el.attrib:
            groups[el.tag].append(el)
    for tag, els in groups.items():
        ids = sorted(int(e.get("ID")) for e in els)
        if ids != list(range(n)):
            errors.append(
                f"<{tag}> ID set is {ids[:3]}...({len(ids)}), expected 0..{n - 1}"
            )

    # Per-tilt grids must have W*H*N nodes and Z == 0..N-1.
    for grid in per_tilt_grids(root, n):
        w = int(grid.get("Width"))
        h = int(grid.get("Height"))
        nodes = grid.findall("Node")
        if len(nodes) != w * h * n:
            errors.append(
                f"<{grid.tag}> has {len(nodes)} nodes, expected W*H*Depth={w * h * n}"
            )
        zs = sorted(set(int(nd.get("Z")) for nd in nodes))
        if zs != list(range(n)):
            errors.append(f"<{grid.tag}> Z range is not 0..{n - 1}")
    return errors


def assert_after_delete(root, m):
    """Re-check every per-tilt list length and every grid Depth/Z range against
    the surviving tilt count M. Raises AssertionError on any corruption."""
    for el in per_tilt_text_elems(root, m):
        assert len(split_lines(el.text)) == m, f"<{el.tag}> length != {m} after edit"
    for tag, els in per_tilt_indexed(root, m).items():
        ids = sorted(int(e.get("ID")) for e in els)
        assert ids == list(range(m)), f"<{tag}> IDs not 0..{m - 1} after edit"
    for grid in per_tilt_grids(root, m):
        w = int(grid.get("Width"))
        h = int(grid.get("Height"))
        assert int(grid.get("Depth")) == m, f"<{grid.tag}> Depth != {m}"
        nodes = grid.findall("Node")
        assert len(nodes) == w * h * m, f"<{grid.tag}> node count != W*H*{m}"
        zs = sorted(set(int(nd.get("Z")) for nd in nodes))
        assert zs == list(range(m)), f"<{grid.tag}> Z not 0..{m - 1} after edit"


# --------------------------------------------------------------------------- #
# The delete edit (in memory)
# --------------------------------------------------------------------------- #
def delete_tilts_in_xml(root, n, deleted):
    """Physically remove ``deleted`` (a set of 0-based tilt indices) from every
    per-tilt structure, renumbering grid Z / indexed IDs to be contiguous.
    Returns M, the number of surviving tilts."""
    keep = [i for i in range(n) if i not in deleted]
    m = len(keep)
    zmap = {old: new for new, old in enumerate(keep)}

    # Gather targets up front (before we start mutating the tree).
    text_elems = per_tilt_text_elems(root, n)
    indexed = per_tilt_indexed(root, n)
    grids = per_tilt_grids(root, n)

    # 1. Per-tilt text lists: keep surviving entries, preserving formatting.
    for el in text_elems:
        lines = split_lines(el.text)
        el.text = "\n".join(lines[i] for i in keep)

    # 2. Indexed elements: drop deleted, renumber surviving IDs 0..M-1.
    for els in indexed.values():
        els_sorted = sorted(els, key=lambda e: int(e.get("ID")))
        survivors = []
        for e in els_sorted:
            if int(e.get("ID")) in deleted:
                root.remove(e)
            else:
                survivors.append(e)
        for new_id, e in enumerate(survivors):
            e.set("ID", str(new_id))

    # 3. Per-tilt grids: drop deleted Z-slices, renumber Z, fix Depth + tails.
    for grid in grids:
        nodes = grid.findall("Node")
        inner_tail = nodes[0].tail if len(nodes) >= 2 else None
        close_tail = nodes[-1].tail
        survivors = []
        for nd in nodes:
            if int(nd.get("Z")) in deleted:
                grid.remove(nd)
            else:
                survivors.append(nd)
        for nd in survivors:
            nd.set("Z", str(zmap[int(nd.get("Z"))]))
        for i, nd in enumerate(survivors):
            if i == len(survivors) - 1:
                nd.tail = close_tail
            else:
                nd.tail = inner_tail if inner_tail is not None else close_tail
        grid.set("Depth", str(m))

    return m


def movie_basenames(root, n):
    """basename() of each per-tilt MoviePath entry (tilt index -> basename)."""
    mp = root.find("MoviePath")
    lines = split_lines(mp.text)
    return [os.path.basename(p.strip()) for p in lines]


# --------------------------------------------------------------------------- #
# tomostar (STAR loop_ block) editing -- minimal, format-preserving.
# --------------------------------------------------------------------------- #
def parse_tomostar(path):
    """Return (lines, cols, movie_col, data_idx) for a .tomostar file.

    ``lines`` is the file split on '\\n' (verbatim). ``data_idx`` are the line
    indices of the loop_ data rows. Non-data lines are preserved untouched.
    """
    raw = open(path, "r", encoding="utf-8").read()
    lines = raw.split("\n")
    loop_i = next(i for i, l in enumerate(lines) if l.strip() == "loop_")
    cols = []
    last_decl = loop_i
    j = loop_i + 1
    while j < len(lines) and lines[j].strip().startswith("_"):
        cols.append(lines[j].strip().split()[0])
        last_decl = j
        j += 1
    data_idx = []
    k = last_decl + 1
    while k < len(lines):
        if lines[k].strip() == "":
            if data_idx:  # a blank line after data marks the end of the block
                break
            k += 1
            continue
        data_idx.append(k)
        k += 1
    movie_col = cols.index("_wrpMovieName")
    return lines, cols, movie_col, data_idx


def delete_tomostar_rows(path, deleted, xml_basenames, n):
    """Compute the new tomostar text with the deleted tilts' rows removed.

    Rows are matched to XML tilts by movie basename; falls back to row order
    (with a warning) only if names can't be matched one-to-one. Returns
    (new_text, warning_or_None, surviving_row_count) or raises ValueError if the
    row count does not match N.
    """
    lines, cols, movie_col, data_idx = parse_tomostar(path)
    if len(data_idx) != n:
        raise ValueError(
            f"{os.path.basename(path)}: {len(data_idx)} rows != {n} XML tilts"
        )

    row_bn = [os.path.basename(lines[i].split()[movie_col]) for i in data_idx]
    names_ok = set(row_bn) == set(xml_basenames) and len(set(row_bn)) == len(row_bn)

    warning = None
    if names_ok:
        deleted_movies = {xml_basenames[i] for i in deleted}
        del_positions = {p for p, bn in enumerate(row_bn) if bn in deleted_movies}
    else:
        warning = (
            "movie names could not be matched to tomostar rows; "
            "falling back to row order"
        )
        del_positions = set(deleted)

    drop_line_idx = {data_idx[p] for p in del_positions}
    new_lines = [l for i, l in enumerate(lines) if i not in drop_line_idx]
    return "\n".join(new_lines), warning, len(data_idx) - len(del_positions)


# --------------------------------------------------------------------------- #
# Per-file handlers
# --------------------------------------------------------------------------- #
def describe_deletion(root, deleted, tilts_in_log):
    """Human-readable per-tilt description for --dry-run / logging."""
    bns = movie_basenames(root, get_n_tilts(root))
    angles = split_lines(root.find("Angles").text)
    rows = []
    for i in sorted(deleted):
        rows.append(f"    view {i + 1:>3}  tilt {angles[i]:>8}  {bns[i]}")
    return rows


def handle_flip(xml_file, backup_dir, mask, dry_run):
    """Existing behaviour: rewrite UseTilt to True/False from the keep-mask."""
    tree = parse_xml(xml_file)
    root = tree.getroot()
    current = split_lines(root.find("UseTilt").text)
    updated = ["True" if keep else "False" for keep in mask]
    changes = sum(1 for a, b in zip(current, updated) if a != b)

    if dry_run:
        click.echo(f"{xml_file}: [dry-run] would change {changes} UseTilt values.")
        return

    backup = make_backup(xml_file, os.path.join(backup_dir, os.path.basename(xml_file)))
    root.find("UseTilt").text = "\n".join(updated)
    atomic_write_bytes(xml_file, serialize_xml(tree))
    click.echo(f"{xml_file}: {changes} changes made to UseTilt (backup: {backup}).")


def handle_delete(xml_file, tomostar_dir, mask, tilts_in_log, dry_run):
    """Physically remove excluded tilts from the XML and its .tomostar."""
    basename = os.path.splitext(os.path.basename(xml_file))[0]
    tomostar_file = os.path.join(tomostar_dir, basename + ".tomostar")

    tree = parse_xml(xml_file)
    root = tree.getroot()
    n = get_n_tilts(root)
    deleted = {i for i, keep in enumerate(mask) if not keep}

    if not deleted:
        click.echo(f"{xml_file}: nothing to delete (all {n} tilts kept).")
        return

    # Pre-flight: abort this file untouched if any per-tilt structure != N.
    errors = validate_structure(root, n)
    if errors:
        click.echo(f"{xml_file}: ABORT (structure inconsistent, left untouched):")
        for e in errors:
            click.echo(f"    - {e}")
        return

    if not os.path.exists(tomostar_file):
        click.echo(f"{xml_file}: tomostar {tomostar_file} not found, skipping.")
        return

    xml_bns = movie_basenames(root, n)

    # Parse + edit + validate BOTH in memory before writing anything.
    try:
        new_tomostar, warning, remaining_rows = delete_tomostar_rows(
            tomostar_file, deleted, xml_bns, n
        )
    except ValueError as exc:
        click.echo(f"{xml_file}: ABORT ({exc}).")
        return

    if dry_run:
        click.echo(
            f"{xml_file}: [dry-run] would delete {len(deleted)} of {n} tilts, "
            f"{n - len(deleted)} remain:"
        )
        for line in describe_deletion(root, deleted, tilts_in_log):
            click.echo(line)
        if warning:
            click.echo(f"    WARNING: {warning}")
        return

    m = delete_tilts_in_xml(root, n, deleted)
    assert_after_delete(root, m)
    xml_bytes = serialize_xml(tree)

    # Both artifacts are ready; back up in place (versioned) and write atomically.
    xml_bak = make_backup(xml_file, xml_file + ".bak")
    tomostar_bak = make_backup(tomostar_file, tomostar_file + ".bak")
    atomic_write_bytes(xml_file, xml_bytes)
    atomic_write_bytes(tomostar_file, new_tomostar.encode("utf-8"))

    click.echo(
        f"{xml_file}: deleted {len(deleted)} tilts, {m} remain "
        f"(backups: {xml_bak}, {tomostar_bak})."
    )
    if warning:
        click.echo(f"    WARNING: {warning}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
@click.command()
@click.option("--xml-dir", default="./", help="Directory containing XML files")
@click.option("--xml-pattern", default="*.xml", help="Pattern to match XML files")
@click.option("--backup-dir", default="backup_xml", help="Directory to store XML backups (flip mode)")
@click.option("--tiltstack-dir", default="tiltstack", help="Base directory for tiltstack logs")
@click.option("--tomostar-dir", default="../tomostar", help="Directory containing .tomostar files (delete mode)")
@click.option("--all-true", is_flag=True, default=False, help="Set all UseTilt values to True")
@click.option("--max-tilt", default=0, help="Maximum tilt angle (from the minimum dose) to keep, others discarded")
@click.option("--delete", "do_delete", is_flag=True, default=False, help="Physically remove excluded tilts from the XML and .tomostar instead of setting UseTilt=False")
@click.option("--dry-run", is_flag=True, default=False, help="Report what would change and write nothing")
def main(xml_dir, xml_pattern, backup_dir, tiltstack_dir, tomostar_dir, all_true,
         max_tilt, do_delete, dry_run):
    """Process Warp tilt-series XML files: update UseTilt values based on
    taSolution.log, or (with --delete) physically remove excluded tilts from
    both the XML and the matching .tomostar file.

    Example:
        python remove_skipped_view.py --xml-dir ./ --xml-pattern '*.xml' \\
            --backup-dir backup_xml --tiltstack-dir tiltstack
    """
    if do_delete and all_true:
        raise click.UsageError("--delete with --all-true is a contradiction (deletes nothing).")

    xml_files = glob.glob(os.path.join(xml_dir, xml_pattern))
    if not xml_files:
        click.echo("No XML files found.")
        return

    if not (do_delete and dry_run):
        os.makedirs(backup_dir, exist_ok=True)

    if do_delete:
        click.echo(
            "!! DELETE MODE: this removes tilts from the XML and .tomostar and is "
            "NOT idempotent.\n"
            "!! Once tilts are removed, taSolution.log view numbers no longer line "
            "up with the XML rows,\n"
            "!! so re-running would delete the WRONG tilts. Keep the .bak files, and "
            "restore from them\n"
            "!! before re-running on the same tilt series. Downstream WarpTools steps "
            "(ts_ctf,\n"
            "!! ts_aretomo, ts_reconstruct) must be re-run because the tilt count "
            "changed."
        )

    for xml_file in xml_files:
        log_path = os.path.join(
            tiltstack_dir, os.path.splitext(os.path.basename(xml_file))[0], "taSolution.log"
        )
        views_in_log, tilts_in_log = set(), []
        if not all_true:
            log = read_log(log_path)
            if log is None:
                click.echo(f"Log file {log_path} not found or empty, skipping {xml_file}.")
                continue
            views_in_log, tilts_in_log = log

        tree = parse_xml(xml_file)
        root = tree.getroot()
        mask = compute_keep_mask(root, views_in_log, tilts_in_log, all_true, max_tilt)

        if do_delete:
            handle_delete(xml_file, tomostar_dir, mask, tilts_in_log, dry_run)
        else:
            handle_flip(xml_file, backup_dir, mask, dry_run)


if __name__ == "__main__":
    main()
