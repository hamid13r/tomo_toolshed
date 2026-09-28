#!/usr/bin/env python3
"""Replace the placeholder ``_rlnDefocus`` values in an IsoNet star file with
the per-tomogram average defocus read from the corresponding Warp XML file.

IsoNet's ``isonet.py prepare_star`` writes a star file with a placeholder
``_rlnDefocus`` column. Each ``_rlnTomoName`` there points at a reconstruction
(e.g. ``rec/Position_2_2.mrc_10.00Apx.mrc``); the matching Warp XML
(``xml/Position_2_2.mrc.xml``) holds the per-tilt CTF estimate in its
``GridCTF`` node. This module averages those per-tilt values (converted from
microns to Angstroms) and writes the result back into the star file.
"""
import os
import re
import xml.etree.ElementTree as ET

import numpy as np

#: ``_rlnTomoName`` values look like ``rec/<base>.mrc_<pixelsize>Apx.mrc``;
#: group 1 is the base name used to locate the XML file.
DEFAULT_TOMONAME_REGEX = r"^(?:.*/)?(.+)\.mrc_[\d.]+Apx\.mrc$"

#: Appended to the base name to form the Warp XML file name.
DEFAULT_XML_SUFFIX = ".mrc.xml"


class StarFormatError(ValueError):
    """Raised when the star file has no usable ``_rln`` header block."""


def defocus_values(xml_path):
    """Return the per-tilt defocus values from a Warp XML file, in Angstroms.

    Values in ``GridCTF`` are in microns; they are scaled by 10000 and
    truncated to whole Angstroms, matching Warp/RELION conventions.
    """
    root = ET.parse(xml_path).getroot()
    grid_ctf = root.find("GridCTF")
    if grid_ctf is None:
        raise ValueError(f"No <GridCTF> node in {xml_path}")

    values = [
        np.int32(np.float64(node.get("Value")) * 10000)
        for node in grid_ctf.findall("Node")
    ]
    if not values:
        raise ValueError(f"No <GridCTF><Node> entries in {xml_path}")

    return np.array(values, dtype=float)


def average_defocus(xml_path):
    """Return the mean per-tilt defocus of a Warp XML file, in Angstroms."""
    return defocus_values(xml_path).mean()


def parse_star_header(lines):
    """Return ``(column_names, first_data_line_index)`` for a star file.

    Column names are the ``_rln*`` labels in order, stripped of their ``#n``
    suffix.
    """
    columns = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("_rln"):
            columns.append(stripped.split("#")[0].strip())
        elif columns and stripped:
            return columns, i

    raise StarFormatError("No _rln header block followed by data rows found")


def xml_path_for(tomoname, xml_dir, tomoname_regex=DEFAULT_TOMONAME_REGEX,
                 xml_suffix=DEFAULT_XML_SUFFIX):
    """Map a ``_rlnTomoName`` value to its Warp XML path.

    Returns ``None`` if the name does not match ``tomoname_regex``.
    """
    match = re.match(tomoname_regex, tomoname)
    if not match:
        return None
    return os.path.join(xml_dir, match.group(1) + xml_suffix)


def update_star_defocus(star_file, xml_dir, output=None, strict=False,
                        tomoname_regex=DEFAULT_TOMONAME_REGEX,
                        xml_suffix=DEFAULT_XML_SUFFIX, dry_run=False,
                        log=None):
    """Fill in ``_rlnDefocus`` in ``star_file`` from the XMLs in ``xml_dir``.

    Writes to ``output`` when given, otherwise back to ``star_file`` in place.
    Rows whose tomogram name cannot be parsed, or whose XML is missing or
    unreadable, are left untouched and reported -- unless ``strict``, in which
    case the first such row raises and nothing is written.

    ``log`` is an optional ``callable(message)`` for progress output.

    Returns ``(updated, skipped)`` where ``updated`` is a list of
    ``(tomoname, defocus)`` and ``skipped`` a list of ``(tomoname, reason)``.
    """
    def note(message):
        if log is not None:
            log(message)

    with open(star_file) as handle:
        lines = handle.readlines()

    columns, header_end = parse_star_header(lines)
    for required in ("_rlnTomoName", "_rlnDefocus"):
        if required not in columns:
            raise StarFormatError(f"{star_file} has no {required} column")
    tomoname_idx = columns.index("_rlnTomoName")
    defocus_idx = columns.index("_rlnDefocus")

    updated, skipped = [], []

    for i in range(header_end, len(lines)):
        line = lines[i]
        if not line.strip():
            continue

        raw = line.rstrip("\n")
        separator = "\t" if "\t" in raw else " "
        cols = raw.split()
        if len(cols) <= max(tomoname_idx, defocus_idx):
            reason = f"row {i + 1} has {len(cols)} columns, expected {len(columns)}"
            if strict:
                raise StarFormatError(reason)
            skipped.append(("", reason))
            note(f"skipping: {reason}")
            continue

        tomoname = cols[tomoname_idx]
        xml_path = xml_path_for(tomoname, xml_dir, tomoname_regex, xml_suffix)

        if xml_path is None:
            reason = f"could not parse tomogram name: {tomoname}"
        elif not os.path.exists(xml_path):
            reason = f"missing XML file: {xml_path}"
        else:
            reason = None

        if reason is None:
            try:
                mean_defocus = average_defocus(xml_path)
            except (ET.ParseError, ValueError, TypeError) as exc:
                reason = f"could not read defocus from {xml_path}: {exc}"

        if reason is not None:
            if strict:
                raise ValueError(reason)
            skipped.append((tomoname, reason))
            note(f"skipping {tomoname}: {reason}")
            continue

        cols[defocus_idx] = str(int(round(mean_defocus)))
        lines[i] = separator.join(cols) + "\n"
        updated.append((tomoname, int(round(mean_defocus))))
        note(f"{tomoname}: defocus {cols[defocus_idx]} A (from {xml_path})")

    if not dry_run:
        destination = output or star_file
        with open(destination, "w") as handle:
            handle.writelines(lines)

    return updated, skipped
