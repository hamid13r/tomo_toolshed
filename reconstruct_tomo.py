#!/usr/bin/env python
"""
Command-line front-end, mirroring WarpTools `ts_reconstruct`.

Example
-------
    python reconstruct_tomo.py \
        --xml   warp_tiltseries/TS_01.xml \
        --tomostar warp_tiltseries/tomostar/TS_01.tomostar \
        --tilt_stack aligned_stacks/TS_01.st \
        --angpix 10 \
        --output reconstruction \
        --deconv

Tilt-image input (choose one):
    --tilt_stack FILE   a 3D MRC/ST stack, slices ordered like the model's tilts
    --tilt_dir  DIR     folder of per-tilt averages named <movie-root>.mrc
                        (matched to the .xml MoviePath entries)

If --raw_angpix is omitted it is read from the tilt-image MRC header.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import numpy as np

import numpy as np
from warp_recon import load_tiltseries_xml, reconstruct, ReconOptions
from warp_recon import warp_weighting, make_dose_bfactor_weighting
from warp_recon.mrc_io import read_mrc, write_mrc, write_png_slice
from warp_recon.metadata import read_settings_pixelsize, read_settings_tomo_dims


def load_tilt_images(model, args):
    if args.tilt_stack:
        data, vs = read_mrc(args.tilt_stack)
        if data.ndim != 3:
            raise SystemExit("--tilt_stack must be a 3D stack")
        if data.shape[0] != model.n_tilts:
            raise SystemExit(
                f"stack has {data.shape[0]} slices but model has {model.n_tilts} tilts")
        imgs = [data[i] for i in range(data.shape[0])]
        return imgs, vs
    if args.tilt_dir:
        d = Path(args.tilt_dir)
        imgs, vs = [], None
        for mp in model.movie_paths:
            root = Path(mp).stem
            cand = list(d.glob(root + ".mrc")) + list(d.glob(root + ".*"))
            if not cand:
                raise SystemExit(f"could not find tilt image for '{root}' in {d}")
            im, v = read_mrc(cand[0])
            if im.ndim == 3:
                im = im[0]
            imgs.append(im)
            vs = vs or v
        return imgs, vs
    raise SystemExit("provide --tilt_stack or --tilt_dir")


def main():
    ap = argparse.ArgumentParser(description="Faithful ts_reconstruct re-implementation")
    ap.add_argument("--xml", required=True, help="per-tilt-series .xml (Warp metadata)")
    ap.add_argument("--tomostar", help=".tomostar (fallback for raw angles/dose)")
    ap.add_argument("--settings", help="Warp .settings (for raw pixel size)")
    ap.add_argument("--angpix", type=float, required=True,
                    help="OUTPUT tomogram pixel size (A) -- this is the knob that "
                         "sets the tomogram sampling, same as ts_reconstruct --angpix")
    ap.add_argument("--raw_angpix", type=float, help="unbinned tilt pixel size (A)")
    ap.add_argument("--dimensions", type=int, nargs=3, metavar=("X", "Y", "Z"),
                    help="tomogram box in UNBINNED pixels (like Warp's <Tomo> "
                         "Dimensions). Overrides the .settings / .xml box.")
    ap.add_argument("--tilt_stack", help="3D MRC/ST stack ordered like the tilts")
    ap.add_argument("--tilt_dir", help="folder of per-tilt averages <root>.mrc")
    ap.add_argument("--output", default="reconstruction", help="output folder")

    ap.add_argument("--deconv", action="store_true")
    ap.add_argument("--deconv_strength", type=float, default=1.0)
    ap.add_argument("--deconv_falloff", type=float, default=1.0)
    ap.add_argument("--deconv_highpass", type=float, default=300.0)
    ap.add_argument("--dont_invert", action="store_true")
    ap.add_argument("--dont_normalize", action="store_true")
    ap.add_argument("--subvolume_size", type=int, default=64)
    ap.add_argument("--subvolume_padding", type=float, default=3.0)
    ap.add_argument("--pad_factor", type=float, default=1.0,
                    help="reconstruction-cube padding (global engine)")
    ap.add_argument("--float16", action="store_true", help="write 16-bit MRC like Warp")

    # experimentation hooks
    ap.add_argument("--dose_bfactor_scale", type=float, default=None,
                    help="use a custom dose->Bfactor factor instead of Warp's 4")
    args = ap.parse_args()

    model = load_tiltseries_xml(args.xml, args.tomostar)
    tilt_images, header_angpix = load_tilt_images(model, args)

    raw_angpix = args.raw_angpix
    if raw_angpix is None and args.settings:
        raw_angpix = read_settings_pixelsize(args.settings)
    if raw_angpix is None:
        raw_angpix = header_angpix
    if not raw_angpix or raw_angpix <= 0:
        raise SystemExit("could not determine raw pixel size; pass --raw_angpix")
    print(f"raw pixel size: {raw_angpix:.3f} A  ->  output tomogram pixel size "
          f"{args.angpix:.3f} A")

    # --- tomogram box selection ---------------------------------------------
    # priority: --dimensions  >  settings <Tomo> Dimensions  >  .xml value
    box_src = "xml"
    if args.dimensions is not None:
        dx, dy, dz = args.dimensions
        model.volume_dims_A = np.array([dx, dy, dz], np.float32) * raw_angpix
        box_src = "--dimensions"
    elif args.settings:
        dims = read_settings_tomo_dims(args.settings)
        if dims is not None:
            model.volume_dims_A = np.array(dims, np.float32) * raw_angpix
            box_src = "settings <Tomo>"
    nv = (model.volume_dims_A / args.angpix)
    print(f"tomogram box: {model.volume_dims_A} A  (from {box_src})  ->  "
          f"~{int(round(nv[0]))}x{int(round(nv[1]))}x{int(round(nv[2]))} voxels "
          f"at {args.angpix} A")

    opts = ReconOptions(
        angpix=args.angpix, raw_angpix=raw_angpix,
        invert=not args.dont_invert, normalize=not args.dont_normalize,
        do_deconv=args.deconv, deconv_strength=args.deconv_strength,
        deconv_falloff=args.deconv_falloff, deconv_highpass=args.deconv_highpass,
        subvolume_size=args.subvolume_size, subvolume_padding=args.subvolume_padding,
        pad_factor=args.pad_factor, mode="global",
    )

    wfn = warp_weighting
    if args.dose_bfactor_scale is not None:
        wfn = make_dose_bfactor_weighting(args.dose_bfactor_scale)
        print(f"using custom dose->Bfactor scale = {args.dose_bfactor_scale}")

    outputs = reconstruct(model, tilt_images, opts, weighting_fn=wfn)

    os.makedirs(args.output, exist_ok=True)
    base = model.name
    rec_path = os.path.join(args.output, base + ".mrc")
    write_mrc(rec_path, outputs["reconstruction"], args.angpix, as_float16=args.float16)
    write_png_slice(os.path.join(args.output, base + ".png"), outputs["reconstruction"])
    print("wrote", rec_path)
    if "deconv" in outputs:
        dpath = os.path.join(args.output, base + "_deconv.mrc")
        write_mrc(dpath, outputs["deconv"], args.angpix, as_float16=args.float16)
        print("wrote", dpath)


if __name__ == "__main__":
    main()
