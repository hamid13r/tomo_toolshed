# HANDOFF — run `warp_recont_test` steps 4–6 on the remote machine

This picks up `AGENT_TASK.md`. Steps 1–3 are **done on the laptop**; steps 4–6 need
the **VLP3x3_p03_ts_002 frame-series averages**, which are not on the laptop. Run
the rest wherever those tilt images (and, for step 5, a GPU + WarpTools) live.

---

## 0. What is already done (on the laptop)

| Step | Status | Detail |
|------|--------|--------|
| 1. git repo | ✅ | `git init`, 1 commit (`aa1d8e3`), 15 code files tracked, **no data tracked**. No remote yet. |
| 2. env | ✅ | micromamba env `warp` (py3.11): numpy 2.4.6, scipy 1.17.1, mrcfile 1.5.4, pillow. |
| 3. smoke test | ✅ | `python validate.py` → **PASS**, correlation **0.904**. Geometry/normalization verified. |
| 4. real recon | ⏸ blocked | needs `VLP3x3_p03_ts_002_*_blended_frames.mrc` (41 tilts). Not on laptop. |
| 5. WarpTools compare | ⏸ | needs GPU + WarpTools. **Ask the user before running (GPU).** |
| 6. report | ⏸ | write `VALIDATION.md`. |

The package import name is `warp_recon`; the distribution/repo is `warp_recont_test`;
the installed CLI command is `warp-recont-test` (equivalent to `python reconstruct_tomo.py`).

---

## 1. Get the code + metadata onto the remote

The **code** is in git; the **metadata files** (`.xml`, `.tomostar`, `.settings`) are
git-ignored and must be copied separately (do NOT add them to git).

Option A — copy the whole working dir (simplest):
```bash
rsync -av --exclude _selftest --exclude __pycache__ --exclude '.git' \
  <laptop>:/Users/hamid/Documents/Claude/Projects/warptools_new_reconstruction/ \
  ~/warp_recont_test/
```
Option B — clone the repo, then copy just the 4 metadata files:
```bash
# after pushing the repo to a remote named warp_recont_test:
git clone <remote-url> warp_recont_test && cd warp_recont_test
scp <laptop>:.../VLP3x3_p03_ts_002_blended_frames.xml \
    <laptop>:.../VLP3x3_p03_ts_002_blended_frames.tomostar \
    <laptop>:.../warp_tiltseries.settings \
    <laptop>:.../warp_frameseries.settings .
```

You need these four in the working dir:
`VLP3x3_p03_ts_002_blended_frames.xml`, `VLP3x3_p03_ts_002_blended_frames.tomostar`,
`warp_tiltseries.settings`, `warp_frameseries.settings`.

---

## 2. Environment on the remote

```bash
cd ~/warp_recont_test
conda create -y -n warp python=3.11      # or micromamba/mamba; reuse existing `warp` if present
conda activate warp
pip install -r requirements.txt
pip install -e .
python -c "import numpy,scipy,mrcfile;print(numpy.__version__,scipy.__version__,mrcfile.__version__)"
```

## 3. Sanity check (no data)

```bash
python validate.py     # must print PASS, correlation ~0.9. If not, STOP — geometry bug.
```

---

## 4. Real reconstruction — the `--angpix` sweep

**Find the tilt images.** The `.xml` `MoviePath` entries are relative:
`../warp_frameseries/VLP3x3_p03_ts_002_-60.0026_blended_frames.mrc` … (41 tilts,
angles −59.0 … +59.5). `--tilt_dir` must be the folder that actually holds those
`*_blended_frames.mrc` files (in a standard Warp layout that is
`warp_frameseries/average/`). The CLI matches each MoviePath **stem** to
`<stem>.mrc` in `--tilt_dir`.

```bash
TILT_DIR=/path/to/warp_frameseries/average      # <-- set this to the real folder

for A in 20 14 10; do
  python reconstruct_tomo.py \
    --xml       VLP3x3_p03_ts_002_blended_frames.xml \
    --tomostar  VLP3x3_p03_ts_002_blended_frames.tomostar \
    --settings  warp_tiltseries.settings \
    --tilt_dir  "$TILT_DIR" \
    --angpix    $A \
    --output    recon_mine_$A --deconv
done
```

Tip: run `--angpix 20` **first** (lightest) to confirm filename matching + RAM before 14/10.
If it errors with `could not find tilt image for '...'`, the `--tilt_dir` is wrong or the
averages use a different name/extension.

### Expected results (verify these)

Raw pixel size = **1.058 Å** (from `warp_tiltseries.settings`). With `--settings`, the box
comes from `<Tomo>` = **11664 × 11664 × 2400** unbinned px = **12341 × 12341 × 2539 Å**.
Output box voxels = round(box_Å / angpix):

| --angpix | header voxel size | output box (voxels) | approx RAM |
|----------|-------------------|---------------------|-----------|
| 20 | 20.0 Å | ~617 × 617 × 127 | small |
| 14 | 14.0 Å | ~882 × 882 × 181 | ~2–3 GB |
| 10 | 10.0 Å | ~1234 × 1234 × 254 | ~8 GB |

Checks: (a) each output MRC header voxel size **== its `$A`**; (b) box voxels **scale as
box_Å / $A** (halving angpix ~doubles each axis); (c) larger `$A` uses less RAM/time.
If RAM-limited, raise `--angpix` (e.g. 24/28). To force a smaller box instead, pass
`--dimensions X Y Z` (unbinned px) — the `.xml` alternatively carries a smaller
`VolumeDimensionsAngstrom` (4334 × 6094 × 2539 Å) if a cropped box is wanted.

Read a header to verify:
```bash
python -c "import mrcfile,sys; m=mrcfile.open(sys.argv[1],permissive=True); \
print('shape',m.data.shape,'voxel',tuple(round(v,3) for v in m.voxel_size.tolist()))" \
recon_mine_10/VLP3x3_p03_ts_002_blended_frames.mrc
```
Outputs land in `recon_mine_$A/`: `<name>.mrc`, `<name>_deconv.mrc`, `<name>.png`.

---

## 5. Compare to WarpTools  ⚠️ NEEDS GPU — ask the user first

Only if GPU + WarpTools are available **and the user approves the GPU run**:

```bash
WarpTools ts_reconstruct --settings warp_tiltseries.settings --angpix 10
# subset to this series with --input_data if needed
```
Then compare `recon_mine_10` vs the Warp output:
- confirm same box + voxel size (resample/crop if off);
- real-space **Pearson correlation**;
- **FSC** curve;
- central-slice **PNG**s side by side;
- note any **axis flip** needed to align them.

Expect **high correlation, NOT bit-identical** (documented reasons: trilinear vs Warp's
gridding kernel; global vs per-subvolume CTF).

### If correlation is low / orientation wrong — diagnose, don't guess
Check, in order, which single knob is off — **do not flip signs until it matches**, and
check against the source refs cited in `README.md`:
1. tilt-axis sign (`AxisAngle`) — see `warp_recon/geometry.py:tilt_matrix`
2. `AreAnglesInverted` handling — `geometry.py:positions_in_all_tilts`
3. image/volume centering — `img_center` / `vol_center` in `geometry.py`, patch extraction in `reconstruct.py:_extract_patch`
4. defocus sign — `warp_recon/ctf.py` (`deltaf = -(Defocus_um*1e4)`)
5. `--angpix` / box mismatch — box selection in `reconstruct_tomo.py`

---

## 6. Report — write `VALIDATION.md`

`VALIDATION.md` is git-ignored (keep data out of git). Include:
- env: numpy/scipy/mrcfile versions;
- `validate.py` result (PASS + correlation);
- per-`angpix` table: header voxel size, box voxels, wall-clock time, peak RAM;
- if step 5 ran: Warp-vs-mine Pearson r, FSC (e.g. 0.143/0.5 resolution), central-slice PNGs, any flip applied;
- notes on any diagnosis from §5.

---

## Key facts (from this series' metadata)

- Series: `VLP3x3_p03_ts_002`, **41 tilts**, angles −59.0 … +59.5°.
- Raw/unbinned pixel size: **1.058 Å** (`warp_tiltseries.settings`).
- `<Tomo>` box: 11664 × 11664 × 2400 unbinned px (12341 × 12341 × 2539 Å).
- `.xml VolumeDimensionsAngstrom`: 4334 × 6094 × 2539 Å (smaller; used only if no `--settings`/`--dimensions`).
- Tilt images referenced: `../warp_frameseries/VLP3x3_p03_ts_002_<angle>_blended_frames.mrc`.
- Do **NOT** commit any `*.mrc *.st *.settings *.star *.tomostar *.xml` or `recon_mine*/`,
  `VALIDATION.md`, `_selftest/` — already covered by `.gitignore`.
