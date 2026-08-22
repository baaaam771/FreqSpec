#!/usr/bin/env python3
# =============================================================================
#  F4 : qualitative grid  (RUN ON THE SERVER — needs the generated images)
#
#  Rows = story-telling example images chosen from the per-image CSVs:
#    - large-mask cases where s16 is fine but s8 COLLAPSES
#    - an instance case where s8 fails, and one where it does NOT (heterogeneity)
#  Columns = input | mask | s50 | s16 | s12 | s8 | FreqSpec-strict | default
#            (+ optional usage map)
#
#  You MUST set the path patterns below to match how your images are stored.
#  Missing images render as a labeled gray box showing the path tried, so you
#  can fix the pattern and re-run. Cells are annotated with GT-m (from CSV).
#
#  Usage:  python make_f4_qualitative.py --data . --img_root /path/to/images
#  Output: fig_f4_qualitative.pdf/.png
# =============================================================================
import argparse, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---- EXAMPLES (run, idx, short row label) : chosen from the CSVs ------------
EXAMPLES = [
    ("largemask_coco",     32, "large / COCO"),
    ("largemask_ffhq",     27, "large / FFHQ"),
    ("largemask_places2",  44, "large / Places2"),
    ("objectremoval_coco", 17, "instance (s8 fails)"),
    ("objectremoval_coco", 14, "instance (s8 ok)"),
]

# ---- COLUMNS : (header, kind, method) --------------------------------------
#   kind 'input'/'mask' use the per-image patterns; 'gen'/'usage' use method.
COLUMNS = [
    ("Input",   "input", None),
    ("Mask",    "mask",  None),
    ("s50",     "gen",   "target_s50"),
    ("s16 (3x)","gen",   "target_s16"),
    ("s12",     "gen",   "target_s12"),
    ("s8 (6x)", "gen",   "target_s8"),
    ("strict",  "gen",   "freqspec_strict"),
    ("default", "gen",   "freqspec_default"),
    # ("usage",  "usage", "freqspec_default"),   # uncomment if you saved usage maps
]

# ====== EDIT THESE PATH PATTERNS TO MATCH YOUR STORAGE =======================
# available placeholders: {root} {run} {idx} {idx04} {method}
INPUT_PATTERN = "{root}/{run}/input_{idx04}.png"
MASK_PATTERN  = "{root}/{run}/mask_{idx04}.png"
GEN_PATTERN   = "{root}/{run}/{method}/{idx04}.png"
USAGE_PATTERN = "{root}/{run}/{method}/usage_{idx04}.png"
# ============================================================================

def load_gm(data_dir):
    fr=[]
    for f in ["largemask_per_image.csv","instmask_per_image.csv","phase2_per_image.csv"]:
        p=os.path.join(data_dir,f)
        if os.path.exists(p): fr.append(pd.read_csv(p))
    if not fr: return None
    df=pd.concat(fr, ignore_index=True)
    return df.set_index(["run","method","idx"])["gt_masked_lpips"]

def resolve(kind, run, idx, method, root):
    d=dict(root=root, run=run, idx=idx, idx04=f"{idx:04d}", method=method)
    pat={"input":INPUT_PATTERN,"mask":MASK_PATTERN,"gen":GEN_PATTERN,"usage":USAGE_PATTERN}[kind]
    return pat.format(**d)

def load_img(path):
    try:
        import matplotlib.image as mpimg
        return mpimg.imread(path)
    except Exception:
        return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", default=".")
    ap.add_argument("--img_root", default="./samples", help="root dir of generated images")
    args=ap.parse_args()
    gm=load_gm(args.data)

    for fam in ("Times New Roman","Nimbus Roman","DejaVu Serif"):
        if any(fam==f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"]="serif"; plt.rcParams["font.serif"]=[fam]; break

    R, C = len(EXAMPLES), len(COLUMNS)
    fig, axes = plt.subplots(R, C, figsize=(1.35*C, 1.45*R))
    if R==1: axes=axes[None,:]
    missing=[]
    for r,(run,idx,rlab) in enumerate(EXAMPLES):
        for c,(hdr,kind,method) in enumerate(COLUMNS):
            ax=axes[r,c]; ax.set_xticks([]); ax.set_yticks([])
            path=resolve(kind,run,idx,method,args.img_root)
            im=load_img(path)
            if im is None:
                missing.append(path)
                ax.imshow(np.ones((10,10,3))*0.85)
                ax.text(0.5,0.5,os.path.basename(path),fontsize=4,ha="center",va="center",
                        transform=ax.transAxes,color="0.4",wrap=True)
            else:
                ax.imshow(im)
                # annotate GT-m for gen columns
                if kind=="gen" and gm is not None:
                    try:
                        v=gm.loc[(run,method,idx)]
                        ax.text(0.03,0.05,f"{v:.3f}",fontsize=6,color="white",
                                transform=ax.transAxes,
                                bbox=dict(fc="black",alpha=0.5,pad=1,lw=0))
                    except Exception: pass
            if r==0: ax.set_title(hdr, fontsize=8)
            if c==0: ax.set_ylabel(rlab, fontsize=7, rotation=90, labelpad=2)
    fig.suptitle("Qualitative comparison: reduced-step collapse at s8 and FreqSpec presets "
                 "(cell = masked-region LPIPS)", fontsize=9, y=0.995)
    fig.tight_layout(rect=(0,0,1,0.98))
    fig.savefig("fig_f4_qualitative.pdf", bbox_inches="tight")
    fig.savefig("fig_f4_qualitative.png", dpi=150, bbox_inches="tight")
    print("wrote fig_f4_qualitative.pdf/.png")
    if missing:
        print(f"\n[!] {len(missing)} images not found — fix the *_PATTERN at top. Examples:")
        for m in missing[:6]: print("   ", m)
    else:
        print("all images found.")

if __name__=="__main__":
    main()
