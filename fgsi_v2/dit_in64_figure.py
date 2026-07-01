#!/usr/bin/env python
r"""
dit_in64_figure.py -- assemble the ImageNet-64 token-mixing figure (paper Figure 2)
from the four sample grids produced by dit_token_sampler.py.

Reads grid_target.png, grid_draft.png, grid_freqspec.png, grid_random.png from a
mixing result folder, optionally crops each to its top-left NxN cells for larger
samples, lays them out in one row with column titles and FID annotations, and
writes a Type-3-free PDF ready for \includegraphics.

Usage:
    python dit_in64_figure.py \
        --grid_dir /mnt/HDD_12TB/bam_ki/results/dit_in64/mixing_ar0.7 \
        --out figures/in64_mixing.pdf \
        --crop 5 --cell 66 \
        --fid_target 42.49 --fid_draft 93.12 --fid_freqspec 62.06 --fid_random 79.84

--crop N       show only the top-left N x N samples (0 = full grid)
--cell PX      pixel size of one grid cell incl. padding (make_grid: img+2*pad).
               For 64px images with padding 2 this is 66. Used only when --crop>0.
Then convert Type-3 fonts:
    gs -o figures/in64_mixing_out.pdf -dNoOutputFonts -sDEVICE=pdfwrite figures/in64_mixing.pdf
    mv figures/in64_mixing_out.pdf figures/in64_mixing.pdf
(the script already runs this if ghostscript `gs` is on PATH)
"""
import argparse
import os
import shutil
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def load_panel(path, crop, cell):
    im = Image.open(path).convert("RGB")
    if crop and crop > 0:
        s = crop * cell
        im = im.crop((0, 0, min(s, im.width), min(s, im.height)))
    return im


def main(a):
    names = ["target", "draft", "freqspec", "random"]
    titles = ["Target-only\n(DiT-S)", "Draft-only\n(DiT-Nano)",
              "FreqSpec-token\n(30% target)", "Random-token\n(30% target)"]
    fids = [a.fid_target, a.fid_draft, a.fid_freqspec, a.fid_random]
    panels = []
    for nm in names:
        p = os.path.join(a.grid_dir, f"grid_{nm}.png")
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing {p}")
        panels.append(load_panel(p, a.crop, a.cell))

    fig, axes = plt.subplots(1, 4, figsize=(a.width, a.width / 4 + 0.6))
    for ax, im, ti, fd in zip(axes, panels, titles, fids):
        ax.imshow(im)
        ax.set_title(ti, fontsize=11)
        if fd is not None:
            ax.set_xlabel(f"FID {fd:.2f}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    plt.tight_layout(pad=0.4)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[fig] wrote {a.out}")

    # Type-3 font removal via ghostscript, if available
    if shutil.which("gs"):
        tmp = a.out.replace(".pdf", "_out.pdf")
        r = subprocess.run(["gs", "-o", tmp, "-dNoOutputFonts",
                            "-sDEVICE=pdfwrite", a.out],
                           capture_output=True)
        if r.returncode == 0 and os.path.exists(tmp):
            os.replace(tmp, a.out)
            print(f"[fig] Type-3 fonts outlined -> {a.out}")
        else:
            print("[fig] WARN: ghostscript Type-3 removal failed; run gs manually")
    else:
        print("[fig] NOTE: ghostscript not found; run the gs command in the header")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--grid_dir", required=True)
    p.add_argument("--out", default="figures/in64_mixing.pdf")
    p.add_argument("--crop", type=int, default=0, help="top-left NxN cells (0=full)")
    p.add_argument("--cell", type=int, default=66, help="cell px (img+2*pad)")
    p.add_argument("--width", type=float, default=13.0, help="figure width inches")
    p.add_argument("--fid_target", type=float, default=None)
    p.add_argument("--fid_draft", type=float, default=None)
    p.add_argument("--fid_freqspec", type=float, default=None)
    p.add_argument("--fid_random", type=float, default=None)
    main(p.parse_args())
