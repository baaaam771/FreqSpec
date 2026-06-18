#!/usr/bin/env bash
# =====================================================================
#  Table 1 scale-up: COCO n=1000, FFHQ n=500, Places2 n=500
#  target s30/40/50 + FreqSpec strict/mid/default (same manifest per dataset)
#  3 stages per dataset:
#    (1) baseline_sweep.py  -> inference (out.png + results.csv)
#    (2) analyze_speed_matched.py -> summary.json (PSNR/LPIPS/bLPIPS/LPIPSt/speedup)
#    (3) extract_table1.py  -> print Table 1 rows
#  ALL stages are --resume safe. Run ONE dataset block at a time.
#
#  USAGE (run each line yourself, ONE at a time; do NOT paste the whole file):
#    export TMPDIR=/mnt/HDD_12TB/bam_ki/tmp
#    bash run_table1_scaleup.sh coco_sweep
#    bash run_table1_scaleup.sh coco_metric
#    ... etc
# =====================================================================
set -e

TARGET=/mnt/HDD_12TB/bam_ki/checkpoints/stable-diffusion-xl-1.0-inpainting-0.1
RESULTS=/mnt/HDD_12TB/bam_ki/results
CODE=~/FreqSpec/FreqSpec/fgsi_v2

: "${TMPDIR:=/mnt/HDD_12TB/bam_ki/tmp}"
export TMPDIR
mkdir -p "$TMPDIR"

# ---- paths (confirmed) ----
COCO_DRAFT=/mnt/HDD_12TB/bam_ki/runs/sdxl_coco_v2/draft_final.pt
COCO_DATA=/mnt/HDD_12TB/bam_ki/datasets/coco2017/val2017
COCO_CAP=/mnt/HDD_12TB/bam_ki/datasets/coco2017/annotations/captions_val2017.json

FFHQ_DRAFT=/mnt/HDD_12TB/bam_ki/runs/sdxl_ffhq_v1/draft_final.pt
FFHQ_DATA=/mnt/HDD_12TB/bam_ki/datasets/ffhq_hf/images

PLACES_DRAFT=/mnt/HDD_12TB/bam_ki/runs/sdxl_v1/draft_final.pt
PLACES_DATA=/mnt/HDD_12TB/bam_ki/datasets/places2/train_256   # confirmed: matches combo2_places2 manifest

cd "$CODE"

case "$1" in
  # ---------------- COCO (n=1000) ----------------
  coco_sweep)
    python baseline_sweep.py \
      --target_id $TARGET \
      --draft_ckpt $COCO_DRAFT --use_ema_draft \
      --data_root $COCO_DATA --caption_json $COCO_CAP \
      --out_root $RESULTS/table1_coco_n1000 \
      --num_images 1000 --image_size 1024 --seed 42 \
      --target_steps 30 40 50 --resume
    ;;
  coco_metric)
    python analyze_speed_matched.py --sweep_root $RESULTS/table1_coco_n1000
    ;;

  # ---------------- FFHQ (n=500) ----------------
  ffhq_sweep)
    python baseline_sweep.py \
      --target_id $TARGET \
      --draft_ckpt $FFHQ_DRAFT --use_ema_draft \
      --data_root $FFHQ_DATA \
      --out_root $RESULTS/table1_ffhq_n500 \
      --num_images 500 --image_size 1024 --seed 42 \
      --target_steps 30 40 50 --resume
    ;;
  ffhq_metric)
    python analyze_speed_matched.py --sweep_root $RESULTS/table1_ffhq_n500
    ;;

  # ---------------- Places2 (n=500) ----------------
  places_sweep)
    python baseline_sweep.py \
      --target_id $TARGET \
      --draft_ckpt $PLACES_DRAFT --use_ema_draft \
      --data_root $PLACES_DATA \
      --out_root $RESULTS/table1_places2_n500 \
      --num_images 500 --image_size 1024 --seed 42 \
      --target_steps 30 40 50 --resume
    ;;
  places_metric)
    python analyze_speed_matched.py --sweep_root $RESULTS/table1_places2_n500
    ;;

  # ---------------- extract all 3 into Table 1 rows ----------------
  table)
    python3 - <<'PY'
import json, csv, numpy as np, os
R="/mnt/HDD_12TB/bam_ki/results"
DSETS=[("FFHQ",  f"{R}/table1_ffhq_n500",   [50,40,30]),
       ("Places2",f"{R}/table1_places2_n500",[50,40,30]),
       ("COCO",  f"{R}/table1_coco_n1000",  [50,40,30])]
def na(folder, method):
    p=os.path.join(folder,method,"results.csv"); nfe,acc=[],[]
    if os.path.isfile(p):
        for r in csv.DictReader(open(p)):
            try:
                nfe.append(float(r['target_nfe']))
                if r['accept_rate']: acc.append(float(r['accept_rate']))
            except: pass
    return (np.mean(nfe) if nfe else None, np.mean(acc) if acc else None,
            np.std(nfe) if nfe else None, len(nfe))
for name, folder, steps in DSETS:
    sp=os.path.join(folder,"summary.json")
    print(f"\n===== {name}  ({folder}) =====")
    if not os.path.isfile(sp): print("  summary.json 없음 (metric 단계 먼저)"); continue
    s=json.load(open(sp))
    for st in steps:
        k=f"target_s{st}"
        if k in s:
            m=s[k]
            print(f"  {k:18s} NFE={st:>3} acc=--    PSNR={m['psnr']:.2f} SSIM={m.get('ssim',0):.3f} "
                  f"LPIPS={m['lpips']:.4f} bLPIPS={m['boundary_lpips']:.4f} LPIPSt={m['lpips_vs_tgt']:.4f}")
    for p in ["strict","mid","default"]:
        k=f"freqspec_{p}"
        if k in s:
            m=s[k]; n,a,sd,cnt=na(folder,k)
            ns=f"{n:.1f}" if n is not None else "--"
            as_=f"{a:.3f}" if a is not None else "--"
            sds=f"{sd:.1f}" if sd is not None else "--"
            print(f"  {k:18s} NFE={ns}±{sds} acc={as_} PSNR={m['psnr']:.2f} SSIM={m.get('ssim',0):.3f} "
                  f"LPIPS={m['lpips']:.4f} bLPIPS={m['boundary_lpips']:.4f} LPIPSt={m['lpips_vs_tgt']:.4f} (n={cnt})")
PY
    ;;

  *)
    echo "stages: coco_sweep coco_metric | ffhq_sweep ffhq_metric | places_sweep places_metric | table"
    echo "run ONE at a time. export TMPDIR first. check nvidia-smi is free (no Table C training)."
    ;;
esac
