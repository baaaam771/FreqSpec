#!/usr/bin/env bash
# run_noise_seed_repeat.sh — noise seed 재현성 검증 (원고 §4.3 방어용).
#
# 질문 2개:
#   (a) 비단조성(s20이 s16보다 나쁨)이 noise seed를 바꿔도 재현되는가?
#   (b) s12 실패 이미지가 seed가 바뀌어도 같은 이미지인가? (입력 고정성)
#
# 설계: 이미지·마스크는 manifest seed 고정(불변), 초기 noise만
# --noise_seed_offset 1,2 로 이동. 30장 × {s24,s20,s16,s12} × 2 offsets.
# largemask_coco(비단조성 최강) + objectremoval_coco(입력 고정성).
# 예상 총 ~20분.
set -euo pipefail
grep -q "noise_seed_offset" baseline_sweep.py || { echo "ERROR: noise_seed_offset 패치 필요"; exit 1; }

TARGET_ID="/mnt/HDD_12TB/bam_ki/checkpoints/stable-diffusion-xl-1.0-inpainting-0.1"
RESULTS_ROOT="/mnt/HDD_12TB/bam_ki/results"
COCO_ROOT="/mnt/HDD_12TB/bam_ki/datasets/coco2017/val2017"
COCO_CAP="/mnt/HDD_12TB/bam_ki/datasets/coco2017/annotations/captions_val2017.json"
COCO_INST="/mnt/HDD_12TB/bam_ki/datasets/coco2017/annotations/instances_val2017.json"
DRAFT="/mnt/HDD_12TB/bam_ki/runs/sdxl_coco_v2/draft_coco_400k.pt"
NUM_IMAGES=30

run_one () {
  local MODE="$1" NS="$2" OUT EXTRA
  if [[ "${MODE}" == "large" ]]; then
    OUT="${RESULTS_ROOT}/largemask_coco_ns${NS}"
    EXTRA="--mask_mode large --large_coverage 0.40 0.60"
  else
    OUT="${RESULTS_ROOT}/objectremoval_coco_ns${NS}"
    EXTRA="--mask_mode coco_object --obj_area_range 0.02 0.40 --instances_json ${COCO_INST}"
  fi
  echo "=== [nsrep] mode=${MODE} noise_offset=${NS} -> ${OUT} ==="
  python baseline_sweep.py \
    --target_id "${TARGET_ID}" \
    --draft_ckpt "${DRAFT}" \
    --use_ema_draft \
    --data_root "${COCO_ROOT}" \
    --caption_json "${COCO_CAP}" \
    --out_root "${OUT}" \
    --num_images "${NUM_IMAGES}" \
    --image_size 1024 \
    --target_steps 24 20 16 12 \
    --fs_presets strict \
    --resume \
    --noise_seed_offset "${NS}" \
    ${EXTRA} \
    --K 3 --patch 4 --t_spec_start 0.7 --beta 10.0 --boundary_weight 1.0 \
    --x0_thr_strict 0.02 --x0_thr_loose 0.07 \
    --x0_strict_center 0.45 --x0_strict_width 0.12 \
    --blend_temperature 0.10 \
    --mask_interior_weight 0.5 \
    --drift_k_switch_threshold 0.006
}

for NS in 1 2; do
  run_one large "${NS}"
  run_one object "${NS}"
done

echo ""
echo "[nsrep] done. Collect (ref는 s24 — s50은 이 run에 없음):"
echo "  python collect_sweep_metrics_v3.py \\"
echo "    --run_dirs ${RESULTS_ROOT}/largemask_coco_ns1 ${RESULTS_ROOT}/largemask_coco_ns2 \\"
echo "               ${RESULTS_ROOT}/objectremoval_coco_ns1 ${RESULTS_ROOT}/objectremoval_coco_ns2 \\"
echo "    --methods target_s24 target_s20 target_s16 target_s12 \\"
echo "    --ref_method target_s24 --out_prefix nsrep --device cuda"
