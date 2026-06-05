# baseline_sweep.py — 3 minimal patches for usage-map saving
#
# 세 위치만 수정하면 됩니다. 적용 후 모든 freqspec_* sample 디렉터리에
# usage_map.png가 자동 저장됩니다.
# ============================================================


# --------------------------------------------------------------
# PATCH 0: import 추가 (F.interpolate 사용 위함)
# --------------------------------------------------------------
# 위치: baseline_sweep.py 맨 위 import 블록

# Before:
import torch
from PIL import Image

# After:
import torch
import torch.nn.functional as F    # << NEW LINE
from PIL import Image


# --------------------------------------------------------------
# PATCH 1: run_one() 안의 fgsr_inpaint 호출에 return_usage_map=True 추가
# --------------------------------------------------------------
# 위치: baseline_sweep.py의 run_one() 함수, fgsr_inpaint 호출 부분
# (라인 약 360 근처, --x0_threshold 등 인자 마지막에)

# Before:
            (z_out, stats), t_run = timed_run(
                lambda: fgsr_inpaint(
                    target, draft, z_init.clone(), cond_z, mask_z, sch,
                    num_inference_steps=method["num_steps"],
                    K=args.K, patch_size=args.patch,
                    t_spec_start_norm=args.t_spec_start, beta=args.beta,
                    tol_low=method["tol_low"], tol_high=method["tol_high"],
                    boundary_weight=args.boundary_weight,
                    mask_interior_weight=args.mask_interior_weight,
                    uniform_saliency=False,
                    dwt=dwt, verbose=False,
                    guidance_scale=args.guidance_scale,
                    cond_emb=cond_emb, uncond_emb=uncond_emb,
                    known_z=z0, blend_known=True,
                    # Quick-fix forwards
                    x0_threshold=args.x0_threshold,
                    k_switch_threshold=args.k_switch_threshold,
                    spec1_below_tnorm=args.spec1_below_tnorm,
                    log_diagnostics=args.log_diagnostics,
                    blend_temperature=args.blend_temperature,
                    x0_thr_strict=args.x0_thr_strict,
                    x0_thr_loose=args.x0_thr_loose,
                    x0_strict_center=args.x0_strict_center,
                    x0_strict_width=args.x0_strict_width,
                    drift_k_switch_threshold=args.drift_k_switch_threshold,
                ),
                device,
            )

# After (add ONE LINE at the end of the call):
            (z_out, stats), t_run = timed_run(
                lambda: fgsr_inpaint(
                    target, draft, z_init.clone(), cond_z, mask_z, sch,
                    num_inference_steps=method["num_steps"],
                    K=args.K, patch_size=args.patch,
                    t_spec_start_norm=args.t_spec_start, beta=args.beta,
                    tol_low=method["tol_low"], tol_high=method["tol_high"],
                    boundary_weight=args.boundary_weight,
                    mask_interior_weight=args.mask_interior_weight,
                    uniform_saliency=False,
                    dwt=dwt, verbose=False,
                    guidance_scale=args.guidance_scale,
                    cond_emb=cond_emb, uncond_emb=uncond_emb,
                    known_z=z0, blend_known=True,
                    x0_threshold=args.x0_threshold,
                    k_switch_threshold=args.k_switch_threshold,
                    spec1_below_tnorm=args.spec1_below_tnorm,
                    log_diagnostics=args.log_diagnostics,
                    blend_temperature=args.blend_temperature,
                    x0_thr_strict=args.x0_thr_strict,
                    x0_thr_loose=args.x0_thr_loose,
                    x0_strict_center=args.x0_strict_center,
                    x0_strict_width=args.x0_strict_width,
                    drift_k_switch_threshold=args.drift_k_switch_threshold,
                    return_usage_map=args.save_usage_maps,  # << NEW LINE
                ),
                device,
            )


# --------------------------------------------------------------
# PATCH 2: 결과 저장 부분에 usage_map.png 저장 추가
# --------------------------------------------------------------
# 위치: 같은 run_one() 함수 끝부분, save_rgb / save_gray 호출 직후
# (라인 약 405 근처)

# Before:
    out = target.decode_latent(z_out)
    out = img * (1 - mask_pix) + out * mask_pix
    save_rgb(img, os.path.join(m_out, "gt.png"))
    save_rgb(out, os.path.join(m_out, "out.png"))
    save_gray(mask_pix, os.path.join(m_out, "mask.png"))

# After (add the usage-map block):
    out = target.decode_latent(z_out)
    out = img * (1 - mask_pix) + out * mask_pix
    save_rgb(img, os.path.join(m_out, "gt.png"))
    save_rgb(out, os.path.join(m_out, "out.png"))
    save_gray(mask_pix, os.path.join(m_out, "mask.png"))

    # ===== NEW: save draft usage map if present =====
    if "usage_map" in stats:
        um = stats["usage_map"]  # [B,1,H_lat,W_lat], values in [0,1]
        # Upsample to image resolution (bilinear keeps soft gradient).
        um_up = F.interpolate(
            um.to(device), size=(args.image_size, args.image_size),
            mode="bilinear", align_corners=False,
        )
        save_gray(um_up, os.path.join(m_out, "usage_map.png"))
    # ================================================


# --------------------------------------------------------------
# PATCH 3: get_parser()에 --save_usage_maps flag 추가 (마지막 한 줄)
# --------------------------------------------------------------
# 위치: get_parser() 함수 안, --drift_k_switch_threshold 인자 뒤

# Append BEFORE `return p`:
    p.add_argument("--save_usage_maps", action="store_true",
                   help="Save per-image draft usage maps (averaged "
                        "soft-blend weight w(p)) as usage_map.png for "
                        "qualitative figures.")


# --------------------------------------------------------------
# 사용법
# --------------------------------------------------------------
# 패치 적용 후, --save_usage_maps 플래그를 추가해서 sweep 실행:
#
#   python baseline_sweep.py \
#     ...기존 인자들... \
#     --save_usage_maps
#
# 그러면 각 freqspec_*/img_NNN/ 안에 usage_map.png가 저장됩니다.
#
# 그리고 figure 조립:
#
#   python assemble_qualitative_3datasets.py \
#     --ffhq_dir    ... \
#     --places2_dir ... \
#     --coco_dir    ... \
#     --ffhq_idx N --places2_idx N --coco_idx N \
#     --out_path figures/fig5_qualitative_3datasets \
#     --usage_map_dir <auto: same as freqspec_default dir>
#
# 단, 현재 assemble_qualitative_3datasets.py는 usage_map을 별도
# 디렉터리에서 찾는 구조이므로, freqspec_default의 usage_map.png를
# 그 형식으로 모아주는 작은 수집 단계가 필요합니다. 다음과 같이 하면 됩니다:
#
#   mkdir -p /tmp/usage_maps
#   cp /path/to/qualitative_ffhq_run/freqspec_default/img_003/usage_map.png    /tmp/usage_maps/ffhq_img_003_usage.png
#   cp /path/to/qualitative_places2_run/freqspec_default/img_007/usage_map.png /tmp/usage_maps/places2_img_007_usage.png
#   cp /path/to/qualitative_coco_run/freqspec_default/img_005/usage_map.png    /tmp/usage_maps/coco_img_005_usage.png
#
# 그 다음 assemble script에 --usage_map_dir /tmp/usage_maps 를 전달.
