#!/usr/bin/env python
"""
baseline_sweep.py — Reduced-step baseline vs FreqSpec comparison runner.

[PATCHED] adds 4 minimal changes for qualitative-figure preparation:
  1. import torch.nn.functional as F            (for usage_map upsampling)
  2. Combo 2 sanity print at start of main()    (which fixes are active)
  3. fgsr_inpaint(..., return_usage_map=...)    (when --save_usage_maps)
  4. save usage_map.png alongside out/gt/mask   (when --save_usage_maps)
  5. --save_usage_maps argparse flag

Default behavior is unchanged. When --save_usage_maps is NOT passed,
this script behaves identically to the original.

Reviewer-critical experiment: answers "is FreqSpec better than just reducing
the target's denoising steps to match the speed?"

This script reuses the EXISTING inference pipeline (baseline_inpaint /
fgsr_inpaint from inference.speculative) — no new inference logic. It:
  1. Builds a FIXED manifest of (image, mask, prompt, seed) — shared by all
     methods so comparisons are paired and fair.
  2. Runs each method (Target at several step counts + FreqSpec at several
     tolerances) on the SAME manifest.
  3. Times each run with torch.cuda.synchronize() for accurate wall-clock.
  4. Saves per-image outputs and an image-level CSV per method.
  5. (NEW) Optionally saves draft usage maps for FreqSpec methods.

Example (Combo 2 + usage maps):
    python baseline_sweep.py \\
        --target_id /mnt/HDD_12TB/bam_ki/checkpoints/stable-diffusion-xl-1.0-inpainting-0.1 \\
        --draft_ckpt /mnt/HDD_12TB/bam_ki/runs/sdxl_v1/draft_final.pt \\
        --use_ema_draft \\
        --data_root /mnt/HDD_12TB/bam_ki/datasets/coco2017/val2017 \\
        --caption_json /mnt/HDD_12TB/bam_ki/datasets/coco2017/annotations/captions_val2017.json \\
        --out_root /mnt/HDD_12TB/bam_ki/results/qualitative_coco_run \\
        --num_images 12 --image_size 1024 \\
        --target_steps 50 30 \\
        --x0_thr_strict 0.02 --x0_thr_loose 0.07 \\
        --x0_strict_center 0.45 --x0_strict_width 0.12 \\
        --blend_temperature 0.10 \\
        --mask_interior_weight 0.5 \\
        --drift_k_switch_threshold 0.006 \\
        --save_usage_maps
"""
import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F      # [PATCH 0] needed for usage_map upsample
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.target_wrapper import TargetWrapper
from models.draft import DraftEpsUNet
from models.wavelet import DWT2D
from training.scheduler import DDPMSchedule
from inference.speculative import fgsr_inpaint, baseline_inpaint


# ====================================================================
# Method definitions: the set of methods to compare on the same images
# ====================================================================
def build_method_list(args):
    methods = []
    # Reduced-step target baselines
    for steps in args.target_steps:
        methods.append({
            "name": f"target_s{steps}",
            "type": "target",
            "num_steps": steps,
            "tol_low": None, "tol_high": None,
        })
    # FreqSpec tolerance presets (all at full 50-step schedule)
    # [PATCH 6] --fs_presets selects a subset (comma-separated short names,
    # e.g. "default" or "strict,default"). Default runs all three, matching
    # the original behavior.
    fs_presets = [
        ("strict",  0.01, 0.10),
        ("mid",     0.02, 0.15),
        ("default", 0.03, 0.30),
    ]
    wanted = {s.strip() for s in args.fs_presets.split(",") if s.strip()}
    for short, tl, th in fs_presets:
        if short not in wanted:
            continue
        methods.append({
            "name": f"freqspec_{short}",
            "type": "freqspec",
            "num_steps": args.num_steps,
            "tol_low": tl, "tol_high": th,
        })
    return methods


# ====================================================================
# Image / mask helpers (match run_inpaint.py preprocessing)
# ====================================================================
def load_image(path, size):
    img = Image.open(path).convert("RGB")
    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])(img).unsqueeze(0)


def make_mask(size, rng, p_box=0.4):
    """
    Generate a synthetic mask using the SAME distribution as training
    (training.train.random_inpaint_mask): box (40퍼센트) or irregular brush
    strokes (60퍼센트). Uses the provided rng for deterministic per-image masks.

    Returns 1x1xHxW float (1 = masked/inpaint region).
    """
    import math
    H = W = size
    m = torch.zeros(1, 1, H, W)
    if rng.random() < p_box:
        # box mask: random size H/8 .. H/2
        bh = rng.randint(H // 8, H // 2)
        bw = rng.randint(W // 8, W // 2)
        y = rng.randint(0, H - bh)
        x = rng.randint(0, W - bw)
        m[:, :, y:y + bh, x:x + bw] = 1.0
    else:
        # irregular brush strokes: 3..8 segments
        n_strokes = rng.randint(3, 8)
        for _ in range(n_strokes):
            y = rng.randint(0, H - 1)
            x = rng.randint(0, W - 1)
            length = rng.randint(H // 8, H // 3)
            angle = rng.uniform(0, 2 * math.pi)
            thick = rng.randint(3, max(4, H // 16))
            for s in range(length):
                yy = int(y + s * math.sin(angle))
                xx = int(x + s * math.cos(angle))
                if 0 <= yy < H and 0 <= xx < W:
                    m[:, :,
                      max(0, yy - thick):min(H, yy + thick),
                      max(0, xx - thick):min(W, xx + thick)] = 1.0
    return m


def make_large_mask(size, rng, cov_lo=0.40, cov_hi=0.60):
    """
    [PATCH 7] Large/complex mask generator for the reviewer-requested
    large-mask evaluation. Deterministic given rng. Composes large boxes
    plus thick brush strokes until mask coverage reaches a target ratio
    sampled from [cov_lo, cov_hi]. Returns 1x1xHxW float (1 = masked).
    """
    import math
    H = W = size
    target_cov = rng.uniform(cov_lo, cov_hi)
    m = torch.zeros(1, 1, H, W)
    guard = 0
    while m.mean().item() < target_cov and guard < 40:
        guard += 1
        if rng.random() < 0.6:
            # large box: H/4 .. 0.7H per side
            bh = rng.randint(H // 4, int(H * 0.7))
            bw = rng.randint(W // 4, int(W * 0.7))
            y = rng.randint(0, H - bh)
            x = rng.randint(0, W - bw)
            m[:, :, y:y + bh, x:x + bw] = 1.0
        else:
            # thick brush stroke
            y = rng.randint(0, H - 1)
            x = rng.randint(0, W - 1)
            length = rng.randint(H // 4, int(H * 0.6))
            angle = rng.uniform(0, 2 * math.pi)
            thick = rng.randint(H // 16, H // 8)
            for s in range(length):
                yy = int(y + s * math.sin(angle))
                xx = int(x + s * math.cos(angle))
                if 0 <= yy < H and 0 <= xx < W:
                    m[:, :,
                      max(0, yy - thick):min(H, yy + thick),
                      max(0, xx - thick):min(W, xx + thick)] = 1.0
    return m


def _load_coco_instance_map(instances_json_path):
    """[PATCH 9] {file_name: {"width","height","anns":[...]}} from COCO
    instances JSON. Keeps polygon (non-crowd) annotations only."""
    import json
    with open(instances_json_path) as f:
        data = json.load(f)
    imgs = {im["id"]: im for im in data["images"]}
    out = {}
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue
        seg = ann.get("segmentation")
        if not isinstance(seg, list) or not seg:
            continue
        im = imgs.get(ann["image_id"])
        if im is None:
            continue
        e = out.setdefault(im["file_name"], {
            "width": im["width"], "height": im["height"], "anns": []})
        e["anns"].append({"segmentation": seg, "bbox": ann["bbox"],
                          "area": ann["area"]})
    print(f"[sweep] instance map: {len(out)} images with polygon objects")
    return out


def make_coco_object_mask(entry, size, rng, min_frac=0.02, max_frac=0.40,
                          dilate_frac=0.02):
    """[PATCH 9] Object-removal mask: rasterize ONE randomly chosen (rng)
    instance polygon, apply the same Resize+CenterCrop as load_image,
    then dilate slightly to cover the object boundary.
    Returns 1x1xHxW float (1 = region to remove/inpaint) or None if no
    usable instance survives the center crop."""
    from PIL import Image as PILImage, ImageDraw
    from torchvision.transforms import InterpolationMode
    W0, H0 = entry["width"], entry["height"]
    cands = [a for a in entry["anns"]
             if min_frac <= a["area"] / (W0 * H0) <= max_frac]
    if not cands:
        return None
    order = list(range(len(cands)))
    rng.shuffle(order)
    tf = transforms.Compose([
        transforms.Resize(size, interpolation=InterpolationMode.NEAREST),
        transforms.CenterCrop(size),
    ])
    for j in order:
        ann = cands[j]
        m_img = PILImage.new("L", (W0, H0), 0)
        drw = ImageDraw.Draw(m_img)
        ok = False
        for poly in ann["segmentation"]:
            if len(poly) >= 6:
                drw.polygon(poly, fill=255)
                ok = True
        if not ok:
            x, y, w, h = ann["bbox"]
            drw.rectangle([x, y, x + w, y + h], fill=255)
        m_img = tf(m_img)
        import numpy as _np
        m = torch.from_numpy(
            (_np.array(m_img) > 127).astype("float32")
        ).view(1, 1, size, size)
        if m.mean().item() < 0.005:   # object cropped away — try next
            continue
        k = max(3, int(size * dilate_frac) | 1)
        m = F.max_pool2d(m, kernel_size=k, stride=1, padding=k // 2)
        return m.clamp(0, 1)
    return None


def save_rgb(t, path):
    t = (t.clamp(-1, 1) + 1) / 2
    Image.fromarray(
        (t[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    ).save(path)


def save_gray(t, path):
    Image.fromarray(
        (t[0, 0].cpu().numpy() * 255).clip(0, 255).astype("uint8")
    ).save(path)


# ====================================================================
# Manifest: fixed list of images shared by all methods
# ====================================================================
def _load_coco_caption_map(caption_json_path):
    """Load COCO captions JSON and return {file_name: [caption, ...]}.
    Matches the loading logic in training.train.ImageDataset exactly."""
    import json
    with open(caption_json_path) as f:
        data = json.load(f)
    id_to_fn = {im["id"]: im["file_name"] for im in data["images"]}
    caption_map = {}
    for ann in data["annotations"]:
        fn = id_to_fn.get(ann["image_id"])
        if fn is None:
            continue
        caption_map.setdefault(fn, []).append(ann["caption"])
    n = sum(1 for k in caption_map if caption_map[k])
    avg = sum(len(v) for v in caption_map.values()) / max(1, n)
    print(f"[sweep] loaded captions for {n} COCO images "
          f"(avg {avg:.1f} caps/img)")
    return caption_map


def build_manifest(args):
    """Collect num_images images deterministically (seed-fixed)."""
    root = Path(args.data_root)
    # [PATCH 5] case-insensitive, robust image discovery + count print.
    # (Prevents the silent "found 3 images" failure mode when files use
    # uppercase extensions or the directory only holds repo metadata.)
    img_exts = {".jpg", ".jpeg", ".png", ".webp"}
    all_imgs = sorted(
        str(p) for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in img_exts
    )
    print(f"[sweep] found {len(all_imgs)} image files in {root}")
    # [PATCH 9] object-removal mode: keep only images with a usable instance
    if getattr(args, "mask_mode", "train") == "coco_object":
        assert args.instances_json, "--instances_json required for coco_object"
        args._inst_map = _load_coco_instance_map(args.instances_json)
        def _has_obj(p):
            e = args._inst_map.get(os.path.basename(p))
            if not e:
                return False
            return any(0.02 <= a["area"] / (e["width"] * e["height"]) <= 0.40
                       for a in e["anns"])
        before = len(all_imgs)
        all_imgs = [p for p in all_imgs if _has_obj(p)]
        print(f"[sweep] coco_object filter: {before} -> {len(all_imgs)} "
              f"images with a 2-40 percent-area instance")
    if len(all_imgs) < args.num_images:
        print(f"[sweep] WARNING: fewer images than --num_images "
              f"({len(all_imgs)} < {args.num_images}) — check data_root!")
    rng = random.Random(args.seed)
    rng.shuffle(all_imgs)
    chosen = all_imgs[:args.num_images]

    # COCO caption map (if --caption_json given)
    caption_map = {}
    if args.caption_json:
        caption_map = _load_coco_caption_map(args.caption_json)

    manifest = []
    n_with_caption = 0
    for idx, img_path in enumerate(chosen):
        # per-image deterministic seed for mask + diffusion noise
        img_seed = args.seed * 100000 + idx
        prompt = ""
        # Priority: COCO caption > auto_prompt (path-based) > empty
        if caption_map:
            fn = os.path.basename(img_path)
            caps = caption_map.get(fn, [])
            if caps:
                # deterministic per-image caption pick (first one)
                prompt = caps[0]
                n_with_caption += 1
        if not prompt and args.auto_prompt:
            try:
                from training.train import _path_to_prompt
                prompt = _path_to_prompt(img_path)
            except Exception:
                parts = Path(img_path).parts
                cat = parts[-2].replace("_", " ") if len(parts) >= 2 else "scene"
                prompt = f"a photo of a {cat}"
        # [PATCH 8] final fallback: training default prompt
        if not prompt and args.default_prompt:
            prompt = args.default_prompt
        manifest.append({
            "idx": idx,
            "image_path": img_path,
            "prompt": prompt,
            "seed": img_seed,
        })
    if caption_map:
        print(f"[sweep] {n_with_caption}/{len(manifest)} images got per-image "
              f"captions from the COCO JSON")
        # Show a few examples so you can sanity-check
        for item in manifest[:3]:
            print(f"  example: {os.path.basename(item['image_path'])} -> "
                  f"\"{item['prompt']}\"")
    return manifest


# ====================================================================
# Timing with CUDA sync (reviewer-required for accurate wall-clock)
# ====================================================================
def timed_run(fn, device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return out, time.perf_counter() - t0


# ====================================================================
# [PATCH 4] Combo 2 sanity print — show which fixes are active
# ====================================================================
def print_combo2_status(args):
    def _flag(v):
        return (f"on (val={v})"
                if (v is not None and v != 0.0 and v != "")
                else "off")
    print("[sweep] FreqSpec fix configuration:")
    print(f"  Fix 2  (fixed x0 gate)        : {_flag(args.x0_threshold)}")
    print(f"  Fix 3  (soft blend)           : {_flag(args.blend_temperature)}")
    print(f"  Fix 4  (timestep x0 strict)   : {_flag(args.x0_strict_center)} "
          f"[strict={args.x0_thr_strict}, loose={args.x0_thr_loose}, "
          f"width={args.x0_strict_width}]")
    print(f"  Fix 4' (mask-interior)        : "
          f"{_flag(args.mask_interior_weight)}")
    print(f"  Fix 5  (drift-aware K-step)   : "
          f"{_flag(args.drift_k_switch_threshold)} "
          f"[k_switch_thr={args.k_switch_threshold}]")
    all_combo2 = (
        (args.x0_strict_center is not None)
        and (args.blend_temperature is not None)
        and (args.mask_interior_weight > 0)
        and (args.drift_k_switch_threshold is not None)
    )
    label = "COMBO 2 (full)" if all_combo2 else "PARTIAL (not Combo 2)"
    print(f"  >>> Active configuration: {label}")
    if args.save_usage_maps:
        print(f"  >>> --save_usage_maps ON: usage_map.png will be saved "
              f"for each freqspec_* sample")


# ====================================================================
# Main
# ====================================================================
def main(args):
    device = torch.device(args.device)
    target_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                    "fp32": torch.float32}[args.target_dtype]

    target = TargetWrapper(model_id=args.target_id, device=device,
                           dtype=target_dtype)
    assert target.available, "Target model must be available for real eval."

    sch = DDPMSchedule(
        num_train_timesteps=target.scheduler_ref.config.num_train_timesteps,
        beta_start=target.scheduler_ref.config.beta_start,
        beta_end=target.scheduler_ref.config.beta_end,
        beta_schedule=target.scheduler_ref.config.beta_schedule,
        device=device,
    )

    # Load draft (architecture restored from checkpoint args, matching
    # run_inpaint.py exactly)
    draft_kwargs = {
        "latent_ch": target.latent_ch,
        "num_train_timesteps": sch.num_train_timesteps,
    }
    ck = torch.load(args.draft_ckpt, map_location=device)
    saved_args = ck.get("args", {})
    if "draft_base_ch" in saved_args:
        draft_kwargs["base_ch"] = saved_args["draft_base_ch"]
        draft_kwargs["ch_mult"] = tuple(saved_args["draft_ch_mult"])
        draft_kwargs["t_dim"] = saved_args["draft_t_dim"]
    draft = DraftEpsUNet(**draft_kwargs).to(device).eval()
    if args.use_ema_draft and ck.get("ema_draft") is not None:
        draft.load_state_dict(ck["ema_draft"])
        print(f"[sweep] loaded EMA draft")
    else:
        draft.load_state_dict(ck["draft"])
        print(f"[sweep] loaded draft")
    print(f"[sweep] draft params: "
          f"{sum(p.numel() for p in draft.parameters())/1e6:.2f}M")

    dwt = DWT2D("haar").to(device)

    methods = build_method_list(args)
    manifest = build_manifest(args)

    os.makedirs(args.out_root, exist_ok=True)
    with open(os.path.join(args.out_root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[sweep] {len(manifest)} images x {len(methods)} methods")
    print(f"[sweep] methods: {[m['name'] for m in methods]}")

    # [PATCH 4] Combo 2 sanity print
    print_combo2_status(args)

    # Warmup (excluded from timing) — first image, target-only
    print("[sweep] warmup...")
    _warmup(target, draft, sch, dwt, manifest[0], args, device)

    # Run all methods on all images
    for method in methods:
        m_out = os.path.join(args.out_root, method["name"])
        os.makedirs(m_out, exist_ok=True)
        csv_path = os.path.join(m_out, "results.csv")

        done = set()
        if args.resume and os.path.isfile(csv_path):
            with open(csv_path) as f:
                for row in csv.DictReader(f):
                    done.add(int(row["idx"]))

        write_header = not (args.resume and os.path.isfile(csv_path))
        f_csv = open(csv_path, "a", newline="")
        writer = csv.writer(f_csv)
        if write_header:
            writer.writerow([
                "idx", "image_path", "method", "num_steps",
                "time_sec", "target_nfe", "draft_nfe", "accept_rate",
            ])

        print(f"\n[sweep] === {method['name']} ===")
        for item in manifest:
            if item["idx"] in done:
                continue
            row = run_one(target, draft, sch, dwt, item, method, args, device)
            writer.writerow(row)
            f_csv.flush()
            print(f"  [{method['name']}] img {item['idx']:03d}  "
                  f"t={row[4]:.2f}s  tgt_nfe={row[5]}")
        f_csv.close()

    print(f"\n[sweep] done -> {args.out_root}")
    print("[sweep] next: run analyze_speed_matched.py to compute metrics")


def _prepare_latents(target, sch, item, args, device):
    """Encode image, build mask, make initial noise (seeded).
    Mirrors run_inpaint.py exactly."""
    img = load_image(item["image_path"], args.image_size).to(device)
    rng = random.Random(item["seed"])
    # [PATCH 7] mask mode dispatch: "train" = training distribution
    # (original behavior); "large" = 40-60%-coverage large/complex masks.
    if getattr(args, "mask_mode", "train") == "large":
        mask_pix = make_large_mask(args.image_size, rng,
                                   cov_lo=args.large_coverage[0],
                                   cov_hi=args.large_coverage[1]).to(device)
    elif getattr(args, "mask_mode", "train") == "coco_object":
        entry = args._inst_map.get(os.path.basename(item["image_path"]))
        m = make_coco_object_mask(entry, args.image_size, rng) if entry else None
        if m is None:   # safety net (filtered manifest should prevent this)
            m = make_large_mask(args.image_size, rng, 0.10, 0.30)
        mask_pix = m.to(device)
    else:
        mask_pix = make_mask(args.image_size, rng).to(device)
    masked_pix = img * (1 - mask_pix)

    z0 = target.encode_image(img)
    cond_z = target.encode_image(masked_pix)       # known-region conditioning
    mask_z = target.downsample_mask(mask_pix)

    # seeded initial noise for reproducibility across methods
    torch.manual_seed(item["seed"])
    z_init = torch.randn_like(z0)

    return img, mask_pix, z0, mask_z, cond_z, z_init


def _get_emb(target, item, args, z0):
    """Returns (cond_emb, uncond_emb). get_text_embeddings returns a 3-tuple
    (cond, uncond, use_cfg) — matching run_inpaint.py."""
    # [PATCH 8] fallback priority: manifest prompt > default_prompt > generic
    prompt = item["prompt"] or args.default_prompt or "a photograph"
    cond_emb, uncond_emb, use_cfg = target.get_text_embeddings(
        prompt, batch_size=z0.shape[0], guidance_scale=args.guidance_scale
    )
    return cond_emb, uncond_emb


def _warmup(target, draft, sch, dwt, item, args, device):
    img, mask_pix, z0, mask_z, cond_z, z_init = _prepare_latents(
        target, sch, item, args, device)
    cond_emb, uncond_emb = _get_emb(target, item, args, z0)
    with torch.no_grad():
        baseline_inpaint(
            target, z_init.clone(), cond_z, mask_z, sch,
            num_inference_steps=10, guidance_scale=args.guidance_scale,
            cond_emb=cond_emb, uncond_emb=uncond_emb,
            known_z=z0, blend_known=True,
        )


def run_one(target, draft, sch, dwt, item, method, args, device):
    img, mask_pix, z0, mask_z, cond_z, z_init = _prepare_latents(
        target, sch, item, args, device)
    cond_emb, uncond_emb = _get_emb(target, item, args, z0)

    m_out = os.path.join(args.out_root, method["name"], f"img_{item['idx']:03d}")
    os.makedirs(m_out, exist_ok=True)

    with torch.no_grad():
        if method["type"] == "target":
            (z_out, stats), t_run = timed_run(
                lambda: baseline_inpaint(
                    target, z_init.clone(), cond_z, mask_z, sch,
                    num_inference_steps=method["num_steps"],
                    guidance_scale=args.guidance_scale,
                    cond_emb=cond_emb, uncond_emb=uncond_emb,
                    known_z=z0, blend_known=True,
                ),
                device,
            )
            target_nfe = stats["target_calls"]
            draft_nfe = 0
            accept = ""
        else:  # freqspec
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
                    # [PATCH 1] usage-map collection for qualitative figures
                    return_usage_map=args.save_usage_maps,
                ),
                device,
            )
            target_nfe = stats["target_calls"]
            draft_nfe = stats["draft_calls"]
            accept = f"{stats['accept_rate']:.4f}"

    # composite + save (gt = input image, needed for metrics later)
    out = target.decode_latent(z_out)
    out = img * (1 - mask_pix) + out * mask_pix
    save_rgb(img, os.path.join(m_out, "gt.png"))         # ground truth
    save_rgb(out, os.path.join(m_out, "out.png"))        # method output
    save_gray(mask_pix, os.path.join(m_out, "mask.png"))

    # [PATCH 2] save draft usage map if present in stats
    if "usage_map" in stats:
        um = stats["usage_map"]  # [B,1,H_lat,W_lat] on CPU, values in [0,1]
        um_up = F.interpolate(
            um.to(device), size=(args.image_size, args.image_size),
            mode="bilinear", align_corners=False,
        )
        save_gray(um_up, os.path.join(m_out, "usage_map.png"))

    return [item["idx"], item["image_path"], method["name"],
            method["num_steps"], round(t_run, 4), target_nfe, draft_nfe, accept]


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--target_id", type=str, required=True)
    p.add_argument("--draft_ckpt", type=str, required=True)
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--out_root", type=str, required=True)
    p.add_argument("--num_images", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--target_dtype", type=str, default="fp16",
                   choices=["fp16", "bf16", "fp32"])
    p.add_argument("--image_size", type=int, default=1024)
    p.add_argument("--num_steps", type=int, default=50,
                   help="Full schedule length for FreqSpec.")
    p.add_argument("--target_steps", type=int, nargs="+",
                   default=[50, 40, 37, 30, 25],
                   help="Step counts for reduced-step target baselines.")
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--auto_prompt", action="store_true")
    p.add_argument("--caption_json", type=str, default="",
                   help="Path to COCO captions JSON. If given, per-image "
                        "captions from the JSON are used as prompts "
                        "(matches the training pipeline for COCO drafts).")
    p.add_argument("--use_ema_draft", action="store_true",
                   help="Use EMA draft weights if available (recommended).")
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--patch", type=int, default=4)
    p.add_argument("--t_spec_start", type=float, default=0.7)
    p.add_argument("--beta", type=float, default=10.0)
    p.add_argument("--boundary_weight", type=float, default=1.0)
    p.add_argument("--mask_interior_weight", type=float, default=0.0,
                   help="Fix 4 prime: penalize mask interior in saliency. "
                        "0 disables. Try 0.3/0.5/0.8.")
    # --- Quick-fix knobs (forwarded to fgsr_inpaint) ---
    p.add_argument("--x0_threshold", type=float, default=None,
                   help="Fix 2: pred_x0 disagreement gate. Per-patch "
                        "MSE threshold between pred_x0_draft and "
                        "pred_x0_target. None disables the gate. "
                        "Suggested sweep: 0.01 / 0.02 / 0.05.")
    p.add_argument("--k_switch_threshold", type=float, default=0.6,
                   help="Fix 1: rolling accept threshold above which "
                        "spec-K (K>1) is used. Default 0.6 (original). "
                        "Raise to e.g. 0.75 to fall back to spec-1 sooner.")
    p.add_argument("--spec1_below_tnorm", type=float, default=0.0,
                   help="Fix 1b: force spec-1 once t/T < this value. "
                        "0.0 disables. Try 0.4 to use spec-1 in the late "
                        "detail phase.")
    p.add_argument("--log_diagnostics", action="store_true",
                   help="Print per-timestep eps_drift / x0_drift / "
                        "accept_rate. Useful to find where trajectories "
                        "diverge.")
    p.add_argument("--blend_temperature", type=float, default=None,
                   help="Fix 3: soft blend temperature. None disables "
                        "(original hard accept/reject). Suggested sweep: "
                        "0.05 (sharp) / 0.10 (medium) / 0.20 (soft).")
    p.add_argument("--x0_thr_strict", type=float, default=None,
                   help="Fix 4: strict x0 threshold (used near "
                        "x0_strict_center). Suggested: 0.02 or 0.03.")
    p.add_argument("--x0_thr_loose", type=float, default=None,
                   help="Fix 4: loose x0 threshold (used far from "
                        "center). Suggested: 0.07 or 0.10.")
    p.add_argument("--x0_strict_center", type=float, default=None,
                   help="Fix 4: t_norm peak of strictness. None "
                        "disables timestep-dependent gate. Diagnostic "
                        "showed drift spikes near 0.45.")
    p.add_argument("--x0_strict_width", type=float, default=0.12,
                   help="Fix 4: Gaussian width of the strictness peak. "
                        "Default 0.12 covers t/T 0.33-0.57.")
    p.add_argument("--drift_k_switch_threshold", type=float, default=None,
                   help="Fix 5: force spec-1 when x0_drift exceeds this. "
                        "None disables. Try 0.004 / 0.006 / 0.008. "
                        "Diagnostic showed peak drift around 0.009 at t/T=0.45.")
    # [PATCH 8] training-consistent fallback prompt
    p.add_argument("--default_prompt", type=str, default="",
                   help="Fallback prompt when neither COCO captions nor "
                        "--auto_prompt yields one. Set to the prompt used "
                        "during DRAFT TRAINING for train-eval consistency "
                        "(FFHQ: 'a photo of a person').")
    # [PATCH 6] subset of FreqSpec tolerance presets to run
    p.add_argument("--fs_presets", type=str, default="strict,mid,default",
                   help="Comma-separated FreqSpec presets to run: any of "
                        "strict, mid, default. Use a subset (e.g. "
                        "'default') to save time in sensitivity or "
                        "cross-dataset sweeps.")
    # [PATCH 7] mask mode for the large-mask evaluation
    p.add_argument("--mask_mode", type=str, default="train",
                   choices=["train", "large", "coco_object"],
                   help="'train' = training mask distribution (box 40 "
                        "+ brush 60, original). 'large' = large/complex "
                        "masks with 40-60 percent coverage for the "
                        "reviewer-requested robustness evaluation.")
    p.add_argument("--instances_json", type=str, default="",
                   help="[PATCH 9] COCO instances JSON for --mask_mode "
                        "coco_object (object-removal masks from instance "
                        "segmentation polygons).")
    p.add_argument("--large_coverage", type=float, nargs=2,
                   default=[0.40, 0.60],
                   help="Coverage range (lo hi) for --mask_mode large.")
    # [PATCH 3] new flag for usage-map saving
    p.add_argument("--save_usage_maps", action="store_true",
                   help="Save per-image draft usage maps (averaged "
                        "soft-blend weight w(p)) as usage_map.png in each "
                        "freqspec_* sample directory. Required for the "
                        "qualitative figure assembly. Default off.")
    return p


if __name__ == "__main__":
    main(get_parser().parse_args())
