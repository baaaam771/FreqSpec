#!/usr/bin/env python
"""
SDXL/SD2 평가 자동화 스크립트.

- N장 이미지를 dataset에서 sample
- 각 이미지로 run_inpaint.py 실행
- 결과 로그 + CSV로 저장
- Resume 지원 (중간에 멈춰도 이어서)
- 마지막에 요약 통계 출력

Usage:
    # SDXL 20장 평가
    python eval_runner.py \\
        --target_id /mnt/HDD_12TB/bam_ki/checkpoints/stable-diffusion-xl-1.0-inpainting-0.1 \\
        --draft_ckpt /mnt/HDD_12TB/bam_ki/runs/sdxl_v1/draft_step0290000.pt \\
        --data_root /mnt/HDD_12TB/bam_ki/datasets/places2 \\
        --out_root /mnt/HDD_12TB/bam_ki/results/sdxl_eval_natural \\
        --num_images 20 \\
        --image_size 1024 \\
        --guidance_scale 7.5 \\
        --K 3 --tol_low 0.03 --tol_high 0.3

    # SD2 20장 평가 (image_size 512, no CFG)
    python eval_runner.py \\
        --target_id /mnt/HDD_12TB/bam_ki/checkpoints/stable-diffusion-2-inpainting \\
        --draft_ckpt /mnt/HDD_12TB/bam_ki/runs/freqspec_v2/draft_step0150000.pt \\
        --data_root /mnt/HDD_12TB/bam_ki/datasets/places2 \\
        --out_root /mnt/HDD_12TB/bam_ki/results/sd2_eval_rerun \\
        --num_images 20 \\
        --image_size 512 \\
        --guidance_scale 1.0 \\
        --K 3 --tol_low 0.03 --tol_high 0.3
"""
import argparse
import csv
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    # 평가 대상
    p.add_argument("--target_id", type=str, required=True)
    p.add_argument("--draft_ckpt", type=str, required=True)

    # 데이터셋
    p.add_argument("--data_root", type=str, required=True,
                   help="이미지 폴더의 루트 (재귀 탐색)")
    p.add_argument("--image_ext", type=str, default="jpg",
                   choices=["jpg", "jpeg", "png", "webp"])
    p.add_argument("--num_images", type=int, default=20)
    p.add_argument("--seed", type=int, default=42,
                   help="이미지 샘플링 seed (재현 가능성)")

    # 출력
    p.add_argument("--out_root", type=str, required=True)
    p.add_argument("--resume", action="store_true",
                   help="기존 결과 폴더 있으면 건너뛰기")

    # 추론 인자
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--image_size", type=int, default=1024)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--tol_low", type=float, default=0.03)
    p.add_argument("--tol_high", type=float, default=0.3)
    p.add_argument("--uniform_saliency", action="store_true",
                   help="Ablation: turn off wavelet saliency.")
    p.add_argument("--no_boundary", action="store_true",
                   help="Ablation: turn off boundary indicator.")
    p.add_argument("--t_spec_start", type=float, default=0.7,
                   help="Phase 1 stabilization 종료 시점 (normalized t).")
    p.add_argument("--patch", type=int, default=4,
                   help="Saliency patch size.")
    p.add_argument("--beta", type=float, default=10.0,
                   help="Acceptance threshold sharpness.")
    p.add_argument("--use_ema_draft", action="store_true", default=True,
                   help="EMA draft 사용 (기본 켜짐)")
    p.add_argument("--no_ema_draft", action="store_true",
                   help="EMA 사용 안 함")
    p.add_argument("--auto_prompt", action="store_true", default=True,
                   help="path에서 카테고리 자동 추출 (Places2)")
    p.add_argument("--fixed_prompt", type=str, default="",
                   help="auto_prompt 대신 모든 이미지에 같은 prompt 사용")

    # run_inpaint.py 위치 (module 경로)
    p.add_argument("--run_inpaint_module", type=str, default="inference.run_inpaint",
                   help="python -m <module> 형식")

    p.add_argument("--max_retries", type=int, default=2,
                   help="이미지당 최대 재시도 횟수")
    return p.parse_args()


def collect_image_paths(data_root, ext, n, seed):
    """data_root에서 ext 파일들을 모아 n장 sample."""
    paths = []
    for root, _, files in os.walk(data_root):
        for f in files:
            if f.lower().endswith(f".{ext.lower()}"):
                paths.append(os.path.join(root, f))
    if not paths:
        raise RuntimeError(f"no .{ext} files found under {data_root}")
    rng = random.Random(seed)
    rng.shuffle(paths)
    return paths[:n]


def parse_inference_output(text):
    """run_inpaint.py 출력에서 metric 추출.
    Returns dict with keys: baseline_time, baseline_nfe,
    fgsr_time, fgsr_nfe_target, fgsr_nfe_draft, accept_rate, speedup.
    None for missing fields.
    """
    out = {
        "baseline_time": None, "baseline_nfe": None,
        "fgsr_time": None, "fgsr_nfe_target": None, "fgsr_nfe_draft": None,
        "accept_rate": None, "speedup": None,
    }
    # baseline 줄: "time=X.XXs  NFE_target=NN"
    m = re.search(
        r"baseline.*?\n.*?time=([\d.]+)s\s+NFE_target=(\d+)",
        text, re.DOTALL)
    if m:
        out["baseline_time"] = float(m.group(1))
        out["baseline_nfe"] = int(m.group(2))
    # FGSR 줄
    m = re.search(
        r"FGSR:\s*\n.*?time=([\d.]+)s\s+NFE_target=(\d+)\s+NFE_draft=(\d+)\s+"
        r"accept_rate=([\d.]+)\s+target_speedup=([\d.]+)x",
        text, re.DOTALL)
    if m:
        out["fgsr_time"] = float(m.group(1))
        out["fgsr_nfe_target"] = int(m.group(2))
        out["fgsr_nfe_draft"] = int(m.group(3))
        out["accept_rate"] = float(m.group(4))
        out["speedup"] = float(m.group(5))
    return out


def run_one(args, img_path, out_dir, log_path):
    """한 이미지에 대해 run_inpaint.py 실행. 결과는 log_path에 저장.
    Returns (success, metrics_dict)."""
    cmd = [
        sys.executable, "-m", args.run_inpaint_module,
        "--target_id", args.target_id,
        "--image", img_path,
        "--draft_ckpt", args.draft_ckpt,
        "--device", args.device,
        "--num_steps", str(args.num_steps),
        "--image_size", str(args.image_size),
        "--guidance_scale", str(args.guidance_scale),
        "--K", str(args.K),
        "--tol_low", str(args.tol_low),
        "--tol_high", str(args.tol_high),
        "--t_spec_start", str(args.t_spec_start),
        "--patch", str(args.patch),
        "--beta", str(args.beta),
        "--out_dir", out_dir,
    ]
    if args.use_ema_draft and not args.no_ema_draft:
        cmd.append("--use_ema_draft")
    if args.uniform_saliency:
        cmd.append("--uniform_saliency")
    if args.no_boundary:
        cmd.append("--no_boundary")
    if args.fixed_prompt:
        cmd += ["--prompt", args.fixed_prompt]
    elif args.auto_prompt:
        cmd.append("--auto_prompt")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False, timeout=600)
    except subprocess.TimeoutExpired:
        return False, {"error": "timeout"}

    stdout = result.stdout or ""
    stderr = result.stderr or ""
    full_log = f"=== CMD ===\n{' '.join(cmd)}\n=== STDOUT ===\n{stdout}\n=== STDERR ===\n{stderr}\n"
    with open(log_path, "w") as f:
        f.write(full_log)

    if result.returncode != 0:
        return False, {"error": f"returncode {result.returncode}"}

    metrics = parse_inference_output(stdout)
    return True, metrics


def main():
    args = parse_args()
    os.makedirs(args.out_root, exist_ok=True)

    # 이미지 sample
    print(f"[eval_runner] sampling {args.num_images} images from {args.data_root}")
    img_paths = collect_image_paths(
        args.data_root, args.image_ext, args.num_images, args.seed)
    print(f"[eval_runner] selected {len(img_paths)} images")

    # CSV header
    csv_path = os.path.join(args.out_root, "results.csv")
    log_summary_path = os.path.join(args.out_root, "summary.log")
    csv_fields = [
        "idx", "img_path", "out_dir", "status",
        "baseline_time", "baseline_nfe",
        "fgsr_time", "fgsr_nfe_target", "fgsr_nfe_draft",
        "accept_rate", "speedup",
    ]
    # resume 모드면 기존 CSV 읽기
    completed = set()
    if args.resume and os.path.isfile(csv_path):
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("status") == "ok":
                    completed.add(int(row["idx"]))
        print(f"[eval_runner] resume: {len(completed)} completed")

    # 새 파일이면 header 쓰기
    csv_exists = os.path.isfile(csv_path)
    csv_file = open(csv_path, "a" if csv_exists else "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    if not csv_exists:
        writer.writeheader()
        csv_file.flush()

    summary_log = open(log_summary_path, "a")

    # 실행 loop
    t_start = time.time()
    n_ok, n_fail = 0, 0
    for i, img_path in enumerate(img_paths, 1):
        if i in completed:
            print(f"[{i}/{args.num_images}] SKIP (already done): {img_path}")
            continue

        out_dir = os.path.join(args.out_root, f"img_{i:03d}")
        log_path = os.path.join(args.out_root, f"img_{i:03d}.log")

        line = f"\n=== [{i}/{args.num_images}] {img_path} ==="
        print(line)
        summary_log.write(line + "\n")
        summary_log.flush()

        success = False
        metrics = {}
        for retry in range(args.max_retries):
            t0 = time.time()
            success, metrics = run_one(args, img_path, out_dir, log_path)
            dt = time.time() - t0
            if success:
                break
            print(f"  retry {retry+1}/{args.max_retries} after failure: {metrics.get('error')}")

        status = "ok" if success else "fail"
        row = {
            "idx": i,
            "img_path": img_path,
            "out_dir": out_dir,
            "status": status,
            "baseline_time": metrics.get("baseline_time", ""),
            "baseline_nfe": metrics.get("baseline_nfe", ""),
            "fgsr_time": metrics.get("fgsr_time", ""),
            "fgsr_nfe_target": metrics.get("fgsr_nfe_target", ""),
            "fgsr_nfe_draft": metrics.get("fgsr_nfe_draft", ""),
            "accept_rate": metrics.get("accept_rate", ""),
            "speedup": metrics.get("speedup", ""),
        }
        writer.writerow(row)
        csv_file.flush()

        if success:
            n_ok += 1
            msg = (f"  OK  time={dt:.1f}s  "
                   f"speedup={metrics.get('speedup')}x  "
                   f"accept={metrics.get('accept_rate')}  "
                   f"NFE_t={metrics.get('fgsr_nfe_target')}")
        else:
            n_fail += 1
            msg = f"  FAIL  ({metrics.get('error', 'unknown')})  see {log_path}"
        print(msg)
        summary_log.write(msg + "\n")
        summary_log.flush()

    csv_file.close()

    # 요약 통계
    t_total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Eval done: {n_ok} ok, {n_fail} fail, {t_total:.1f}s")
    print(f"Results: {csv_path}")
    print(f"{'='*60}")
    summary_log.write(
        f"\n=== DONE ===\nok={n_ok} fail={n_fail} time={t_total:.1f}s\n")
    summary_log.close()

    # CSV에서 average 계산
    if n_ok > 0:
        speedups, accepts, nfe_ts = [], [], []
        with open(csv_path, "r") as f:
            for row in csv.DictReader(f):
                if row.get("status") != "ok":
                    continue
                try:
                    speedups.append(float(row["speedup"]))
                    accepts.append(float(row["accept_rate"]))
                    nfe_ts.append(int(row["fgsr_nfe_target"]))
                except (ValueError, KeyError):
                    pass
        if speedups:
            import statistics
            print(f"\n=== Average over {len(speedups)} successful runs ===")
            print(f"Speedup:     {statistics.mean(speedups):.3f}x "
                  f"± {statistics.stdev(speedups) if len(speedups)>1 else 0:.3f}")
            print(f"Accept rate: {statistics.mean(accepts):.3f} "
                  f"± {statistics.stdev(accepts) if len(accepts)>1 else 0:.3f}")
            print(f"NFE_target:  {statistics.mean(nfe_ts):.1f} "
                  f"± {statistics.stdev(nfe_ts) if len(nfe_ts)>1 else 0:.1f}")
            print(f"\nNext: python compute_metrics.py --root {args.out_root} --lpips")


if __name__ == "__main__":
    main()