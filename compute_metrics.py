"""
FreqSpec-Inpaint quality metrics evaluation script.

Computes pixel-level (PSNR, SSIM) and perceptual (LPIPS) metrics
between baseline and FGSR outputs in a results directory.

Each result directory should contain:
    out_baseline.png
    out_fgsr.png
    mask.png

Usage:
    # 기본 사용 (PSNR + SSIM)
    python compute_metrics.py --root /mnt/HDD_12TB/bam_ki/results/eval_natural

    # LPIPS도 포함
    python compute_metrics.py --root /mnt/HDD_12TB/bam_ki/results/eval_natural --lpips

    # Speedup도 표시 (로그에서 파싱)
    python compute_metrics.py --root /mnt/HDD_12TB/bam_ki/results/eval_natural --speedup_log all_results.txt
"""
import argparse
import os
import re
import sys
import numpy as np
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, required=True,
                   help="결과 폴더의 루트. 그 안에 img_1, img_2, ... 폴더들이 있어야 함.")
    p.add_argument("--prefix", type=str, default="img_",
                   help="결과 폴더 이름 prefix (기본: img_)")
    p.add_argument("--lpips", action="store_true",
                   help="LPIPS도 계산 (lpips 패키지 필요)")
    p.add_argument("--lpips_net", type=str, default="alex",
                   choices=["alex", "vgg"],
                   help="LPIPS backbone network")
    p.add_argument("--speedup_log", type=str, default="",
                   help="speedup이 기록된 로그 파일 (--root 안의 상대 경로). "
                        "비워두면 speedup 표시 안 함.")
    p.add_argument("--save_csv", type=str, default="",
                   help="결과를 CSV로 저장할 경로 (선택)")
    p.add_argument("--verbose", action="store_true",
                   help="각 이미지의 결과 출력")
    return p.parse_args()


def compute_pixel_metrics(base_arr, fgsr_arr, mask_bool):
    """Returns (psnr_mask, ssim_full)."""
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    from skimage.metrics import structural_similarity as ssim_fn

    if mask_bool.sum() == 0:
        return None, None

    # PSNR: 마스크 영역만
    p = psnr_fn(base_arr[mask_bool], fgsr_arr[mask_bool], data_range=255)

    # SSIM: 전체 이미지 (channel_axis 사용)
    try:
        s = ssim_fn(base_arr, fgsr_arr,
                    channel_axis=-1, data_range=255, win_size=11)
    except TypeError:
        # 오래된 skimage 버전 대응
        s = ssim_fn(base_arr, fgsr_arr,
                    multichannel=True, data_range=255, win_size=11)
    return p, s


def to_lpips_tensor(arr, device):
    """[H,W,3] uint8 -> [1,3,H,W] float in [-1,1]"""
    import torch
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 127.5 - 1.0
    return t.unsqueeze(0).to(device)


def parse_speedup_log(log_path):
    """Returns dict {dirname -> {'speedup': float, 'accept_rate': float, 'nfe_target': int}}
    Tries to parse lines like:
        === Image 1: /path/to/image.jpg ===
        time=0.91s  NFE_target=40  NFE_draft=35  accept_rate=0.350  target_speedup=1.25x
        saved -> /path/to/results/img_1
    """
    if not os.path.isfile(log_path):
        return {}
    with open(log_path, "r") as f:
        text = f.read()

    # 블록 단위 파싱
    blocks = re.split(r"===\s*Image\s*\d+:", text)
    info = {}
    for blk in blocks:
        speed_m = re.search(r"target_speedup=([0-9.]+)x", blk)
        acc_m = re.search(r"accept_rate=([0-9.]+)", blk)
        nfe_m = re.search(r"NFE_target=(\d+)", blk)
        save_m = re.search(r"saved\s*->\s*(\S+)", blk)
        if not (speed_m and save_m):
            continue
        dirname = os.path.basename(save_m.group(1).strip())
        info[dirname] = {
            "speedup": float(speed_m.group(1)),
            "accept_rate": float(acc_m.group(1)) if acc_m else None,
            "nfe_target": int(nfe_m.group(1)) if nfe_m else None,
        }
    return info


def main():
    args = parse_args()

    if not os.path.isdir(args.root):
        print(f"ERROR: root not found: {args.root}")
        sys.exit(1)

    # speedup 로그 파싱 (선택)
    speedup_info = {}
    if args.speedup_log:
        log_path = os.path.join(args.root, args.speedup_log)
        speedup_info = parse_speedup_log(log_path)
        if speedup_info:
            print(f"[parsed speedup log] {len(speedup_info)} entries from "
                  f"{args.speedup_log}")

    # LPIPS 준비
    lpips_fn = None
    device = "cpu"
    if args.lpips:
        try:
            import torch
            import lpips
            device = "cuda" if torch.cuda.is_available() else "cpu"
            lpips_fn = lpips.LPIPS(net=args.lpips_net).to(device)
            print(f"[LPIPS loaded] backbone={args.lpips_net}, device={device}")
        except ImportError:
            print("ERROR: lpips package not installed. Run: pip install lpips")
            sys.exit(1)

    # 결과 폴더 수집
    subdirs = sorted([d for d in os.listdir(args.root)
                      if d.startswith(args.prefix)
                      and os.path.isdir(os.path.join(args.root, d))])
    if not subdirs:
        print(f"ERROR: no subdirs with prefix '{args.prefix}' under {args.root}")
        sys.exit(1)

    # 계산
    rows = []
    psnr_list, ssim_list, lpips_list = [], [], []
    speedups, accept_rates = [], []

    print(f"\n{'name':<10} {'PSNR(mask)':>10} {'SSIM(full)':>10}", end="")
    if args.lpips:
        print(f" {'LPIPS':>8}", end="")
    if speedup_info:
        print(f" {'speedup':>8} {'accept':>7}", end="")
    print()
    print("-" * 60)

    for d in subdirs:
        base_p = os.path.join(args.root, d, "out_baseline.png")
        fgsr_p = os.path.join(args.root, d, "out_fgsr.png")
        mask_p = os.path.join(args.root, d, "mask.png")
        if not all(os.path.exists(p) for p in [base_p, fgsr_p, mask_p]):
            if args.verbose:
                print(f"{d}: skip (missing files)")
            continue

        base = np.array(Image.open(base_p).convert("RGB"))
        fgsr = np.array(Image.open(fgsr_p).convert("RGB"))
        mask = np.array(Image.open(mask_p).convert("L")) > 127

        psnr, ssim = compute_pixel_metrics(base, fgsr, mask)
        if psnr is None:
            continue
        psnr_list.append(psnr)
        ssim_list.append(ssim)

        row = {"name": d, "psnr": psnr, "ssim": ssim}

        # LPIPS
        if lpips_fn is not None:
            import torch
            with torch.no_grad():
                l = lpips_fn(to_lpips_tensor(base, device),
                             to_lpips_tensor(fgsr, device)).item()
            lpips_list.append(l)
            row["lpips"] = l

        # Speedup
        if d in speedup_info:
            row["speedup"] = speedup_info[d]["speedup"]
            row["accept_rate"] = speedup_info[d]["accept_rate"]
            speedups.append(row["speedup"])
            if row["accept_rate"] is not None:
                accept_rates.append(row["accept_rate"])

        rows.append(row)

        # 출력
        line = f"{d:<10} {psnr:>10.2f} {ssim:>10.3f}"
        if "lpips" in row:
            line += f" {row['lpips']:>8.4f}"
        if "speedup" in row:
            line += f" {row['speedup']:>7.2f}x {row['accept_rate']:>7.3f}"
        print(line)

    n = len(rows)
    if n == 0:
        print("ERROR: no valid rows found.")
        sys.exit(1)

    # Summary
    print("\n" + "=" * 60)
    print(f"Summary  (n={n})")
    print("=" * 60)
    print(f"PSNR (mask region):  {np.mean(psnr_list):6.2f} ± {np.std(psnr_list):.2f} dB  "
          f"[min={min(psnr_list):.2f}, max={max(psnr_list):.2f}]")
    print(f"SSIM (full image):    {np.mean(ssim_list):6.3f} ± {np.std(ssim_list):.3f}    "
          f"[min={min(ssim_list):.3f}, max={max(ssim_list):.3f}]")
    if lpips_list:
        print(f"LPIPS ({args.lpips_net}):         {np.mean(lpips_list):6.4f} ± "
              f"{np.std(lpips_list):.4f}  "
              f"[min={min(lpips_list):.4f}, max={max(lpips_list):.4f}]")
    if speedups:
        print(f"Speedup:              {np.mean(speedups):6.2f}x ± {np.std(speedups):.2f}  "
              f"[min={min(speedups):.2f}, max={max(speedups):.2f}]")
    if accept_rates:
        print(f"Accept rate:          {np.mean(accept_rates):6.3f} ± "
              f"{np.std(accept_rates):.3f}  "
              f"[min={min(accept_rates):.3f}, max={max(accept_rates):.3f}]")
    print("=" * 60)

    # CSV 저장
    if args.save_csv:
        keys = ["name", "psnr", "ssim"]
        if lpips_list:
            keys.append("lpips")
        if speedups:
            keys += ["speedup", "accept_rate"]
        with open(args.save_csv, "w") as f:
            f.write(",".join(keys) + "\n")
            for r in rows:
                f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
        print(f"[saved] {args.save_csv}")


if __name__ == "__main__":
    main()