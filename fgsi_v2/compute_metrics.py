# scripts/compute_pixel_metrics.py 만들기
import torch, os
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

result_dirs = sorted(os.listdir('/mnt/HDD_12TB/bam_ki/results/eval_natural'))
psnr_list, ssim_list = [], []

for d in result_dirs:
    if not d.startswith('img_'): continue
    base_p = f'/mnt/HDD_12TB/bam_ki/results/eval_natural/{d}/out_baseline.png'
    fgsr_p = f'/mnt/HDD_12TB/bam_ki/results/eval_natural/{d}/out_fgsr.png'
    mask_p = f'/mnt/HDD_12TB/bam_ki/results/eval_natural/{d}/mask.png'
    if not all(os.path.exists(p) for p in [base_p, fgsr_p, mask_p]): continue
    
    base = np.array(Image.open(base_p).convert('RGB'))
    fgsr = np.array(Image.open(fgsr_p).convert('RGB'))
    mask = np.array(Image.open(mask_p).convert('L')) > 127
    
    # 마스크 영역만 비교 (baseline을 reference로)
    if mask.sum() > 0:
        m = mask
        p = psnr_fn(base[m], fgsr[m], data_range=255)
        s = ssim_fn(base, fgsr, channel_axis=-1, data_range=255, 
                    win_size=11)  # 전체 image SSIM
        psnr_list.append(p)
        ssim_list.append(s)
        print(f"{d}: PSNR={p:.2f}, SSIM={s:.3f}")

print(f"\n=== Summary ===")
print(f"Avg PSNR (mask region): {np.mean(psnr_list):.2f} dB")
print(f"Avg SSIM (full image): {np.mean(ssim_list):.3f}")