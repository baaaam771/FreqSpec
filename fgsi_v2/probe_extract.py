#!/usr/bin/env python
"""
probe_extract.py — Draft-only probe: "s12가 실패할 입력"을 target 호출 0회로
예측할 수 있는가? (경로 2B)

같은 디렉터리의 baseline_sweep.py(=v5)를 모듈로 import하여 기존 sweep과
정확히 동일한 manifest/마스크/seed를 재사용한다. Draft forward 몇 번만으로
per-image feature를 뽑고, 기존 per-image CSV의 paired ΔGT(s12−s24) 라벨과
붙여 AUROC(calib/test 분리, coverage 대비 증분)을 보고한다.

Probe 구성(모두 draft-only, target UNet 호출 0회 — VAE 인코딩만 사용):
  z_mix = z0·(1−m) + cond_z·m  (known 영역은 원본 latent, hole은 masked latent)
  z_t   = sqrt(ᾱ_t)·z_mix + sqrt(1−ᾱ_t)·ε,  t_norm ∈ {0.9, 0.7, 0.5}
  각 t에서 draft ε̂ → x̂0로 변환 후:
    f1  eps_norm_in      : ‖ε̂‖ (mask 내부 평균)
    f2  eps_ratio        : mask 내부/외부 ε̂-norm 비
    f3  x0_err_known     : known 영역에서 |x̂0 − z0| (draft 신뢰도 프록시)
    f4  x0_bnd_disc      : mask 경계 밴드의 |∇x̂0| (경계 불연속)
    f5  x0_hf_in         : DWT 고주파 에너지 (mask 내부)
    f6  selfcons         : t=0.7에서 noise 2-draw 간 |x̂0−x̂0'| (mask 내부)
  + mask_coverage (공짜 baseline으로 함께 기록)

사용 (sweep을 돌린 것과 같은 인자 + probe 인자; out_root의 manifest.json 재사용):
  python probe_extract.py \
      --target_id <sweep과 동일> --draft_ckpt <동일> --data_root <동일> \
      --out_root /mnt/HDD_12TB/bam_ki/results/objectremoval_coco \
      --mask_mode coco_object --use_ema_draft [나머지 마스크 인자 sweep sh와 동일] \
      --per_image_csv instmask_per_image.csv \
      --probe_out probe_instmask.csv
분석만 다시(추출 재사용): --analyze_only 추가.

주의: draft forward 시그니처는 코드베이스에 맞춰 DRAFT_CALL 한 곳에서 조정.
실행 전 확인:  python -c "import inspect; from models.draft import DraftEpsUNet; \
print(inspect.signature(DraftEpsUNet.forward))"
"""
import csv
import json
import os
import time

import numpy as np
import torch

import baseline_sweep as bs


# ---------------------------------------------------------------- draft call
def DRAFT_CALL(draft, z_t, t_idx, cond_z, mask_z):
    """>>> 코드베이스의 draft.forward 시그니처에 맞게 이 한 줄만 조정 <<<
    (draft는 text embedding을 받지 않음 — z_t, timestep, masked latent, mask)"""
    t_tensor = torch.full((z_t.shape[0],), int(t_idx),
                          device=z_t.device, dtype=torch.long)
    return draft(z_t, t_tensor, cond_z, mask_z)


def _alpha_bar(sch, t_idx):
    for name in ("alphas_cumprod", "alpha_bar", "abar", "alphas_bar"):
        a = getattr(sch, name, None)
        if a is not None:
            return a[t_idx]
    raise AttributeError(
        f"DDPMSchedule에서 alpha-bar 텐서를 못 찾음. dir(sch)={dir(sch)}")


def _grad_mag(x):
    gx = x[..., :, 1:] - x[..., :, :-1]
    gy = x[..., 1:, :] - x[..., :-1, :]
    return gx.abs().mean().item() + gy.abs().mean().item()


def _boundary_band(mask_z, width=2):
    """mask 경계 밴드(내외 width픽셀, latent 해상도)."""
    m = (mask_z > 0.5).float()
    k = 2 * width + 1
    dil = torch.nn.functional.max_pool2d(m, k, stride=1, padding=width)
    ero = 1 - torch.nn.functional.max_pool2d(1 - m, k, stride=1, padding=width)
    return (dil - ero).clamp(0, 1)


def extract_features(target, draft, sch, dwt, item, args, device):
    img, mask_pix, z0, mask_z, cond_z, z_init = bs._prepare_latents(
        target, sch, item, args, device)
    pdtype = next(draft.parameters()).dtype
    z0f, condf, mf = (x.to(pdtype) for x in (z0, cond_z, mask_z))
    z_mix = z0f * (1 - mf) + condf * mf
    band = _boundary_band(mf)
    inside = (mf > 0.5)
    n_in = inside.sum().clamp(min=1)
    T = sch.num_train_timesteps

    feats = {"mask_coverage": float(mask_pix.mean().item())}
    t0 = time.time()
    x0_cache = {}
    with torch.no_grad():
        for t_norm in (0.9, 0.7, 0.5):
            t_idx = int(t_norm * (T - 1))
            ab = _alpha_bar(sch, t_idx).to(pdtype)
            gen = torch.Generator(device="cpu").manual_seed(item["seed"] + 777)
            eps = torch.randn(z_mix.shape, generator=gen).to(device, pdtype)
            z_t = ab.sqrt() * z_mix + (1 - ab).sqrt() * eps
            e_hat = DRAFT_CALL(draft, z_t, t_idx, condf, mf)
            x0 = (z_t - (1 - ab).sqrt() * e_hat) / ab.sqrt()
            x0_cache[t_norm] = x0
            tag = f"t{int(t_norm*100)}"
            e_in = (e_hat.pow(2) * inside).sum() / n_in
            e_out = (e_hat.pow(2) * (~inside)).sum() / (~inside).sum().clamp(min=1)
            feats[f"eps_norm_in_{tag}"] = float(e_in.sqrt().item())
            feats[f"eps_ratio_{tag}"] = float((e_in / e_out.clamp(min=1e-8)).item())
            feats[f"x0_err_known_{tag}"] = float(
                (((x0 - z0f).abs() * (~inside)).sum()
                 / (~inside).sum().clamp(min=1)).item())
            feats[f"x0_bnd_disc_{tag}"] = _grad_mag(x0 * band)
            try:
                ll, hh = dwt(x0.float())
                feats[f"x0_hf_in_{tag}"] = float(
                    (hh.abs().mean(1, keepdim=True)
                     * torch.nn.functional.interpolate(
                         mf.float(), size=hh.shape[-2:])).mean().item())
            except Exception:
                feats[f"x0_hf_in_{tag}"] = float("nan")
        # self-consistency at t=0.7 with a 2nd noise draw
        t_idx = int(0.7 * (T - 1))
        ab = _alpha_bar(sch, t_idx).to(pdtype)
        gen = torch.Generator(device="cpu").manual_seed(item["seed"] + 778)
        eps2 = torch.randn(z_mix.shape, generator=gen).to(device, pdtype)
        z_t2 = ab.sqrt() * z_mix + (1 - ab).sqrt() * eps2
        e2 = DRAFT_CALL(draft, z_t2, t_idx, condf, mf)
        x0b = (z_t2 - (1 - ab).sqrt() * e2) / ab.sqrt()
        feats["selfcons_t70"] = float(
            (((x0_cache[0.7] - x0b).abs() * inside).sum() / n_in).item())
    feats["probe_time_sec"] = time.time() - t0
    return feats


# ------------------------------------------------------------------ analysis
def _auroc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, int)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    o = s.argsort()
    r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    for v in np.unique(s):
        m = s == v
        r[m] = r[m].mean()
    npos = y.sum()
    return float((r[y == 1].sum() - npos * (npos + 1) / 2)
                 / (npos * (len(y) - npos)))


def _auprc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, int)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def _logistic(X, y, iters=3000, lr=0.1):
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    X = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        w -= lr * X.T @ (p - y) / len(y)
    return w, X @ w


def analyze(probe_csv, per_image_csv, hard="target_s12", ref="target_s24",
            tau=0.02, boots=1000):
    P = {(r["run"], int(r["idx"])): r for r in csv.DictReader(open(probe_csv))}
    rows = list(csv.DictReader(open(per_image_csv)))
    runs = sorted({r["run"] for r in rows})
    for run in runs:
        R = {m: {int(r["idx"]): r for r in rows
                 if r["run"] == run and r["method"] == m}
             for m in {r["run"] == run and r["method"] for r in rows}}
        R = {m: {int(r["idx"]): r for r in rows
                 if r["run"] == run and r["method"] == m}
             for m in (hard, ref)}
        idx = sorted(i for i in (set(R[hard]) & set(R[ref]))
                     if (run, i) in P)
        if len(idx) < 10:
            print(f"[probe] {run}: matched n={len(idx)} — skip")
            continue
        y = np.array([1 if (float(R[hard][i]["gt_masked_lpips"])
                            - float(R[ref][i]["gt_masked_lpips"])) > tau
                      else 0 for i in idx])
        fnames = [k for k in P[(run, idx[0])]
                  if k not in ("run", "idx", "probe_time_sec")]
        X = np.array([[float(P[(run, i)][k]) for k in fnames] for i in idx])
        X = np.nan_to_num(X)
        cov = X[:, fnames.index("mask_coverage")]
        cal = np.array([i % 2 == 0 for i in idx])
        te = ~cal
        print(f"\n===== {run}  n={len(idx)}  fail={y.sum()} "
              f"({y.mean()*100:.0f}%)  probe_t="
              f"{np.mean([float(P[(run,i)]['probe_time_sec']) for i in idx]):.3f}s =====")
        scored = sorted(
            ((_auroc(X[:, j], y), fnames[j]) for j in range(len(fnames))),
            reverse=True)
        for a, nm in scored[:8]:
            print(f"  {nm:22s} AUROC(all)={a:.3f}")
        # combo (probe-only, no coverage) vs coverage — fit calib, eval test
        pj = [j for j, nm in enumerate(fnames) if nm != "mask_coverage"]
        w, _ = _logistic(X[cal][:, pj], y[cal])
        Xt = (X[te][:, pj] - X[cal][:, pj].mean(0)) / (X[cal][:, pj].std(0) + 1e-8)
        s_probe = np.hstack([Xt, np.ones((te.sum(), 1))]) @ w
        a_probe = _auroc(s_probe, y[te])
        a_cov = _auroc(cov[te], y[te])
        # bootstrap CI of (probe − coverage) AUROC on test
        rng = np.random.default_rng(0)
        diffs = []
        ti = np.where(te)[0]
        for _ in range(boots):
            b = rng.integers(0, len(ti), len(ti))
            diffs.append(_auroc(s_probe[b], y[te][b])
                         - _auroc(cov[te][b], y[te][b]))
        diffs = np.array([d for d in diffs if not np.isnan(d)])
        lo, hi = np.percentile(diffs, [2.5, 97.5]) if len(diffs) else (np.nan,)*2
        print(f"  TEST: probe_combo AUROC={a_probe:.3f}  coverage={a_cov:.3f}  "
              f"Δ={a_probe-a_cov:+.3f} [{lo:+.3f},{hi:+.3f}]")
        print(f"  TEST AUPRC: probe={_auprc(s_probe, y[te]):.3f}  "
              f"coverage={_auprc(cov[te], y[te]):.3f}  "
              f"(base rate={y[te].mean():.2f})")
        print(f"  통과선: probe ≥0.75 & Δ vs coverage ≥ +0.05 (CI>0)")


def main():
    p = bs.get_parser()
    p.add_argument("--per_image_csv", type=str, required=True)
    p.add_argument("--probe_out", type=str, required=True)
    p.add_argument("--label_hard", type=str, default="target_s12")
    p.add_argument("--label_ref", type=str, default="target_s24")
    p.add_argument("--tau", type=float, default=0.02)
    p.add_argument("--analyze_only", action="store_true")
    args = p.parse_args()

    if not args.analyze_only:
        device = torch.device(args.device)
        tdt = {"fp16": torch.float16, "bf16": torch.bfloat16,
               "fp32": torch.float32}[args.target_dtype]
        target = bs.TargetWrapper(model_id=args.target_id, device=device,
                                  dtype=tdt)
        sch = bs.DDPMSchedule(
            num_train_timesteps=target.scheduler_ref.config.num_train_timesteps,
            beta_start=target.scheduler_ref.config.beta_start,
            beta_end=target.scheduler_ref.config.beta_end,
            beta_schedule=target.scheduler_ref.config.beta_schedule,
            device=device)
        dk = {"latent_ch": target.latent_ch,
              "num_train_timesteps": sch.num_train_timesteps}
        ck = torch.load(args.draft_ckpt, map_location=device)
        sa = ck.get("args", {})
        if "draft_base_ch" in sa:
            dk["base_ch"] = sa["draft_base_ch"]
            dk["ch_mult"] = tuple(sa["draft_ch_mult"])
            dk["t_dim"] = sa["draft_t_dim"]
        draft = bs.DraftEpsUNet(**dk).to(device).eval()
        draft.load_state_dict(ck["ema_draft"] if args.use_ema_draft
                              and ck.get("ema_draft") is not None
                              else ck["draft"])
        dwt = bs.DWT2D("haar").to(device)
        if getattr(args, "mask_mode", "train") == "coco_object":
            args._inst_map = bs._load_coco_instance_map(args.instances_json)

        manifest = json.load(open(os.path.join(args.out_root, "manifest.json")))
        run_name = os.path.basename(os.path.normpath(args.out_root))
        first = True
        with open(args.probe_out, "w", newline="") as fo:
            for item in manifest:
                feats = extract_features(target, draft, sch, dwt, item,
                                         args, device)
                if first:
                    w = csv.DictWriter(fo, fieldnames=["run", "idx"]
                                       + list(feats))
                    w.writeheader(); first = False
                w.writerow({"run": run_name, "idx": item["idx"], **feats})
                fo.flush()
                print(f"[probe] {run_name} img {item['idx']:03d} "
                      f"t={feats['probe_time_sec']:.3f}s")
    analyze(args.probe_out, args.per_image_csv,
            hard=args.label_hard, ref=args.label_ref, tau=args.tau)


if __name__ == "__main__":
    main()
