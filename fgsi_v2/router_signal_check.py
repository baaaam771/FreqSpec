#!/usr/bin/env python
"""
router_signal_check.py — B(step-count router)의 전제 검증 (GPU-free).

전제: "FreqSpec의 verifier 신호(수용률/궤적 드리프트)가 aggressive
step-reduction(s20)이 실패할 입력을 사전에 예측할 수 있는가?"

같은 이미지 집합에 대해 per-image CSV가 이미 갖고 있는 것:
  - 신호 후보:  freqspec_* 의 accept, lpips_t   (+ mask_coverage 비교기준)
  - 실패 대상:  target_s20 (또는 --hard 로 지정)의 lpips_t / gt_masked_lpips

보고:
  1) Spearman 상관 (신호 vs s20의 lp_t, gt)
  2) AUROC — "s20 lp_t 상위 25% (실패 위험군)"을 신호로 분류
  3) 간이 router 시뮬레이션: 신호 하위 q% -> s20, 나머지 -> s30
     (calibration 절반에서 q 선택, test 절반에서 평균 speedup·tail 보고,
      비교선: all-s20 / all-s30 / coverage-threshold router)

주의: 신호 추출에 draft+verifier 실행 비용이 들므로, router로서의 실익은
"신호 계산 오버헤드 << s30-s20 시간 차"일 때만 성립. 본 스크립트는 예측력
존재 여부만 판정한다 (예측력이 없으면 그 뒤는 볼 필요도 없음).

Usage:
  python router_signal_check.py --csv largemask_per_image.csv \\
      [--signal_method freqspec_strict] [--hard target_s20] [--mid target_s30]
"""
import argparse
import csv
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr


def auroc(scores, labels):
    s = np.asarray(scores, float)
    y = np.asarray(labels, int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # rank-based AUROC (ties handled by average rank)
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    for v in np.unique(allv):
        m = allv == v
        ranks[m] = ranks[m].mean()
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--signal_method", default="freqspec_strict")
    ap.add_argument("--hard", default="target_s20")
    ap.add_argument("--mid", default="target_s30")
    args = ap.parse_args()

    rows = []
    for p in args.csv:
        rows += list(csv.DictReader(open(p)))
    data = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        data[r["run"]][r["method"]][int(r["idx"])] = r

    for run in sorted(data):
        md = data[run]
        need = [args.signal_method, args.hard, args.mid]
        if any(m not in md for m in need):
            print(f"[router] {run}: missing one of {need} — skip")
            continue
        idxs = sorted(set(md[args.signal_method]) & set(md[args.hard])
                      & set(md[args.mid]))
        f = lambda m, i, k: float(md[m][i][k])
        sig_acc = np.array([f(args.signal_method, i, "accept") for i in idxs])
        sig_lpt = np.array([f(args.signal_method, i, "lpips_t") for i in idxs])
        cov = np.array([f(args.signal_method, i, "mask_coverage")
                        for i in idxs])
        hard_lpt = np.array([f(args.hard, i, "lpips_t") for i in idxs])
        hard_gt = np.array([f(args.hard, i, "gt_masked_lpips") for i in idxs])

        print(f"\n===== {run} (n={len(idxs)}; signal={args.signal_method}, "
              f"hard={args.hard}) =====")
        for name, sig in [("accept(-)", -sig_acc), ("fs_lpips_t", sig_lpt),
                          ("coverage", cov)]:
            for tgt_name, tgt in [("s20 lp_t", hard_lpt),
                                  ("s20 gt_m", hard_gt)]:
                rho, p = spearmanr(sig, tgt)
                print(f"  corr {name:11s} -> {tgt_name:9s}: "
                      f"rho={rho:+.3f} (p={p:.3f})")
        thr = np.percentile(hard_lpt, 75)
        y = (hard_lpt >= thr).astype(int)
        print(f"  AUROC (predict s20 lp_t top-25% risk):")
        print(f"    -accept    : {auroc(-sig_acc, y):.3f}")
        print(f"    fs_lpips_t : {auroc(sig_lpt, y):.3f}")
        print(f"    coverage   : {auroc(cov, y):.3f}   <- cheap baseline")

        # tiny router sim: calib picks q on signal; test reports outcome
        calib = [k for k, i in enumerate(idxs) if i % 2 == 0]
        test = [k for k, i in enumerate(idxs) if i % 2 == 1]
        t_hard = np.array([f(args.hard, i, "time_sec") for i in idxs])
        t_mid = np.array([f(args.mid, i, "time_sec") for i in idxs])
        mid_lpt = np.array([f(args.mid, i, "lpips_t") for i in idxs])
        best = None
        for q in [25, 50, 75]:
            cut = np.percentile(sig_lpt[calib], q)
            easy = sig_lpt <= cut  # low drift -> route to hard (s20)
            lp = np.where(easy, hard_lpt, mid_lpt)
            tt = np.where(easy, t_hard, t_mid)
            w10 = float(np.mean(np.sort(lp[test])[-max(1, len(test)//10):]))
            cand = (float(np.mean(tt[test])), w10, q)
            if best is None or cand[1] < best[1]:
                best = cand
        for nm, lp, tt in [("all-"+args.hard, hard_lpt, t_hard),
                           ("all-"+args.mid, mid_lpt, t_mid)]:
            w10 = float(np.mean(np.sort(lp[test])[-max(1, len(test)//10):]))
            print(f"  {nm:18s}: t_mean={float(np.mean(tt[test])):.3f} "
                  f"lp_t_w10={w10:.4f}")
        print(f"  router(q={best[2]}%)    : t_mean={best[0]:.3f} "
              f"lp_t_w10={best[1]:.4f}")
        print("  판정: AUROC>0.7 이상 + router가 all-s30 대비 시간 단축 & "
              "all-s20 대비 tail 개선이면 B 전제 성립")


if __name__ == "__main__":
    main()
