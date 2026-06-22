# freqspec_core: task-agnostic verifier (AAAI Phase 1)

Phase-1 of the AAAI generalization extracts FreqSpec's per-step acceptance logic
into `freqspec_core/verifier.py`, a module that knows nothing about the task
(inpainting vs. super-resolution), the backbone, or the sampler schedule. This
is what lets the paper state:

> The same verifier is used across inpainting and super-resolution without
> task-specific acceptance logic.

## The four primitives (per the plan)

```
compute_agreement(eps_d, eps_t, x0_d, x0_t, patch_size, beta) -> (s_eps, d_x0)
compute_acceptance(s_eps, d_x0, saliency, region, t_norm, cfg) -> (accept, w)
blend_predictions(eps_d, eps_t, w_full)                        -> eps_mix
compute_target_usage(accept, region)                          -> accept_rate
```

`s_eps` is the epsilon-agreement score (Eq. 8), `d_x0` the predicted-clean-latent
disagreement (Eq. 11). `compute_acceptance` folds the saliency-modulated
tolerance (Eq. 7), the x0 gate (Eq. 12, static or timestep-Gaussian Eq. 15), and
the soft-blend weight (Eq. 13). All four are pure tensor functions with no model
or sampler dependency.

`verify_step(...)` composes them so each task calls the verifier exactly once per
verified timestep:

```python
out = verify_step(eps_draft, eps_target, x0_draft, x0_target,
                  saliency_patch, region_patch, t_norm, cfg)
eps_mix, accept_rate = out["eps_mix"], out["accept_rate"]
```

Inpainting passes `region = mask`; super-resolution passes `region = ones`. There
is no other task branch in the acceptance path.

## What stays in the sampler (by design)

The verifier owns the *per-step* decision. Trajectory-level controls — the
two-phase target-only warm-up and the drift-aware K-step lookahead — remain in
the sampler loop (`inference/speculative_general.py`), which calls the primitives
at each verified timestep. This is the correct seam: lookahead spans multiple
steps and is not a per-patch decision.

## Equivalence guarantee

`inference/speculative_general.fgsr_refine` now routes its decision through
`verify_step`. A bit-for-bit check (deterministic stub target/draft, blending
disabled to remove the known-region re-noising RNG) confirms the refactored
sampler reproduces the WACV `inference/speculative.fgsr_inpaint` output exactly
(`max|Δz| = 0`, identical target NFE and accept rate). The WACV file itself is
untouched, so all existing inpainting results remain reproducible.

## Layout

```
freqspec_core/
  verifier.py            # 4 primitives + VerifierConfig + verify_step  (task-agnostic)
inference/
  speculative.py         # WACV inpainting sampler (untouched, validated)
  speculative_general.py # AAAI sampler; calls freqspec_core; region abstraction
  run_sr.py              # SR runner
models/
  target_wrapper.py      # inpainting target (untouched)
  sr_target_wrapper.py   # x4-upscaler target
  draft.py               # generalized draft (cond_ch / use_mask)
training/
  train.py               # inpainting draft training (untouched)
  train_sr.py            # SR draft training
docs/
  sr_extension.md        # SR task design + experiment plan
  verifier_core.md       # this file
```

## Status

Phase 1 (task-agnostic core) and Phase 2/3 (SR task adapter + runner + training)
are implemented and smoke-tested in dummy mode. Phase 4 (DiT token-wise PoC) is
not started; the verifier core was written so a token-grid instantiation only
needs a token analogue of patch pooling plus a DiT target/draft wrapper pair —
no change to the acceptance primitives.
