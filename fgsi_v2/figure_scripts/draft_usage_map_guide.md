# Extracting Draft Usage Maps for Fig 5 (column 7)

Figure 5 column 7 shows the soft-blend weight `w(p)` averaged across
Phase-2 verified timesteps. This requires `fgsr_inpaint` to expose
the per-step weight map so it can be saved per image.

There are three options (pick whichever is easiest to integrate).

----------------------------------------------------------------------
Option A (recommended): minimal patch to speculative.py
----------------------------------------------------------------------

Add an optional argument to `fgsr_inpaint`:

```python
def fgsr_inpaint(
    target, draft, z_init, cond_z, mask_z, sch,
    ...,
    return_usage_map=False,          # NEW
    ...,
):
    ...
    # Inside the main Phase-2 loop, you already compute a soft-blend
    # weight w_full per verified timestep. Collect those.
    if return_usage_map:
        usage_acc = torch.zeros_like(mask_z)   # [B,1,H_lat,W_lat]
        usage_count = 0
    ...
    # At each verified timestep:
        # after you compute w_full   (shape [B,1,H_lat,W_lat]):
        if return_usage_map:
            usage_acc = usage_acc + w_full.detach()
            usage_count = usage_count + 1
    ...
    # At the end:
    if return_usage_map and usage_count > 0:
        usage_map = (usage_acc / usage_count).clamp(0, 1)
        stats["usage_map"] = usage_map.cpu()    # add to stats dict
    return z_out, stats
```

Then update `baseline_sweep.py` to save the map when present:

```python
# inside run_one(), after the fgsr_inpaint call:
if "usage_map" in stats:
    # upsample to image resolution for visualization
    um = stats["usage_map"]
    um_up = F.interpolate(um, size=(args.image_size, args.image_size),
                          mode="bilinear", align_corners=False)
    save_gray(um_up, os.path.join(m_out, "usage_map.png"))
```

And add `return_usage_map=True` when calling `fgsr_inpaint`:

```python
(z_out, stats), t_run = timed_run(
    lambda: fgsr_inpaint(
        ...,
        return_usage_map=True,    # NEW
    ),
    device,
)
```

After running baseline_sweep again, each freqspec_* sample dir will
have a `usage_map.png` alongside `out.png`, `gt.png`, `mask.png`.

----------------------------------------------------------------------
Option B: save usage maps to a shared directory
----------------------------------------------------------------------

If you prefer not to touch baseline_sweep.py, save the usage maps
inside the same loop and gather them into a single `usage_maps/`
dir using the naming convention expected by
`assemble_qualitative_3datasets.py`:

    usage_maps/
        ffhq_img_NNN_usage.png
        places2_img_NNN_usage.png
        coco_img_NNN_usage.png

Then pass `--usage_map_dir usage_maps/` to the assembly script.

----------------------------------------------------------------------
Option C: skip usage maps for v1
----------------------------------------------------------------------

If you want a first version of Fig 5 without code changes, run the
assembly script without `--usage_map_dir`. Column 7 will render a
gray panel labeled "draft usage map n/a", and the rest of Fig 5
will be a normal 7-method qualitative comparison. The placeholder
note in the caption will need a small adjustment in main.tex; let
me know if you go this route and I'll patch it.

----------------------------------------------------------------------
Which option to pick
----------------------------------------------------------------------

- For maximum visual impact in Fig 5: Option A.
- For minimum code churn: Option B (just save the maps to a side dir).
- For "just compile something now": Option C.

Send your speculative.py and I'll prepare a ready-to-apply diff.
