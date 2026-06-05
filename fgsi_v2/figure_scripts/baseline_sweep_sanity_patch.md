# baseline_sweep.py — optional sanity print patch
#
# Adds a clear "Combo 2 status" print at the start of main() so you can
# verify which fixes are active before the sweep runs. Use this together
# with the usage-map patch in baseline_sweep_patch.md.
# ============================================================


# --------------------------------------------------------------
# PATCH: add this block right after the methods/manifest print
# in main(), i.e., right after the line:
#   print(f"[sweep] methods: {[m['name'] for m in methods]}")
# --------------------------------------------------------------

    # --- Combo 2 sanity print (which fixes are active) ---
    def _flag(v, off="off"):
        return f"on (val={v})" if (v is not None and v != 0.0 and v != "") else off
    print("[sweep] FreqSpec fix configuration:")
    print(f"  Fix 2 (fixed x0 gate)        : {_flag(args.x0_threshold)}")
    print(f"  Fix 3 (soft blend)           : {_flag(args.blend_temperature)}")
    print(f"  Fix 4 (timestep x0 strict)   : "
          f"{_flag(args.x0_strict_center)} "
          f"[strict={args.x0_thr_strict}, loose={args.x0_thr_loose}, "
          f"width={args.x0_strict_width}]")
    print(f"  Fix 4' (mask-interior)       : "
          f"{_flag(args.mask_interior_weight)}")
    print(f"  Fix 5 (drift-aware K-step)   : "
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


# --------------------------------------------------------------
# Expected output when running run_combo2_sweep.sh:
# --------------------------------------------------------------
# [sweep] FreqSpec fix configuration:
#   Fix 2 (fixed x0 gate)        : off
#   Fix 3 (soft blend)           : on (val=0.1)
#   Fix 4 (timestep x0 strict)   : on (val=0.45) [strict=0.02, loose=0.07, width=0.12]
#   Fix 4' (mask-interior)       : on (val=0.5)
#   Fix 5 (drift-aware K-step)   : on (val=0.006) [k_switch_thr=0.6]
#   >>> Active configuration: COMBO 2 (full)
#
# If you see "PARTIAL", check which fix is "off" and add the corresponding
# argument. Note: Fix 2 (fixed x0 gate) being "off" is intentional when
# Fix 4 (timestep-dependent x0) is on -- Fix 4 supersedes the fixed gate.
