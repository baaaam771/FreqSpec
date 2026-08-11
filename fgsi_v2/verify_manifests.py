#!/usr/bin/env python
"""
verify_manifests.py — Safety check for target-dir reuse (symlink+resume).

Reusing target_s50/s30 from an older run is only valid if BOTH runs sample
the exact same (image_path, prompt, seed) per idx. This compares two
manifest.json files field-by-field and exits nonzero on any mismatch.

Run this as soon as the new sweep has written its manifest.json (it is
written at sweep start, before methods run):

    python verify_manifests.py \\
        --new /mnt/.../main_places2_400k/manifest.json \\
        --old /mnt/.../qualitative_places2_run100/manifest.json

If it FAILS: delete the target symlinks in the new out dir and let the
sweep recompute targets fresh — do not reuse mismatched results.
"""
import argparse
import json
import sys


def load(path):
    with open(path) as f:
        return {int(e["idx"]): e for e in json.load(f)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--new", required=True)
    p.add_argument("--old", required=True)
    p.add_argument("--fields", nargs="+",
                   default=["image_path", "prompt", "seed"])
    args = p.parse_args()

    new, old = load(args.new), load(args.old)
    common = sorted(set(new) & set(old))
    print(f"[verify] new={len(new)} old={len(old)} common={len(common)} idx")
    if not common:
        print("[verify] FAIL: no overlapping idx")
        sys.exit(1)

    bad = 0
    for idx in common:
        for f in args.fields:
            if new[idx].get(f) != old[idx].get(f):
                bad += 1
                if bad <= 5:
                    print(f"[verify] MISMATCH idx={idx} field={f}: "
                          f"new={new[idx].get(f)!r} old={old[idx].get(f)!r}")
    if bad:
        print(f"[verify] FAIL: {bad} mismatching fields — "
              f"target reuse is INVALID. Remove symlinks and recompute.")
        sys.exit(1)
    print("[verify] PASS: manifests identical on "
          f"{args.fields} for all {len(common)} common idx — reuse valid.")


if __name__ == "__main__":
    main()
