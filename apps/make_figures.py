#!/usr/bin/env python
"""Run each exercise script and save all figures it produces to docs/figures/.

By default only figures are saved. Pass -o to also write an org-mode document
(docs/exercises.org) that embeds every figure under a heading named after its
source script.

Scripts ending in _no_test.py are skipped.
"""

import matplotlib
matplotlib.use("Agg")

import argparse
import runpy
import sys
from pathlib import Path
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    description=(
        "Run every exercise script in apps/exercises/ and save the figures "
        "they produce as PNGs to docs/figures/. "
        "Scripts ending in _no_test.py are skipped."
    )
)
args = parser.parse_args()

EXERCISES_DIR = Path(__file__).parent / "exercises"
OUT_DIR = Path(__file__).parent.parent / "docs" / "figures"
ORG_FILE = Path(__file__).parent.parent / "docs" / "figures.org"

OUT_DIR.mkdir(parents=True, exist_ok=True)

scripts = sorted(p for p in EXERCISES_DIR.glob("*.py") if not p.stem.endswith("_no_test"))

results = []   # list of (stem, [png_paths], error_or_None)
total_figures = 0

for script in scripts:
    plt.close("all")
    error = None
    try:
        runpy.run_path(str(script))
    except Exception as e:
        error = e
        print(f"  ERROR in {script.name}: {e}", file=sys.stderr)

    fig_nums = plt.get_fignums()
    png_paths = []
    for i, num in enumerate(fig_nums):
        fname = f"{script.stem}_fig{i}.png"
        fpath = OUT_DIR / fname
        plt.figure(num).savefig(fpath, bbox_inches="tight")
        png_paths.append(fpath)

    total_figures += len(png_paths)
    results.append((script.stem, png_paths, error))

    status = "ERROR" if error else f"{len(png_paths)} figure(s)"
    print(f"{script.name}: {status}")

with ORG_FILE.open("w") as f:
    f.write("#+TITLE: Exercise Figures\n")
    f.write("#+OPTIONS: ^:nil\n\n")

    for stem, png_paths, error in results:
        f.write(f"* {stem}\n\n")
        if error:
            f.write(f"#+begin_quote\nERROR: {error}\n#+end_quote\n\n")
        elif not png_paths:
            f.write("(no figures produced)\n\n")
        else:
            for p in png_paths:
                f.write(f"[[file:figures/{p.name}]]\n\n")
print(f"Org file written to {ORG_FILE}")

errors = sum(1 for _, _, e in results if e is not None)
print(f"\nDone: {len(scripts)} scripts, {total_figures} figures saved to {OUT_DIR}")
if errors:
    print(f"{errors} script(s) had errors — see above.", file=sys.stderr)
    sys.exit(1)
