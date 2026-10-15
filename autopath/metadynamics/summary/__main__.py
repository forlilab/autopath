"""python -m autopath.metadynamics.summary RUN_DIR [--funnel F] [--learned-cv L] [--burn 0.2] [--t-rej 0.2] [--step-ns 1] [--min-points 12] [--min-count 0] [--min-bin-ess 0] [--min-density 0.05] [--bins N] [--bins-per-hill 4] [--bins-fallback 60] [--hill-mask-sigmas 2] [--smooth-sigma 2] [--fmax KCAL] [--bands bands.json] [--out DIR]"""
import argparse
import json

from .report import report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--funnel"); ap.add_argument("--learned-cv")
    ap.add_argument("--burn", type=float, default=0.2); ap.add_argument("--t-rej", type=float, default=0.2)
    ap.add_argument("--step-ns", type=float, default=1.0)
    ap.add_argument("--min-points", type=int, default=12)
    ap.add_argument("--min-count", type=float, default=0); ap.add_argument("--min-bin-ess", type=float, default=0.0)
    ap.add_argument("--min-density", type=float, default=0.05); ap.add_argument("--bins", type=int)
    ap.add_argument("--bins-per-hill", type=float, default=4); ap.add_argument("--bins-fallback", type=int, default=60)
    ap.add_argument("--hill-mask-sigmas", type=float, default=2.0)
    ap.add_argument("--smooth-sigma", type=float, default=2.0); ap.add_argument("--fmax", type=float)
    ap.add_argument("--bands"); ap.add_argument("--out"); ap.add_argument("--prefix", default="metad")
    ap.add_argument("--show-numbers", action="store_true", help="draw the ΔG° numbers as text on the FE figure")
    a = ap.parse_args(argv)
    bands = None
    if a.bands:
        with open(a.bands) as fh:
            bands = json.load(fh)
    s = report(a.run_dir, funnel=a.funnel, learned_cv=a.learned_cv, bands=bands, burn=a.burn, t_rej=a.t_rej,
               step_ns=a.step_ns, out_dir=a.out, prefix=a.prefix,
               min_points=a.min_points, min_count=a.min_count, min_bin_ess=a.min_bin_ess,
               hill_mask_sigmas=a.hill_mask_sigmas, smooth_sigma=a.smooth_sigma, fmax=a.fmax,
               bins=a.bins, bins_per_hill=a.bins_per_hill, bins_fallback=a.bins_fallback, min_density=a.min_density,
               show_numbers=a.show_numbers)
    print(f"layout: {s['layout']}")
    for k, v in s["files"].items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
