"""python -m autopath.metadynamics.summary RUN_DIR [--funnel F] [--learned-cv L] [--burn 0.2] [--t-rej 0.2] [--step-ns 1] [--min-points 12] [--min-count 5] [--min-bin-ess 3] [--hill-mask-sigmas 2] [--smooth-sigma 2] [--fmax KCAL] [--bands bands.json] [--out DIR]"""
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
    ap.add_argument("--min-count", type=float, default=5); ap.add_argument("--min-bin-ess", type=float, default=3.0)
    ap.add_argument("--hill-mask-sigmas", type=float, default=2.0)
    ap.add_argument("--smooth-sigma", type=float, default=2.0); ap.add_argument("--fmax", type=float)
    ap.add_argument("--bands"); ap.add_argument("--out"); ap.add_argument("--prefix", default="metad")
    a = ap.parse_args(argv)
    bands = None
    if a.bands:
        with open(a.bands) as fh:
            bands = json.load(fh)
    s = report(a.run_dir, funnel=a.funnel, learned_cv=a.learned_cv, bands=bands, burn=a.burn, t_rej=a.t_rej,
               step_ns=a.step_ns, out_dir=a.out, prefix=a.prefix,
               min_points=a.min_points, min_count=a.min_count, min_bin_ess=a.min_bin_ess,
               hill_mask_sigmas=a.hill_mask_sigmas, smooth_sigma=a.smooth_sigma, fmax=a.fmax)
    print(f"layout: {s['layout']}")
    for k, v in s["files"].items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
