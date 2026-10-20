# Membrane permeation PMF with metadynamics

PMF of a small solute crossing a lipid bilayer, following Sun, Dama, Tan, Rose & Voth,
JCTC 12, 5157 (2016). Everything here is a thin driver over autopath:

| piece | autopath |
|---|---|
| signed z of the solute from the membrane centre (box-size free, wraps at ±Lz/2) | `metadynamics.CV.membrane_z_cv` |
| orientation of a solute bond to the normal (cos θ) | `metadynamics.CV.axis_cos_cv` |
| walker, WT / TT / WT×TT hills, shared or isolated bias, HILLS/COLVAR/REWEIGHT | `metadynamics.MetadynamicsMD.run` |
| PMF(z) per independent bias, symmetry error, profile features, transits | `metadynamics.permeation` |

## Input

An equilibrated membrane system: serialized `System` (barostat included) and a PDB with the box.
The system must contain a membrane-centre reference (default residue `DUM`, e.g. a fixed dummy
atom) and whatever keeps the solute in its column (e.g. a cylinder restraint). The solute is
residue `UNK` by default.

## Run

One arm = `RUNS` independent runs × `WALKERS` walkers, one GPU job per walker:

```bash
cd examples/08_Membrane-Permeation
CASE=/path/to/system
./launch_arm.sh runs/wt_shared 3 5 -p alphafold,forli,a100 -- \
    --system $CASE/system.xml --pdb $CASE/system.pdb --axis-atoms O1x C2x \
    --method wt --layout shared --ns 100
```

- `--method wt | tt | ttwt` (defaults γ 11 / TT γ 5 / WT 11 × TT 5, as in the paper)
- `--layout shared` (walkers of a run share one bias) or `isolated` (each walker its own)
- `--basins symmetric | paper | leaflet | bulk` or explicit points `"0,0 1.3,0"` (z,cos θ) for TT
- `--state state.xml` starts from an OpenMM State instead of the PDB positions
- `python run_permeation.py -h` lists the rest (hill height/pace/widths, grid, temperature)

Outputs in `runs/<arm>/r<k>/`: per walker `HILLS_w<j>.npz`, `COLVAR_w<j>.npy`, `REWEIGHT_w<j>.npz`
(with `--headgroup-atom P` also `monitor_bilayer_z`, the headgroup centroid z, to catch bilayer drift; ~25% slower), the DCD,
`walker_<j>.json` (every setting) and `w<j>.log`; the bias in `bias/` (shared) or `bias_w<j>/`.

The sbatch excludes the slow GPU nodes (pick fast cards with sbatch options, e.g. `-p forli --gres=gpu:rtxa6000:1`) and kills a walker running below `MIN_NS_DAY` (default 300)
after `GUARD_AFTER_S` (default 900 s); resubmit that array task.

## Analyse

```bash
python analyse_permeation.py WT=runs/wt_shared TT=runs/tt_shared --out analysis/
```

Works mid-run. Each independent bias is one PMF estimate (isolated: one per walker; shared: one
per run); an arm's PMF is the free-energy-space mean of its estimates ± SD (paper Eq 13), never
a pool of frames across biases. Time on the convergence plots is walker-ns summed over the arm,
so shared and isolated arms compare at equal cost.

| output | what to look at |
|---|---|
| `summary.csv` | centre barrier and well depth ± SD, symmetry error, full crossings, walkers that never crossed, bilayer drift |
| `pmf.png` | PMF ± SD; raw (asymmetry = unconverged) and symmetrised |
| `convergence.png` | symmetry error, barrier, well and estimate spread vs walker-ns: flat and small = converged |
| `fes-2d.png` | final 2D PMF (z, cos θ) per arm from the bias, summary-report palette; TT basins as stars |
| `z_traces.png` | does every walker cross, or sit on one side? |

There is no reference PMF: the symmetry F(z) = F(-z) is the known-answer check. It is necessary,
not sufficient; agreement between independent estimates is the second check.
