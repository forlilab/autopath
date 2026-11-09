# Membrane orientation

`addMembrane` builds the bilayer flat in the xy plane. It never moves the protein. So the
protein has to be upright before it arrives, and nothing used to check that it was.

A tilted system still builds and still runs to the end. It gives a wrong answer with no
error, which can waste GPU-days per ligand before anyone notices.

`orient_membrane` fixes that. It works out one rotation, applies it to the receptor and to
every ligand file, and returns the new paths. Using one rotation for everything is what
keeps docked poses sitting in the pocket.

## Using it

In `ap_equilibration.py` or `ap_prep.py`, just above the `SystemPreparation(...)` block:

```python
membrane_orientation = True
if membrane_orientation:
    receptor_path, ligands = orient_membrane(receptor_path, ligands, sys_name)
```

Both scripts ship with it set to `False`. Turn it on when you set `is_membrane=True`.

`n_term="in"` or `"out"` says which side the N-terminus sits on. Geometry alone cannot
work that out, so PPM needs telling.

## What it writes

Into the output directory:

- `<name>_oriented.pdb`, and an oriented copy of every ligand file.
- `membrane_orientation.json`, holding the rotation, the shift, the normal and the units.
- `ppm.log`, PPM's own output.

The oriented PDB also carries `REMARK 100` lines, so it still describes itself if it gets
separated from the directory. The old `CRYST1` line is removed, because a rotation makes it
wrong and `addMembrane` would otherwise size the box from it.

If the receptor is already upright, nothing is written and the input paths come back.

## PPM is the only method

[PPM 3.0](https://opm.phar.umich.edu/ppm_server) (`immers`) does the work. It is the
program the OPM database is built from, and it ships inside AmberTools' packmol-memgen.
Point `AUTOPATH_PPM` at the build you want to use.

PPM returns an oriented copy of the PDB you gave it. Its own answer for the normal is never
read. Instead the oriented copy is superposed back onto the input to recover the rotation,
which also proves the structure did not bend or get mirrored.

**There is no fallback.** If PPM is missing or refuses the structure, `orient_membrane`
raises an error rather than guessing. A wrong orientation that looks plausible costs
GPU-days. An error costs a minute.

### Why two other methods were removed

An earlier version tried PPM, then memembed, then an in-house hydrophobic-slab scan, and
used the scan to check whatever came back. Measured against OPM's published answers, from
the deposited RCSB structure, in degrees:

| structure | PPM | memembed | slab scan |
|---|---|---|---|
| 7NQK PepT2 (SLC15A2) | 0.7 | 1.7 | 4.6 |
| 1OKC ADP/ATP carrier | 3.5 | 7.7 | 3.4 |
| 1QJP OmpA, beta-barrel | 4.9 | - | 6.4 |
| 4PE5 GLUT1 | - | - | 2.5 |
| 1U19 bovine rhodopsin | 11.7 | 14.1 | 8.7 |

A dash means the method refused. memembed does not win a single row, so it earned nothing.

The slab scan was worse. On 8JZX chain A it was 87 degrees out and reported itself as
perfect, while PPM and memembed both got it right and agreed with each other to within 9
degrees. Because the scan was also the judge, it threw out both correct answers and kept
its own. It is gone.

### The PPM that ships with packmol-memgen is broken

It fails on every membrane run. `opm.f` has a FORMAT line with a missing comma, between
`f7.0` and the text after it, and gfortran stops there at runtime. That line sits just
after PPM works out the answer, so it does all the work and throws it away. packmol-memgen
never notices, because it runs `immers` through `os.system()` and ignores the exit code.

Add the comma, rebuild with the `Makefile` that ships with it, and point `AUTOPATH_PPM` at
your build.

## Benchmark

`benchmark_opm.py` fetches each structure twice: once from the RCSB as deposited, and once
from OPM where it is already upright. Superposing one onto the other gives the true answer
in the deposited frame, and the script reports how far PPM is from it. The reference comes
from outside, so this is not PPM marking its own work.

```bash
export AUTOPATH_PPM=/path/to/patched/immers
python benchmark_opm.py ./bench_data
```
