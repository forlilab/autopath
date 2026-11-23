# Troubleshooting

Each of these has broken a real run.

| Symptom | Cause | Fix |
|---|---|---|
| Log shows `Skipping harmonic restraints for ligand` although there is a ligand | The protocol selects `resname UNK` and the ligand has another name | Rewrite `components_lookup` in a copy of the protocol ([Equilibrate](equilibrate.md)) |
| `FileNotFoundError` on the protocol | `Equilibration`'s default `protocol_fname` is relative to the repository root | Pass an absolute path, e.g. from `importlib.resources.files("autopath") / "data"` |
| `NonbondedForce: The cutoff distance cannot be greater than half the periodic box size` / `box is X nm across its narrowest axis` | Box narrower than twice the cutoff (small solute) | Increase `padding`; ligand-only boxes need ≥ 1.5 nm |
| `No template found for residue ...` | A cofactor or heterogen kept in the receptor | Pass it as a ligand tuple with its SMILES |
| A job cannot load `checkpoint_equil_<sys>.chk` | Checkpoints are tied to the GPU model and platform | Start from `checkpoint_equil_<sys>.xml` (`state_file=` / `state_xml_file=`) |
| `Expected end of text, found '*'` | `wrap_align_save_traj(align_select=...)` uses mdtraj syntax | `resname LIG and not element H` |
| Periodic-image check rejects the trajectory / distances look wrong | Run on a superposed (`*_aligned`) trajectory | Run it on the raw DCD before `wrap_align_save_traj` |
| LIE or any pytraj analysis gives absurd energies or a box-mismatch error | pytraj misreads DCD unit cells | Convert to NetCDF with MDAnalysis first |
| cw-RMSF or RMSF ≈ 0 | Fewer than ~10 frames | Save more often or run longer |
| `system` is `None` after MD | `VanillaMD.run()` returns `None` | Keep the system from `load_system` |
| Pulls stop with `[PBC guard] Stopping at move ...` | The ligand approached a periodic image of the protein | Rebuild with more padding |
| Two pulls overwrite each other | Default pull ids are `HHMMSS`; parallel jobs at the same speed collided | Split jobs by speed, or pass unique integer `run_id`s |
| `ValueError: A ligand-only membrane system takes one ligand file: ligands='ligand.sdf'.` | `protein=None` with `is_membrane=True` and several ligands or a tuple | Pass a single ligand file; set `membrane_ligand_z` to position it |
