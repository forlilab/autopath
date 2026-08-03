import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid

import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array
import tqdm

import logging
logger = logging.getLogger("autopath.pulling.SMDData")
from collections import defaultdict

class SMDData:
    """Container and preprocessor for steered MD log files.

    Wraps one or more ``.dat`` log files produced by a dcTMD/SMD run.
    On construction the logs are loaded, filtered, and augmented with
    cumulative work and analysis-coordinate columns; the result is stored
    in ``self.raw_data``.  Estimator results are accumulated later via
    :meth:`add_estimator_results` and stored in ``self.results``.

    Coordinate conventions
    ----------------------
    ``r_target`` is the theoretical protocol grid — the ideal constraint
    position at each step, used by dcTMD.  Because it follows a fixed
    schedule it is identical across replicas and suitable for grouping.

    ``r_before`` is the actual ligand distance measured *before* the
    constraint force is applied.  It reflects stochastic fluctuations, so
    different replicas land on different grids; it requires binning before
    it can be used as an analysis coordinate.

    Parameters (selected)
    ---------------------
    completion_threshold_nm : float
        Distance threshold (nm) used to drop trajectories that did not
        reach their end state (default 0.1 nm).
    """

    def __init__(self,
                log_files: list[str],
                sysname: str = 'autopath',
                exclude_speeds: list[float] = None,
                r_column: str = 'r_target',
                temperature: float = 300.0,
                reference_pdb: str = None,
                work_mode: str = 'protocol',  # 'force_dx', 'protocol' or 'auto'
                protocol_work_column: str = 'dW_protocol',
                completion_threshold_nm: float = 0.1,
                ):
        
        self.sysname = sysname
        self.log_files = log_files
        self.traj_files = [self._log_to_traj_file(fn) for fn in log_files]
        self.reference_pdb = reference_pdb # for topology if needed and reference
        
        self.temperature = temperature
        self.R = 0.008314462618  # kJ/(mol*K)
        self.RT = self.R * self.temperature
        self.beta = 1.0 / self.RT
        
        self.pulling_direction = self.log_files[0].split("_")[-1].replace(".dat","")
        self.force_column = 'force'
        self.outdir = os.path.dirname(self.log_files[0])

        self.r_column = r_column
        if r_column not in ['r_target', 'r_before']:
            raise ValueError(f'r_column must be "r_target" or "r_before", got: {r_column!r}')
        self.completion_threshold_nm = completion_threshold_nm

        self.work_mode = work_mode
        self.protocol_work_column = protocol_work_column

        raw_data = self.load_logs()

        # Logged per-ligand spring constant (median over replicas that recorded a
        # header); None when no log carried one (older data -> derived downstream).
        _ks = [m['spring_constant_kJ_mol_nm2'] for m in self.log_metadata.values()
               if isinstance(m.get('spring_constant_kJ_mol_nm2'), (int, float))]
        self.spring_constant = float(np.median(_ks)) if _ks else None
        # nominal (filename) speed -> realized speed, for v->0 / dF/dv regressions.
        self.realized_speed_map = (
            raw_data.groupby('speed')['realized_speed'].median().to_dict()
            if raw_data is not None else {})

        # Build protocol grids before any filtering so all speeds are captured.
        self.protocol_grids = self.build_protocol_grids(raw_data)

        if exclude_speeds is not None:
            logger.warning(f'The following speeds will be excluded from analysis: {exclude_speeds}')
            raw_data = raw_data[~raw_data['speed'].isin(exclude_speeds)]
            
        to_drop = []
        for traj_name, traj_data in raw_data.groupby('trajname'):
            if self.pulling_direction == 'forward':
                if traj_data["r_after"].min() < self.completion_threshold_nm:
                    logger.warning(f'Dropping {traj_name}, min distance {traj_data["r_after"].min():.2f} nm')
                    to_drop.append(traj_name)
            else:  # backward pulling
                if traj_data["r_after"].min() > self.completion_threshold_nm:
                    logger.warning(f'Dropping {traj_name}, min distance {traj_data["r_after"].min():.2f} nm')
                    to_drop.append(traj_name)
        
        raw_data = raw_data[~raw_data['trajname'].isin(to_drop)]

        raw_data = self.integrate_force_dx(
            raw_data,
            r_column,
            work_mode=self.work_mode,
            protocol_work_column=self.protocol_work_column,
        )
        
        raw_data['lag'] = raw_data['r_after'] - raw_data['r_target']
        
        # r_coord is a median-based analysis coordinate shared across replicas;
        # it must not be used for binning or indexing — only for plotting/analysis.
        raw_data = self.build_analysis_coord(raw_data)

        raw_data['path'] = 1  # default single path

        self.raw_data = raw_data
        self.results = None


    def load_logs(self) -> pd.DataFrame:
        """Read all log files and return a concatenated DataFrame.

        Each file is read with ``pd.read_csv`` (comment lines starting
        with ``#`` are skipped).  The columns ``trajname``, ``speed``,
        and ``repid`` are appended from the filename.  Files that cannot
        be parsed are skipped with an error log.

        After this call ``self.raw_data`` does not yet exist — the
        returned DataFrame is passed through further processing steps in
        ``__init__`` before being assigned to ``self.raw_data``.
        """
        count = 0
        raw_data = []
        self.log_metadata = {}
        for fn in self.log_files:
            try:
                df = pd.read_csv(fn, comment='#')
                trajname = os.path.basename(fn)[:-4]  # remove .dat extension
                df['trajname'] = trajname
                nominal_speed = self._speed_from_log(fn)
                df['speed'] = nominal_speed
                df['repid'] = self._replica_idx_from_log(fn)
                meta = self.parse_log_metadata(fn)
                self.log_metadata[trajname] = meta
                # realized pulling speed (steps_per_move rounding). Falls back to the
                # nominal filename speed for logs written before k/speed logging.
                df['realized_speed'] = float(
                    meta.get('realized_speed_nm_per_ps', nominal_speed))
                raw_data.append(df)
                count += 1
            except Exception as e:
                logger.error(f"Error loading {fn}: {e}")
                continue
        if count == 0:
            logger.warning("No valid log files found.")
            return None
        logger.info(f"Loaded {count} log files.")

        if not raw_data:
            logger.warning("No data loaded from log files.")
            return None
        
        return pd.concat(list(raw_data))

    def build_protocol_grids(self, raw_data: pd.DataFrame) -> dict[float, pd.DataFrame]:
        """
        Reconstruct the ideal protocol grid per speed analytically.

        Returns
        -------
        protocol_grids : dict
            protocol_grids[speed] = DataFrame with columns:
            ['step', 'r_target_protocol']
        """
        protocol_grids = {}

        for speed, g in raw_data.groupby("speed"):
            step0 = g["step"].min()
            stepN = g["step"].max()
            # Median across replicas, not the first row: r_target at step0 is
            # not perfectly reproducible across replicas (see
            # build_analysis_coord below), and .iloc[0] would make the grid
            # depend on the arbitrary order log files were concatenated in.
            r0 = g.loc[g["step"] == step0, "r_target"].median()

            # Median step-to-step increment across replicas (robust to outliers).
            dx_vals = (
                g.groupby("trajname")["r_target"]
                .diff()
                .dropna()
                .values
            )
            dx = np.median(dx_vals)

            steps = np.arange(step0, stepN + 1)
            r_target_protocol = r0 + (steps - step0) * dx

            protocol_grids[speed] = pd.DataFrame({
                "step": steps,
                "r_target_protocol": r_target_protocol,
            })

        return protocol_grids
    
    def build_analysis_coord(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Attach r_coord: the per-(speed, step) median of r_target.

        r_target should in theory match across replicas at a given speed, but
        tiny numerical differences (mostly from r0 not being exact) mean it
        doesn't; the median gives one shared coordinate for analysis/plotting.
        """
        df = raw_data.copy()

        df["r_coord"] = (
            df.groupby(["speed", "step"])["r_target"]
            .transform("median")
        )

        return df
    
    def integrate_force_dx(self,
                           raw_data: pd.DataFrame,
                           r_column: str,
                           work_mode: str = 'auto',
                           protocol_work_column: str = 'dW_protocol') -> pd.DataFrame:
        """Build cumulative work per trajectory.

        Modes
        -----
        - 'force_dx': legacy definition using trapezoidal integration of force over r_column.
        - 'protocol': cumulative sum of per-move protocol increments (default column dW_protocol).
        - 'auto': use 'protocol' when the column is present, else fallback to 'force_dx'.
        """

        valid_modes = {'auto', 'force_dx', 'protocol'}
        if work_mode not in valid_modes:
            raise ValueError(f"Unknown work_mode '{work_mode}'. Allowed: {sorted(valid_modes)}")

        resolved_mode = work_mode
        if work_mode == 'auto':
            resolved_mode = 'protocol' if protocol_work_column in raw_data.columns else 'force_dx'

        grouped = raw_data.groupby('trajname')
        for traj, group in grouped:
            group = group.sort_values(by='time')

            if resolved_mode == 'protocol':
                if protocol_work_column not in group.columns:
                    raise ValueError(
                        f"work_mode='protocol' but column '{protocol_work_column}' was not found in logs."
                    )
                dW = pd.to_numeric(group[protocol_work_column], errors='coerce').fillna(0.0).to_numpy(dtype=float)
                work = np.cumsum(dW)
            else:
                work = cumulative_trapezoid(
                    group[self.force_column],
                    group[r_column],
                    initial=0.0,
                )

            raw_data.loc[raw_data['trajname'] == traj, 'work'] = work

        if resolved_mode == 'protocol':
            logger.info(
                f"Computed cumulative work from protocol increments column '{protocol_work_column}'."
            )
        else:
            logger.info(
                f"Computed cumulative work by integrating '{self.force_column}' over '{r_column}'."
            )

        return raw_data
    
    def filter_by_r_range(self, r_range, r_column) -> None:
        """Restrict ``self.raw_data`` to rows where ``r_column`` falls within ``r_range``.

        Modifies ``self.raw_data`` in-place and returns ``None``.

        Parameters
        ----------
        r_range : tuple[float, float]
            ``(r_min, r_max)`` bounds (inclusive) in the same units as ``r_column``.
        r_column : str
            Column name to filter on (typically ``'r_target'`` or ``'r_before'``).
        """

        logger.warning(f'Filtering data by x-range: {r_range}')
        self.raw_data = self.raw_data[(self.raw_data[r_column] >= r_range[0]) &
                                        (self.raw_data[r_column] <= r_range[1])]


    @staticmethod
    def parse_log_metadata(fn) -> dict:
        """Parse leading ``# key=value`` comment lines from an sMD .dat file.

        These carry per-replica constants written by
        :class:`SteeredMD` (spring_constant_kJ_mol_nm2, requested/realized
        speed, steps_per_move, force_n_samples). Values are coerced to float
        where possible. Returns an empty dict for older logs without a header.
        """
        meta = {}
        try:
            with open(fn) as fh:
                for line in fh:
                    if not line.startswith('#'):
                        break  # header block ends at the first data/column line
                    body = line[1:].strip()
                    if '=' not in body:
                        continue
                    key, _, val = body.partition('=')
                    key, val = key.strip(), val.strip()
                    try:
                        meta[key] = float(val)
                    except ValueError:
                        meta[key] = val
        except OSError:
            pass
        return meta

    @staticmethod
    def _speed_from_log(fn):
        # Example log name: sMD_replica-182557_v0.005_forward.dat
        base = os.path.basename(fn)
        for token in base.split("_"):
            if token.startswith("v"):
                return float(token[1:])
        return None
    
    @staticmethod
    def _replica_idx_from_log(fn):
        """Extract the integer replica ID from a log filename.

        Expected format: ``sMD_replica-{ID}_v{speed}_{direction}.dat``
        (e.g. ``sMD_replica-182557_v0.005_forward.dat``).  Index ``-3``
        after splitting on ``_`` yields the ``replica-{ID}`` token.
        """
        base = os.path.basename(fn)[:-4]
        rep = base.split("_")[-3]
        return int(rep.split("-")[1])

    @staticmethod
    def _replica_start_datetime(fn):
        """Chronological sort key for a replica log: its start ``datetime``.

        Why the filesystem is needed
        ----------------------------
        The replica ID in the filename is a bare ``HHMMSS`` clock time with
        **no date**, and the log header records only physical parameters --
        neither carries a calendar date.  So ``_replica_idx_from_log`` is a
        valid *identifier* but not a valid *ordering*: for any cell whose
        replicas were produced over more than one day it sorts by time of
        day, so "the first k replicas" becomes "the k with the earliest
        wall-clock time of day" rather than "the k run first".

        The missing date is recovered from the file's mtime, which marks
        when the replica finished writing.  Combining that date with the
        ``HHMMSS`` start time from the filename reconstructs the start
        instant (rolling back one day if the run crossed midnight).  This
        mirrors ``scratch/paper_figures/wdr5_conv_lib.py:_wall_seconds``.

        Caveat
        ------
        This relies on mtimes being preserved.  Copy sMD data with ``cp -a``
        / ``rsync -a``; a plain ``cp`` restamps every log to the copy time
        and collapses the ordering back to time-of-day within that instant.

        Falls back to the plain integer ID (as a ``datetime`` offset from the
        epoch, so the key type stays comparable) if the mtime cannot be read.
        """
        idx = SMDData._replica_idx_from_log(fn)
        try:
            end = datetime.fromtimestamp(os.path.getmtime(fn))
        except OSError:
            return datetime.fromtimestamp(0) + timedelta(seconds=idx)
        hh, mm, ss = idx // 10000, (idx // 100) % 100, idx % 100
        start = end.replace(hour=hh, minute=mm, second=ss, microsecond=0)
        if start > end:  # the run crossed midnight
            start -= timedelta(days=1)
        return start

    @staticmethod
    def _traj_to_log_name(traj_path: str) -> str:
        """
        Map a trajectory file path to the corresponding trajname used in raw_data.
        """
        base = os.path.basename(traj_path)
        root, ext = os.path.splitext(base)
        log_name = root.replace('traj', 'log')
        log_name = log_name.replace('_aligned', '')
        return log_name
    
    @staticmethod
    def _log_to_traj_file(log_path: str) -> str:
        """
        Map a log file path to the corresponding trajectory file name.
        """
        traj_name = log_path.replace('log', 'traj')
        traj_name = traj_name.replace('.dat', '_aligned.dcd')
        return traj_name
        
    def get_trace_features(self, 
                           features: list[str]=[ 'work', 'lag', 'r_before'],
                           stride: int=1, 
                           rescale_by_speed: bool = False,
                           zscore_by_speed: bool = False) -> pd.DataFrame:
        """
        Build a feature table from trace-like quantities (force, lag, work, ...),
        organized in r-space (x_col) for each trajectory. Optional rescaling/normalization
        by speed (in case of clustering multi-speed data).
        """

        df = self.raw_data.copy()
        
        # make sure trajs are ordered by step
        df = df.sort_values(['trajname', 'step']).reset_index(drop=True)

        # rescale trace features by speed
        if rescale_by_speed:
            for col in df.columns:
                if col in features:
                    df[col] = df[col] / df['speed']

        # z-score features within each speed 
        if zscore_by_speed:
            def _zscore_speed(group):
                for col in df.columns:
                    if col in features:
                        mean = group[col].mean()
                        std = group[col].std(ddof=0)
                        if std == 0 or np.isnan(std):
                            group[col] = 0.0
                        else:
                            group[col] = (group[col] - mean) / std
                return group

            df = df.groupby('speed', group_keys=False).apply(_zscore_speed)

        # apply a stride to the data to reduce size
        df = df.groupby(['speed', 'trajname']).apply(lambda g: g.iloc[::stride]).reset_index(drop=True)
        
        # round up values
        for col in df.columns:
            if col in features:
                df[col] = df[col].round(3)
                
        # keep only relevant columns
        keep_cols = ['trajname', 'speed', 'step', 'time'] + features
        df = df[keep_cols]
        
        return df
    
    @staticmethod
    def _speed_tag_from_trajname(traj_name: str) -> str:
        """Speed token embedded in a trajectory name, e.g. 'v0.015'.

        Trajectory names follow ``sMD_replica-<id>_v<speed>_<direction>``, so the
        speed tag is the second-to-last underscore field (mirrors the parsing used
        when building distance rows).
        """
        return traj_name.split("_")[-2]

    def _pocket_distances_for_trajs(self, trajs, group_A, group_B, stride):
        """Compute pocket→ligand distance rows for a list of trajectory files.

        Both selections are periodic-image "no-jump" corrected before distances
        are computed. sMD DCDs are written wrapped into the primary cell, so when
        the ligand crosses a box face during unbinding its whole COM is teleported
        by one lattice vector; without correction every pocket–ligand distance
        would jump discontinuously (a pure PBC artifact, not real geometry). We
        undo it by tracking each selection's COM and, whenever it moves more than
        half the smallest box edge between consecutive frames, subtracting that
        *observed* step from all later frames. Subtracting the observed step
        (rather than a box-frame lattice vector, or relying on
        ``distance_array(box=...)`` / MDAnalysis ``NoJump``) keeps it correct even
        when the DCD is RMSD-aligned (rotated) with an unrotated triclinic box,
        where box-based min-image is inconsistent. Both groups are corrected
        because callers pass group_A/group_B in either order (e.g. run() passes
        group_A=ligand, group_B=pocket); the static group's correction stays ~0,
        and since the two groups start co-located (bound) staying continuous from
        frame 0 keeps them in a common periodic image. No-wrap trajs are unchanged.
        """
        def _nojump(u, atoms):
            """Per-frame cumulative shift undoing periodic-image teleports of `atoms`
            (a COM step > half the min box edge is a wrap, not real motion here)."""
            corr = np.zeros(3); prev = None; out = {}
            for ts in u.trajectory:
                box = ts.dimensions[:3]
                half = 0.5 * float(np.min(box)) if np.all(box > 0) else np.inf
                com = atoms.center_of_mass()
                if prev is None:
                    out[ts.frame] = corr.copy(); prev = com + corr; continue
                cur = com + corr
                if np.linalg.norm(cur - prev) > half:
                    corr = corr - (cur - prev); cur = com + corr
                out[ts.frame] = corr.copy(); prev = cur
            return out

        all_rows = []
        for traj in tqdm.tqdm(trajs, desc="Calculating distances.."):
            u = mda.Universe(self.reference_pdb, traj)

            atoms_A = u.select_atoms(group_A)
            atoms_B = u.select_atoms(group_B)
            if atoms_A.n_atoms == 0 or atoms_B.n_atoms == 0:
                print(f"Warning: No atoms found for selection in trajectory {traj}. Skipping.")
                continue

            traj_name = self._traj_to_log_name(traj)
            speed = self._speed_tag_from_trajname(traj_name).strip("v")

            corrA = _nojump(u, atoms_A)
            corrB = _nojump(u, atoms_B)

            for ts in u.trajectory[::stride]:
                A_pos = atoms_A.positions + corrA[ts.frame]
                B_pos = atoms_B.positions + corrB[ts.frame]
                distances = distance_array(A_pos, B_pos)
                dist_flat = distances.flatten() / 10.0  # nm
                time_ps = getattr(ts, "time", ts.frame)  # ts.time in ps

                row = {
                    "trajname": traj_name,
                    "speed": float(speed),
                    "step": ts.frame,   # index within this trajectory
                    "time": time_ps,
                }
                for i, v in enumerate(dist_flat):
                    row[f"dist_{i}"] = v
                all_rows.append(row)

        return pd.DataFrame(all_rows)

    def calculate_pocket_distances(self,
                          group_A: str = None,
                          group_B: str = None,
                          recompute: bool = False,
                          stride: int = 2,
                          ) -> pd.DataFrame:
        """Compute (or load) pocket-ligand distance features.

        Persisted **per speed** to ``{outdir}/{sysname}_pocketDistances_v{speed}.csv``
        — check_convergence calls this once per speed with per-speed SMDData
        instances sharing the same outdir, so a single combined file was
        previously clobbered by whichever speed ran last. Results for all
        requested speeds are concatenated on return.

        Parameters
        ----------
        group_A : str
            MDAnalysis selection string for the **pocket** atoms.
        group_B : str
            MDAnalysis selection string for the **ligand** atoms.
        recompute : bool, optional
            If False (default), reuse each speed's cache file when present and
            only compute missing speeds. If True, recompute and overwrite all.
        stride : int, optional
            Frame stride for trajectory reading (default 2).

        Returns
        -------
        pd.DataFrame
            Columns ``['trajname', 'speed', 'step', 'time'] + dist_*``. ``step``
            is the frame index; ``time`` falls back to ``step`` if unavailable.
        """
        trajs_by_speed = defaultdict(list)
        for traj in self.traj_files:
            speed_tag = self._speed_tag_from_trajname(self._traj_to_log_name(traj))
            trajs_by_speed[speed_tag].append(traj)

        # Legacy single-file cache (pre per-speed scheme); reused per speed if present.
        legacy_file = f"{self.outdir}/{self.sysname}_pocketDistances.csv"
        legacy_df = None
        if (not recompute) and os.path.exists(legacy_file):
            try:
                legacy_df = pd.read_csv(legacy_file)
            except (OSError, pd.errors.ParserError):
                legacy_df = None

        frames = []
        for speed_tag, trajs in trajs_by_speed.items():
            per_speed_file = f"{self.outdir}/{self.sysname}_pocketDistances_{speed_tag}.csv"

            if (not recompute) and os.path.exists(per_speed_file):
                frames.append(pd.read_csv(per_speed_file))
                continue

            # Migrate this speed's rows out of the legacy combined file if available.
            if legacy_df is not None and "speed" in legacy_df.columns:
                speed_val = float(speed_tag.strip("v"))
                sub = legacy_df[legacy_df["speed"] == speed_val]
                if not sub.empty:
                    sub.to_csv(per_speed_file, index=False)  # promote to per-speed cache
                    frames.append(sub)
                    continue

            df_speed = self._pocket_distances_for_trajs(trajs, group_A, group_B, stride)
            if not df_speed.empty:
                df_speed.to_csv(per_speed_file, index=False)
            frames.append(df_speed)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def merge_feature_sets(*feature_dfs: pd.DataFrame,
                           tolerance_ps: float | None = None) -> pd.DataFrame:
        """Merge N feature DataFrames via a nearest-time asof merge per trajectory.

        The first DataFrame is the reference (left side); the rest are merged
        onto it one by one, e.g. combining trace features, pocket distances,
        and ligand shape descriptors into one feature table for clustering.
        All DataFrames must have ``trajname`` and ``time`` columns; shared
        metadata columns are taken from the reference to avoid duplicates.

        ``tolerance_ps`` caps the allowed asof time gap (ps); ``None`` always
        keeps the nearest match regardless of gap size.
        """
        if len(feature_dfs) < 2:
            raise ValueError("merge_feature_sets requires at least two DataFrames.")

        for idx, df in enumerate(feature_dfs):
            if "trajname" not in df.columns:
                raise ValueError(f"DataFrame {idx} is missing 'trajname' column")
            if "time" not in df.columns:
                raise ValueError(f"DataFrame {idx} is missing 'time' column")

        # Start with the reference (first df) and merge the rest onto it
        result = feature_dfs[0].copy()
        result["time"] = pd.to_numeric(result["time"], errors="coerce")
        result = result.dropna(subset=["time"])

        for right_df in feature_dfs[1:]:
            r = right_df.copy()
            r["time"] = pd.to_numeric(r["time"], errors="coerce")
            r = r.dropna(subset=["time"])

            common_traj = sorted(
                set(result["trajname"].unique()) & set(r["trajname"].unique())
            )

            # Drop shared metadata columns from the right side to avoid _x/_y suffixes.
            shared_cols = (set(result.columns) & set(r.columns)) - {"time"}
            r_keep = [c for c in r.columns if c not in shared_cols]

            merged_chunks = []
            for traj in common_traj:
                left_traj = result[result["trajname"] == traj].sort_values("time").reset_index(drop=True)
                right_traj = r.loc[r["trajname"] == traj, r_keep].sort_values("time").reset_index(drop=True)

                if len(left_traj) == 0 or len(right_traj) == 0:
                    continue

                kwargs = dict(
                    left=left_traj,
                    right=right_traj,
                    on="time",
                    direction="nearest",
                    allow_exact_matches=True,
                )
                if tolerance_ps is not None:
                    kwargs["tolerance"] = tolerance_ps

                merged_traj = pd.merge_asof(**kwargs)
                merged_traj["trajname"] = traj
                merged_chunks.append(merged_traj)

            if not merged_chunks:
                raise RuntimeError(
                    "No trajectories could be merged. "
                    "Check that trajname and time columns are consistent between DataFrames."
                )

            result = pd.concat(merged_chunks, ignore_index=True)

        return result

    @staticmethod
    def build_merged_features(trace_df: pd.DataFrame,
                              geom_df: pd.DataFrame,
                              tolerance_ps: float | None = None) -> pd.DataFrame:
        """Merge geometry and trace feature DataFrames.

        Convenience wrapper around :meth:`merge_feature_sets` that preserves
        the original two-argument signature.  The geometry DataFrame is used
        as the reference (left side) and trace features are merged onto it.
        """
        return SMDData.merge_feature_sets(geom_df, trace_df, tolerance_ps=tolerance_ps)

    @staticmethod
    def merge_geom_features(trace_df: pd.DataFrame,
                            geom_df: pd.DataFrame,
                            mode: str = "impute") -> pd.DataFrame:
        """Merge geom sidecar features onto trace features.

        mode='impute'  : merge_feature_sets nearest-fill (no NaN, full rows).
        mode='aligned' : keep only exact (trajname, time) matches.
        """
        if mode == "impute":
            return SMDData.merge_feature_sets(trace_df, geom_df)
        if mode == "aligned":
            geom_cols = [c for c in geom_df.columns if c.startswith("geom_")]
            right = geom_df[["trajname", "time"] + geom_cols]
            return trace_df.merge(right, on=["trajname", "time"], how="inner")
        raise ValueError(f"geom_merge mode must be 'impute' or 'aligned', got {mode!r}")

    def load_geom_features(self) -> pd.DataFrame:
        """Load per-run geom sidecars (``sMD_*_geom.dat``) for the loaded logs.

        Returns an empty DataFrame when no sidecars are present; never raises
        on missing files. Columns: trajname, speed, step, time, geom_*.
        """
        rows = []
        for log_fn in self.log_files:
            geom_fn = log_fn[:-4] + "_geom.dat"      # replace trailing '.dat'
            if not os.path.exists(geom_fn):
                continue
            try:
                g = pd.read_csv(geom_fn, comment='#')
            except Exception as e:
                logger.warning(f"Could not read geom sidecar {geom_fn}: {e}")
                continue
            g["trajname"] = os.path.basename(log_fn)[:-4]
            g["speed"] = self._speed_from_log(log_fn)
            rows.append(g)
        if not rows:
            logger.info("No geom sidecars (_geom.dat) found; geom features unavailable.")
            return pd.DataFrame()
        return pd.concat(rows, ignore_index=True)

    def add_estimator_results(self, estimator_name: str, results_df: pd.DataFrame):
        """Append per-(speed, path, step) estimator output to ``self.results``.

        Tags ``results_df`` with an ``'estimator'`` column set to
        ``estimator_name``, then concatenates into ``self.results``.
        Multiple calls accumulate results from different estimators.
        """
        if self.results is None:
            self.results = pd.DataFrame()
        results_df['estimator'] = estimator_name
        self.results = pd.concat([self.results, results_df], ignore_index=True)

    @staticmethod
    def _compute_p_neq(path_traj_counts: dict) -> dict:
        """Compute non-equilibrium path probabilities from trajectory counts.

        Parameters
        ----------
        path_traj_counts :
            Mapping of path label -> number of trajectories assigned to that path.

        Returns
        -------
        dict
            Mapping of path label -> p_neq (fraction of trajectories).  Empty
            dict when the total count is zero.
        """
        total_trajs = sum(path_traj_counts.values())
        if total_trajs <= 0:
            return {}
        return {
            path: count / total_trajs
            for path, count in path_traj_counts.items()
            if count > 0
        }

    def get_p_neq(self,
                  byspeed: bool = True
                  ) -> dict[str, float]:
        """p_neq Non-equilibrium path probabilities from sMD analysis.
        Returns a dictionary mapping path labels to p_neq values."""

        data = self.raw_data.copy()

        if byspeed:
            grouping_iter = data.groupby("speed")
        else:
            grouping_iter = [(None, data)]

        p_neq_dic = {}
        for speed, g in grouping_iter:
            logger.info(f"Speed {speed} nm/ps has {g['path'].nunique()} unique paths.")
            path_traj_counts = g.groupby('path')['trajname'].nunique().to_dict()
            speed_key = speed if byspeed else "all_speeds"
            p_neq_dic[speed_key] = SMDData._compute_p_neq(path_traj_counts)

        return p_neq_dic

    @staticmethod
    def _choose_estimator_for_weights(results: pd.DataFrame, estimator: str = 'auto') -> str:
        """Resolve which estimator's dG to use for p_eq computation.

        ``estimator='auto'`` selects by ROBUSTNESS: among genuine free-energy
        candidates (all available estimators EXCLUDING ``'force'``, whose
        ``dG=Wmean`` is non-negative by construction), pick the one with the
        smallest fraction of negative-dG bins — those are exactly the bins
        masked to 0 in ``_compute_p_eq`` and what destabilises the p_eq integral.
        Ties resolve to the order ``['cumulant','jarzynski']`` then first
        available.  An explicitly named estimator is returned as-is (raises if
        not present).
        """
        if 'estimator' not in results.columns:
            raise ValueError("results must include an 'estimator' column.")

        available_estimators = results['estimator'].dropna().unique().tolist()
        if len(available_estimators) == 0:
            raise ValueError("No estimator results available to compute p_eq.")

        if estimator != 'auto':
            if estimator not in available_estimators:
                raise ValueError(
                    f"Estimator '{estimator}' not available in results. "
                    f"Available: {sorted(available_estimators)}"
                )
            return estimator

        candidates = [e for e in available_estimators if e != 'force']
        if not candidates:
            raise ValueError("No non-force estimator available to weight paths.")

        def neg_frac(est: str) -> float:
            dG = results.loc[results['estimator'] == est, 'dG'].to_numpy(dtype=float)
            dG = dG[np.isfinite(dG)]
            return float((dG < 0).mean()) if dG.size else 1.0

        preference = {'cumulant': 0, 'jarzynski': 1}
        # sort by (neg fraction asc, preference asc, name) -> deterministic
        best = min(candidates, key=lambda e: (neg_frac(e),
                                              preference.get(e, 2),
                                              e))
        return best

    @staticmethod
    def choose_reference_estimator(results: pd.DataFrame) -> str:
        """Robustness-selected reference estimator for shared p_eq weighting."""
        return SMDData._choose_estimator_for_weights(results, 'auto')
