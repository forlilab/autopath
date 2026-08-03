"""Chronological ordering of sMD replica logs.

The replica ID baked into an sMD log filename is a bare ``HHMMSS`` clock
time with no date, and the log header carries only physical parameters.
Sorting by that integer therefore orders a multi-day cell by *time of day*,
so "the first k replicas" (the rung-k set used by ``check_convergence`` and
by the force ladder) is not "the first k replicas that were run".

``SMDData._replica_start_datetime`` recovers the missing date from the file
mtime (when the replica finished) and combines it with the filename's start
time.  These tests pin the three behaviours that matter: it fixes the
multi-day case, it is a no-op on the single-day case, and it handles a run
that crossed midnight.
"""
import os
from datetime import datetime, timedelta

import pytest

from autopath.pulling.SMDData import SMDData


def _write_log(dirpath, hhmmss, mtime, speed=0.015):
    """Create an sMD log named for `hhmmss` and stamped with `mtime`."""
    fn = os.path.join(dirpath, f"sMD_replica-{hhmmss}_v{speed}_forward.dat")
    with open(fn, "w") as fh:
        fh.write("step,time,r_target\n0,0.0,1.0\n")
    ts = mtime.timestamp()
    os.utime(fn, (ts, ts))
    return fn


def test_chronological_key_beats_filename_id_across_days(tmp_path):
    """The bug itself: filename order contradicts real chronology.

    Two campaigns on the same cell -- an early batch run in the afternoon of
    day 1 and a later batch run in the morning of day 2.  The later batch has
    the *smaller* HHMMSS, so the integer key puts it first.
    """
    d = str(tmp_path)
    day1 = datetime(2026, 7, 18, 14, 40, 50)
    day2 = datetime(2026, 8, 2, 11, 4, 50)

    # ran first (day 1, afternoon) but has the LARGER filename id
    first_run = _write_log(d, "144050", day1)
    # ran second (day 2, morning) but has the SMALLER filename id
    second_run = _write_log(d, "110450", day2)

    files = [second_run, first_run]

    by_time = sorted(files, key=SMDData._replica_start_datetime)
    assert by_time == [first_run, second_run]

    # ...and the old key demonstrably gets it backwards.
    by_id = sorted(files, key=SMDData._replica_idx_from_log)
    assert by_id == [second_run, first_run]
    assert by_id != by_time


def test_chronological_key_orders_a_whole_multi_day_batch(tmp_path):
    """Scaled-up version of the same contradiction: three campaigns.

    Mirrors the real 6dy7_A/sMD-murcko v=0.015 layout, where the 2026-07-18
    batch has the latest times of day and so sorted last under the old key.
    """
    d = str(tmp_path)
    batches = [
        (datetime(2026, 7, 18, 14, 15, 0), ["141500", "142000", "144000"]),
        (datetime(2026, 8, 1, 13, 38, 0), ["133800", "133900", "134000"]),
        (datetime(2026, 8, 2, 11, 4, 0), ["110400", "110500", "110600"]),
    ]
    expected = []
    files = []
    for day, times in batches:
        for hhmmss in times:
            mtime = day.replace(hour=int(hhmmss[:2]), minute=int(hhmmss[2:4]),
                                second=int(hhmmss[4:])) + timedelta(seconds=30)
            fn = _write_log(d, hhmmss, mtime)
            files.append(fn)
            expected.append(fn)

    shuffled = list(reversed(files))
    assert sorted(shuffled, key=SMDData._replica_start_datetime) == expected

    # The old key would have put the 2026-08-02 batch first and 07-18 last.
    by_id = sorted(shuffled, key=SMDData._replica_idx_from_log)
    assert by_id[:3] == expected[6:]
    assert by_id[-3:] == expected[:3]


def test_single_day_ordering_is_unchanged(tmp_path):
    """Backward compatibility: on a single-day cell the new key must agree
    with the old integer key exactly."""
    d = str(tmp_path)
    day = datetime(2026, 7, 18, 0, 0, 0)
    times = ["141500", "141534", "142011", "133000", "094500", "231259"]
    files = [
        _write_log(d, t,
                   day.replace(hour=int(t[:2]), minute=int(t[2:4]),
                               second=int(t[4:])) + timedelta(seconds=25))
        for t in times
    ]

    by_time = sorted(files, key=SMDData._replica_start_datetime)
    by_id = sorted(files, key=SMDData._replica_idx_from_log)
    assert by_time == by_id


def test_midnight_crossing_places_start_on_the_previous_day(tmp_path):
    """A replica started at 23:55 and finished at 00:10 the next day: its
    mtime date is day+1, but its start belongs to day."""
    d = str(tmp_path)
    end = datetime(2026, 7, 19, 0, 10, 0)
    fn = _write_log(d, "235500", end)

    start = SMDData._replica_start_datetime(fn)
    assert start == datetime(2026, 7, 18, 23, 55, 0)
    assert start < end

    # and it must sort before a replica that started at 00:05 on day+1
    later = _write_log(d, "000500", datetime(2026, 7, 19, 0, 20, 0))
    assert sorted([later, fn], key=SMDData._replica_start_datetime) == [fn, later]


def test_missing_file_falls_back_to_the_integer_id(tmp_path):
    """The key must never raise on a log that has disappeared."""
    missing = os.path.join(str(tmp_path), "sMD_replica-141500_v0.015_forward.dat")
    assert not os.path.exists(missing)

    key = SMDData._replica_start_datetime(missing)
    assert isinstance(key, datetime)

    other = os.path.join(str(tmp_path), "sMD_replica-141600_v0.015_forward.dat")
    assert SMDData._replica_start_datetime(other) > key


def test_key_is_usable_as_a_sort_key_type(tmp_path):
    """All keys must be mutually comparable (sorted() must not raise)."""
    d = str(tmp_path)
    files = [
        _write_log(d, "141500", datetime(2026, 7, 18, 14, 15, 30)),
        _write_log(d, "110450", datetime(2026, 8, 2, 11, 5, 15)),
    ]
    assert len(sorted(files, key=SMDData._replica_start_datetime)) == 2
