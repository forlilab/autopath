import logging

import pytest

from autopath import __version__
from autopath.utils import setup_logging, warning_count


@pytest.fixture(autouse=True)
def _restore_logger():
    yield
    lg = logging.getLogger("autopath")
    for h in list(lg.handlers):
        lg.removeHandler(h)
        h.close()
    lg.propagate = True
    lg.setLevel(logging.NOTSET)


def test_relative_logfile_without_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    setup_logging("autopath.log")
    logging.getLogger("autopath.x").info("hi")
    assert (tmp_path / "autopath.log").exists()


def test_second_call_switches_file(tmp_path):
    a, b = tmp_path / "a.log", tmp_path / "b.log"
    setup_logging(str(a))
    logging.getLogger("autopath.x").info("one")
    setup_logging(str(b))
    logging.getLogger("autopath.x").info("two")
    assert "two" in b.read_text()
    assert "two" not in a.read_text()
    assert a.read_text().count("one") == 1


def test_no_duplicate_handlers(tmp_path):
    f = str(tmp_path / "a.log")
    setup_logging(f)
    lg = setup_logging(f)
    assert len(lg.handlers) == 3


def test_warning_count(tmp_path):
    setup_logging(str(tmp_path / "a.log"))
    lg = logging.getLogger("autopath.x")
    lg.warning("w")
    lg.warning("w")
    lg.error("e")
    assert warning_count() == 3
    setup_logging(str(tmp_path / "b.log"))
    assert warning_count() == 0


def test_warning_count_without_setup():
    assert warning_count() == 0


def test_run_separator(tmp_path):
    f = tmp_path / "a.log"
    setup_logging(str(f))
    lines = [l for l in f.read_text().splitlines() if l.strip()]
    assert "=" * 60 in lines[0]
    assert "AutoPath" in lines[1] and __version__ in lines[1]


def test_level_applies_on_second_call(tmp_path):
    setup_logging(str(tmp_path / "a.log"), log_level="INFO")
    f = tmp_path / "b.log"
    setup_logging(str(f), log_level="DEBUG")
    logging.getLogger("autopath.x").debug("dbg")
    assert "dbg" in f.read_text()
