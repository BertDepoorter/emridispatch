"""Multi-rank (mpirun) behaviour: rank 0 computes the Fisher box and owns the
outdir files, other ranks receive the box by broadcast; serial runs never
import mpi4py.

A fake communicator stands in for MPI.COMM_WORLD, one rank at a time. The
real two-rank test is skipped without mpirun and mpi4py.
"""

import logging
import os
import pickle
import shutil
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from emridispatch import mpi_utils
from emridispatch.logging_utils import setup_logging
from emridispatch.pipeline import build_problem, run_from_config

LAUNCHER_VARS = ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_RANK",
                 "SLURM_STEP_NUM_TASKS")


class FakeComm:
    """One rank of a world of `size`; bcast pickles like mpi4py does.

    Rank 0 records what it broadcasts; other ranks receive `inbox`.
    """

    def __init__(self, rank, size=2, inbox=None):
        self.rank, self.size, self.inbox = rank, size, inbox
        self.sent = []

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.size

    def bcast(self, obj, root=0):
        if self.rank == root:
            payload = pickle.dumps(obj)
            self.sent.append(pickle.loads(payload))
            return obj
        return pickle.loads(pickle.dumps(self.inbox))


@pytest.fixture(autouse=True)
def _reset_mpi(monkeypatch):
    for var in LAUNCHER_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(mpi_utils, "_COMM", None)
    monkeypatch.setattr(mpi_utils, "_RESOLVED", False)
    yield
    logger = logging.getLogger("emridispatch")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.propagate = True
    logger.setLevel(logging.NOTSET)


def _use_comm(monkeypatch, comm):
    monkeypatch.setattr(mpi_utils, "_COMM", comm)
    monkeypatch.setattr(mpi_utils, "_RESOLVED", True)


# --- launcher detection ------------------------------------------------------

@pytest.mark.parametrize("env, expected", [
    ({}, 1),
    ({"OMPI_COMM_WORLD_SIZE": "4"}, 4),
    ({"PMI_SIZE": "3"}, 3),
    ({"PMIX_RANK": "0", "SLURM_STEP_NUM_TASKS": "2"}, 2),
    # A plain Slurm step (srun --pty bash, sbatch) is not an MPI launch.
    ({"SLURM_STEP_NUM_TASKS": "1", "SLURM_JOB_ID": "1"}, 1),
])
def test_launcher_size(monkeypatch, env, expected):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert mpi_utils._launcher_size() == expected


def test_serial_run_never_imports_mpi4py(monkeypatch, toy_cfg):
    # Importing mpi4py.MPI calls MPI_Init; serial (eryn) runs must not.
    monkeypatch.delitem(sys.modules, "mpi4py", raising=False)
    monkeypatch.delitem(sys.modules, "mpi4py.MPI", raising=False)
    problem = build_problem(toy_cfg)
    assert mpi_utils.comm() is None
    assert (mpi_utils.size(), mpi_utils.rank(), mpi_utils.is_main()) == (1, 0, True)
    assert "mpi4py" not in sys.modules
    assert os.path.exists(os.path.join(problem.outdir, "prior_bounds.npz"))


def test_multi_rank_launch_without_mpi4py_falls_back(monkeypatch, caplog):
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "2")
    monkeypatch.setitem(sys.modules, "mpi4py", None)    # ImportError
    with caplog.at_level("WARNING", logger="emridispatch.mpi_utils"):
        assert mpi_utils.comm() is None
    assert "mpi4py is not installed" in caplog.text


# --- run_on_main ---------------------------------------------------------------

def test_run_on_main_rank0_computes_and_broadcasts(monkeypatch):
    comm = FakeComm(rank=0)
    _use_comm(monkeypatch, comm)
    assert mpi_utils.run_on_main(lambda: 42) == 42
    assert comm.sent == [(True, 42)]


def test_run_on_main_other_rank_receives(monkeypatch):
    _use_comm(monkeypatch, FakeComm(rank=1, inbox=(True, 42)))

    def never():
        raise AssertionError("only rank 0 runs fn")

    assert mpi_utils.run_on_main(never) == 42


def test_run_on_main_error_reaches_every_rank(monkeypatch):
    comm = FakeComm(rank=0)
    _use_comm(monkeypatch, comm)

    def fail():
        raise np.linalg.LinAlgError("Fisher not positive definite")

    with pytest.raises(np.linalg.LinAlgError):
        mpi_utils.run_on_main(fail)
    ((ok, msg),) = comm.sent
    assert not ok and "Fisher not positive definite" in msg

    _use_comm(monkeypatch, FakeComm(rank=1, inbox=(False, msg)))
    with pytest.raises(RuntimeError, match="rank 0 failed: LinAlgError"):
        mpi_utils.run_on_main(fail)


# --- pipeline ------------------------------------------------------------------

def test_other_ranks_get_rank0_box_and_write_nothing(monkeypatch, toy_cfg,
                                                     tmp_path):
    rank0 = FakeComm(rank=0)
    _use_comm(monkeypatch, rank0)
    p0 = build_problem(toy_cfg, resume=False)
    ((ok, bounds),) = rank0.sent
    assert ok

    toy_cfg.run.outdir = str(tmp_path / "rank1")
    os.makedirs(toy_cfg.run.outdir)
    _use_comm(monkeypatch, FakeComm(rank=1, inbox=(True, bounds)))
    p1 = build_problem(toy_cfg, resume=False)

    assert os.listdir(toy_cfg.run.outdir) == []
    np.testing.assert_array_equal(p1.prior.mins, p0.prior.mins)
    np.testing.assert_array_equal(p1.prior.maxes, p0.prior.maxes)
    np.testing.assert_array_equal(p1.proposal_cov, p0.proposal_cov)
    np.testing.assert_allclose(p1.reparam.to_u(p1.x0), p0.reparam.to_u(p0.x0))


@pytest.mark.parametrize("backend", ["eryn", "impulse"])
def test_serial_backend_refuses_multiple_ranks(monkeypatch, toy_cfg, backend):
    _use_comm(monkeypatch, FakeComm(rank=0))
    toy_cfg.sampler.backend = backend
    try:
        with pytest.raises(ValueError, match="serial"):
            run_from_config(toy_cfg)
    except ImportError:
        pytest.skip(f"{backend} not installed")
    assert not os.path.exists(os.path.join(toy_cfg.run.outdir,
                                           "prior_bounds.npz"))


# --- logging ---------------------------------------------------------------------

def test_other_ranks_log_warnings_only_and_no_file(monkeypatch, tmp_path):
    _use_comm(monkeypatch, FakeComm(rank=3, size=4))
    logger = setup_logging(outdir=str(tmp_path), filename="run.log")
    (console,) = logger.handlers
    assert console.level == logging.WARNING
    assert "[rank 3]" in console.formatter._fmt
    assert not os.path.exists(tmp_path / "run.log")


def test_rank0_logging_unchanged(monkeypatch, tmp_path):
    _use_comm(monkeypatch, FakeComm(rank=0, size=4))
    logger = setup_logging(outdir=str(tmp_path), filename="run.log")
    assert [type(h) for h in logger.handlers] == [logging.StreamHandler,
                                                   logging.FileHandler]
    assert all(h.level == logging.NOTSET for h in logger.handlers)


# --- real MPI --------------------------------------------------------------------

def _mpirun():
    return shutil.which("mpirun") or shutil.which("mpiexec")


@pytest.mark.skipif(_mpirun() is None, reason="no mpirun on PATH")
def test_run_on_main_under_mpirun(tmp_path):
    pytest.importorskip("mpi4py")
    script = tmp_path / "bcast.py"
    script.write_text(textwrap.dedent("""
        from emridispatch import mpi_utils
        calls = []
        out = mpi_utils.run_on_main(lambda: calls.append(1) or {"box": [1, 2]})
        print(mpi_utils.rank(), mpi_utils.size(), len(calls), out["box"])
    """))
    res = subprocess.run([_mpirun(), "-n", "2", sys.executable, str(script)],
                         capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr
    lines = sorted(res.stdout.split("\n"))[-2:]
    assert lines == ["0 2 1 [1, 2]", "1 2 0 [1, 2]"]
