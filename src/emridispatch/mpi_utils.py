"""Optional MPI support for multi-rank runs (`mpirun -n N emridispatch ...`).

mpi4py is only imported when the environment says this process is one of
several MPI ranks. Importing mpi4py.MPI calls MPI_Init, which aborts a plain
process inside a Slurm job step when Open MPI finds no PMI server; serial
runs (eryn, impulse, single-rank gpry) therefore never touch MPI and behave
exactly as without mpi4py installed.
"""

import logging
import os

logger = logging.getLogger(__name__)

# Backends whose run() shares one problem across ranks (rank 0 collects).
MPI_BACKENDS = ("gpry",)

_COMM = None
_RESOLVED = False


def _launcher_size():
    """World size announced by the MPI launcher's environment, else 1.

    Open MPI's mpirun sets OMPI_COMM_WORLD_SIZE, MPICH/Intel MPI (and
    srun --mpi=pmi2) set PMI_SIZE; srun --mpi=pmix sets PMIX_RANK, with the
    step size in SLURM_STEP_NUM_TASKS.
    """
    env = os.environ
    for var in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE"):
        if var in env:
            return int(env[var])
    if "PMIX_RANK" in env and "SLURM_STEP_NUM_TASKS" in env:
        return int(env["SLURM_STEP_NUM_TASKS"])
    return 1


def comm():
    """MPI.COMM_WORLD for a multi-rank launch, None otherwise."""
    global _COMM, _RESOLVED
    if not _RESOLVED:
        _RESOLVED = True
        if _launcher_size() > 1:
            try:
                from mpi4py import MPI
            except ImportError:
                logger.warning(
                    "launched as %d MPI ranks but mpi4py is not installed; "
                    "every rank runs independently", _launcher_size())
            else:
                _COMM = MPI.COMM_WORLD
    return _COMM


def size():
    c = comm()
    return 1 if c is None else c.Get_size()


def rank():
    c = comm()
    return 0 if c is None else c.Get_rank()


def is_main():
    return rank() == 0


def run_on_main(fn):
    """fn() on rank 0, its return value broadcast to every rank.

    An exception on rank 0 is re-raised there and as RuntimeError on the
    other ranks, so they fail instead of waiting at the broadcast forever.
    """
    c = comm()
    if c is None or c.Get_size() == 1:
        return fn()
    if c.Get_rank() == 0:
        try:
            result = fn()
        except BaseException as err:
            c.bcast((False, f"{type(err).__name__}: {err}"), root=0)
            raise
        try:
            c.bcast((True, result), root=0)
        except Exception as err:    # unpicklable result: nothing was sent
            c.bcast((False, f"broadcast failed: {err}"), root=0)
            raise
        return result
    ok, payload = c.bcast(None, root=0)
    if not ok:
        raise RuntimeError(f"rank 0 failed: {payload}")
    return payload
