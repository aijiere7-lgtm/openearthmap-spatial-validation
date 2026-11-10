"""runlock.py — pid-based locks that keep one `run_id` from being trained twice.

WHY THIS EXISTS
---------------
`experiments/experiments_registry.csv` is append-only and is the study's record
of what was executed. Two processes training the same `run_id` would append two
rows with the same identity, which the statistics gate treats as an error, and
would put two 6-9 GB training working sets on one 15.82 GB host and one 8 GB GPU
-- the memory failure this project has already paid for twice (D-043).

The grid is now started by a scheduled task that repeats every 15 minutes so the
plan restarts itself after a reboot or a session close (D-051/D-052). That makes
concurrent launches a normal event rather than an exceptional one, so the guard
has to be real:

* a lock is created with `O_CREAT | O_EXCL`, which is atomic on Windows;
* a lock whose holder is gone (or is not a Python process) is reclaimed, because
  refusing to work on a stale lock would leave the grid stopped after a reboot --
  the exact state the lock exists to prevent;
* `run_experiments.py` takes a lock per `run_id` immediately before training, so
  the guard sits at the dangerous action rather than only at the launcher.

What prompted the per-run lock: on 2026-09-20 at 19:33:31 and 19:33:33 two
processes with an identical launcher command line appeared two seconds apart,
which looked like the repeating task having started two grids. It had not. The
interpreter in this environment is a two-stage shim
(`envs/default/Scripts/python.exe` re-executes `versions/3.13.12/python.exe`
with the same command line), so every Python command here is a parent/child
pair with identical `CommandLine` text; the
"second launcher" was the first launcher's own real interpreter, and only one
training process ever existed. The run was stopped by hand before it wrote a
row, so nothing was lost or corrupted.

The lock is kept because the cause of the *appearance* is now known rather than
removed: the repeating task does make concurrent launches a normal event, and a
guard at the moment of training is cheap next to re-running a grid whose
registry has two rows for one `run_id`.
"""
from __future__ import annotations

import ctypes
import json
import os
import time
from pathlib import Path

_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def process_image(pid: int) -> str | None:
    """Full image path of `pid`, or None if it is not running / not readable."""
    if not pid:
        return None
    h = ctypes.windll.kernel32.OpenProcess(
        _PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not h:
        return None
    try:
        buf = ctypes.create_unicode_buffer(32768)
        size = ctypes.c_ulong(32768)
        if ctypes.windll.kernel32.QueryFullProcessImageNameW(
                h, 0, buf, ctypes.byref(size)):
            return buf.value
        return None
    finally:
        ctypes.windll.kernel32.CloseHandle(h)


def is_python(pid: int) -> bool:
    """True if `pid` is alive and its image is a Python interpreter.

    The image name is checked, not just liveness: Windows recycles pids, and a
    lock left by a dead launcher whose pid now belongs to, say, `explorer.exe`
    must be treated as stale rather than honoured.
    """
    exe = process_image(pid)
    return bool(exe) and Path(exe).name.lower().startswith("python")


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def remove_quietly(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def acquire(path: Path, **payload) -> tuple[bool, int]:
    """Take the lock at `path` for this process.

    Returns `(acquired, holder_pid)`. `holder_pid` is 0 when the lock was taken,
    and otherwise names the live process that holds it. A lock whose recorded pid
    is dead, or is alive but is not a Python process, is reclaimed.
    """
    body = json.dumps({"pid": os.getpid(), "started_at_utc": stamp(),
                       **payload}).encode("utf-8")
    for attempt in (1, 2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            pid = int(read_json(path).get("pid") or 0)
            if pid == os.getpid():
                return True, 0
            if is_python(pid):
                return False, pid
            remove_quietly(path)          # stale: holder is gone or not Python
            if attempt == 2:
                return False, pid
            continue
        with os.fdopen(fd, "wb") as fh:
            fh.write(body)
        return True, 0
    return False, 0


def release(path: Path) -> None:
    """Drop the lock only if this process owns it."""
    if int(read_json(path).get("pid") or 0) == os.getpid():
        remove_quietly(path)


def holder(path: Path) -> int:
    """Pid of the live Python process holding `path`, or 0 if it is free.

    A lock whose holder is gone counts as free even though the file is still
    there: treating it as held would stop the grid after a crash, which is the
    opposite of what the lock is for.
    """
    pid = int(read_json(path).get("pid") or 0)
    return pid if (pid and is_python(pid)) else 0
