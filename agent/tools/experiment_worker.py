"""Namespace/chroot worker for one authored experiment; invoked by fixed argv only."""

from __future__ import annotations

import contextlib
import ctypes
from datetime import date, datetime
import errno
import io
import json
import math
import os
from pathlib import Path
import resource
import runpy
import sys
from collections.abc import Mapping
from typing import Any


MS_RDONLY = 1
MS_NOSUID = 2
MS_NODEV = 4
MS_NOEXEC = 8
MS_REMOUNT = 32
MS_BIND = 4096
MS_REC = 16384
PR_SET_DUMPABLE = 4
PR_SET_NO_NEW_PRIVS = 38
LINUX_CAPABILITY_VERSION_3 = 0x20080522


class _CapHeader(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]


class _CapData(ctypes.Structure):
    _fields_ = [
        ("effective", ctypes.c_uint32),
        ("permitted", ctypes.c_uint32),
        ("inheritable", ctypes.c_uint32),
    ]


class _LimitedWriter(io.StringIO):
    def __init__(self, limit: int) -> None:
        super().__init__()
        self.limit = limit

    def write(self, value: str) -> int:
        if self.tell() + len(value.encode("utf-8")) > self.limit:
            raise RuntimeError("experiment stdout exceeded output limit")
        return super().write(value)


def _emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False))


def _normalize_mapping(items, field: str) -> dict[str, Any]:
    normalized = {}
    for key, item in items:
        key = _normalize_json(key, f"{field} key")
        if not isinstance(key, (str, bool, int, float)):
            raise TypeError(f"{field} has a mapping key that is not a JSON-compatible scalar")
        key = str(key)
        if key in normalized:
            raise TypeError(f"{field} has duplicate keys after JSON normalization: {key!r}")
        normalized[key] = _normalize_json(item, f"{field}.{key}")
    return normalized


def _normalize_json(value: Any, field: str = "result") -> Any:
    """Normalize common research values without repairing envelope structure."""
    module = type(value).__module__.split(".", 1)[0]
    name = type(value).__name__
    if module == "pandas" and name in {"NAType", "NaTType"}:
        return None
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if module == "numpy":
        converter = getattr(value, "tolist", None) or getattr(value, "item", None)
        if callable(converter):
            return _normalize_json(converter(), field)
    if module == "pandas":
        if name == "Timedelta":
            return str(value)
        if name == "Series":
            return _normalize_mapping(value.items(), field)
        if name == "DataFrame":
            return _normalize_json(value.to_dict(orient="records"), field)
        converter = getattr(value, "tolist", None) or getattr(value, "item", None)
        if callable(converter):
            return _normalize_json(converter(), field)
    if isinstance(value, Mapping):
        return _normalize_mapping(value.items(), field)
    if isinstance(value, (list, tuple)):
        return [
            _normalize_json(item, f"{field}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{field} contains unsupported JSON value type {type(value).__name__}"
    )


def _mount(
    source: str | None,
    target: Path,
    filesystem: str | None = None,
    flags: int = 0,
    data: str | None = None,
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.mount(
        source.encode() if source is not None else None,
        str(target).encode(),
        filesystem.encode() if filesystem is not None else None,
        ctypes.c_ulong(flags),
        data.encode() if data is not None else None,
    )
    if result != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(target))


def _bind_read_only(source: str, target: Path, *, noexec: bool = False) -> None:
    if Path(source).is_dir():
        target.mkdir(parents=True, exist_ok=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    _mount(source, target, flags=MS_BIND | MS_REC)
    flags = MS_BIND | MS_REMOUNT | MS_RDONLY | MS_NOSUID | MS_NODEV
    if noexec:
        flags |= MS_NOEXEC
    _mount(None, target, flags=flags)


def _mount_workspace(source: str, target: Path, size: int) -> None:
    program = (Path(source) / "experiment.py").read_bytes()
    target.mkdir(parents=True, exist_ok=True)
    _mount(
        "tmpfs",
        target,
        filesystem="tmpfs",
        flags=MS_NOSUID | MS_NODEV | MS_NOEXEC,
        data=f"size={size},nr_inodes=4096,mode=700",
    )
    (target / "experiment.py").write_bytes(program)
    (target / "cache/duckdb").mkdir(parents=True, mode=0o700)


def _drop_capabilities() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        os.setgroups([])
    except PermissionError:
        pass
    header = _CapHeader(LINUX_CAPABILITY_VERSION_3, 0)
    data = (_CapData * 2)()
    if libc.capset(ctypes.byref(header), ctypes.byref(data)) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    libc.prctl(PR_SET_DUMPABLE, 0, 0, 0, 0)


def _apply_limits(limits: dict[str, int]) -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (limits["cpu_seconds"], limits["cpu_seconds"]))
    resource.setrlimit(resource.RLIMIT_AS, (limits["memory_bytes"], limits["memory_bytes"]))
    resource.setrlimit(resource.RLIMIT_NPROC, (limits["processes"], limits["processes"]))
    resource.setrlimit(resource.RLIMIT_NOFILE, (limits["open_files"], limits["open_files"]))
    resource.setrlimit(resource.RLIMIT_FSIZE, (limits["workspace_bytes"], limits["workspace_bytes"]))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _setup(config: dict[str, Any]) -> None:
    root = Path(config["root"])
    _mount(str(root), root, flags=MS_BIND | MS_REC)
    _bind_read_only(config["usr_lib"], root / "usr/lib")
    for path in config["python_library_paths"]:
        _bind_read_only(path, root / path.lstrip("/"))
    _bind_read_only(config["research_path"], root / "app/research", noexec=True)
    _bind_read_only(config["utils_path"], root / "app/utils", noexec=True)
    for name in config["data_entries"]:
        _bind_read_only(
            str(Path(config["data_path"]) / name), root / "data" / name, noexec=True
        )
    _mount_workspace(config["work_path"], root / "work", config["limits"]["workspace_bytes"])
    _mount(
        "proc",
        root / "proc",
        filesystem="proc",
        flags=MS_RDONLY | MS_NOSUID | MS_NODEV | MS_NOEXEC,
    )
    _mount(
        None,
        root,
        flags=MS_BIND | MS_REMOUNT | MS_RDONLY | MS_NOSUID | MS_NODEV,
    )

    os.chroot(root)
    os.chdir("/work")
    os.environ.clear()
    os.environ.update(config["environment"])
    os.umask(0o077)
    _apply_limits(config["limits"])
    _drop_capabilities()
    sys.path[:] = config["python_path"]


def _configure_duckdb(settings: dict[str, Any]) -> None:
    """Apply worker-only settings before any canonical loader imports connect."""
    import duckdb

    original_connect = duckdb.connect

    def connect(database=":memory:", read_only=False, config=None):
        return original_connect(
            database, read_only=read_only, config={**(config or {}), **settings}
        )

    duckdb.connect = connect


def _error_type(exc: BaseException) -> str:
    if isinstance(exc, ImportError):
        return "experiment_import_error"
    if isinstance(exc, MemoryError):
        return "experiment_resource_limit"
    if type(exc).__module__ in {"duckdb", "_duckdb"} and type(exc).__name__ == "OutOfMemoryException":
        return "experiment_resource_limit"
    if isinstance(exc, RuntimeError) and "output limit" in str(exc):
        return "experiment_resource_limit"
    if isinstance(exc, (PermissionError, BlockingIOError)):
        return "experiment_sandbox_violation"
    if isinstance(exc, OSError) and exc.errno in {
        errno.EACCES,
        errno.EPERM,
        errno.EAGAIN,
        errno.EMFILE,
        errno.ENFILE,
        errno.ENETDOWN,
        errno.ENETUNREACH,
        errno.EHOSTUNREACH,
    }:
        return "experiment_sandbox_violation"
    if isinstance(exc, OSError) and exc.errno in {errno.ENOSPC, errno.EFBIG}:
        return "experiment_resource_limit"
    if isinstance(exc, (TypeError, ValueError)) and "JSON" in str(exc):
        return "experiment_invalid_result"
    return "experiment_runtime_error"


def main() -> int:
    try:
        config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
        _setup(config)
    except Exception as exc:
        _emit({
            "status": "error",
            "error_type": "experiment_unavailable",
            "message": f"isolation setup failed: {type(exc).__name__}: {exc}",
        })
        return 70

    try:
        with contextlib.redirect_stdout(_LimitedWriter(config["limits"]["output_bytes"])), contextlib.redirect_stderr(
            _LimitedWriter(config["limits"]["output_bytes"])
        ):
            _configure_duckdb(config["duckdb"])
            namespace = runpy.run_path("/work/experiment.py", run_name="__experiment__")
            run = namespace.get("run")
            if not callable(run):
                raise TypeError("run() was not found")
            result = _normalize_json(run())
        serialized = json.dumps(result, ensure_ascii=False, allow_nan=False)
        if len(serialized.encode("utf-8")) > config["limits"]["output_bytes"]:
            raise RuntimeError("experiment result exceeded output limit")
        _emit({"status": "success", "result": result})
        return 0
    except BaseException as exc:
        _emit({
            "status": "error",
            "error_type": _error_type(exc),
            "message": f"{type(exc).__name__}: {exc}"[:2000],
        })
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
