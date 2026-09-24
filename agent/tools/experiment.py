"""Author and execute bounded one-off research experiments."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
from collections.abc import Mapping
import time
from typing import Any

from ..core.contracts import ToolResult
from ..core.resources import REPO_ROOT, load_json_value, load_prompt


MAX_PROGRAM_CHARS = 50_000
WALL_TIMEOUT_SECONDS = 30
CPU_SECONDS = 20
MEMORY_BYTES = 2 * 1024**3
MAX_PROCESSES = 1
MAX_OPEN_FILES = 64
MAX_OUTPUT_BYTES = 64 * 1024
EXPERIMENT_SPEC_FIELDS = {
    "objective",
    "method",
    "inputs",
    "assumptions",
    "outputs",
}
EXPERIMENT_PROVENANCE_FIELDS = {
    "source_sha256",
    "manifest_version",
    "repo_revision",
    "actual_data_bounds",
    "validation_status",
}
VALIDATION_STATUS_FIELDS = {"authoring", "source", "result", "execution"}
VALIDATION_STATES = {"passed", "failed", "not_run", "unavailable"}
CAPABILITY_MANIFEST = load_json_value("experiment_capabilities.json")
EXPERIMENT_RESULT_SCHEMA = CAPABILITY_MANIFEST["result_schema"]
EXPERIMENT_RESULT_FIELDS = set(EXPERIMENT_RESULT_SCHEMA["required"])
AUTHORING_PROMPT = load_prompt("prompts/experiment_authoring.md")
AUTHORING_TOOL_NAME = "submit_research_program"
AUTHORING_TOOL_SCHEMA = {
    "type": "function",
    "name": AUTHORING_TOOL_NAME,
    "description": "Submit one authored research program for validation and isolation.",
    "parameters": {
        "type": "object",
        "properties": {"program": {"type": "string"}},
        "required": ["program"],
        "additionalProperties": False,
    },
    "strict": True,
}
_ALLOWED_IMPORTS = {
    (item["module"], item["name"])
    for item in CAPABILITY_MANIFEST["apis"]
}
_ALLOWED_LIBRARY_ALIASES = {
    item["module"]: item["alias"]
    for item in CAPABILITY_MANIFEST["safe_libraries"]
}
_DENIED_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "getattr",
    "globals",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}
_DENIED_ATTRIBUTES = {
    "chmod",
    "chown",
    "exec",
    "fork",
    "kill",
    "makedirs",
    "mkdir",
    "popen",
    "remove",
    "rename",
    "rmdir",
    "spawn",
    "system",
    "to_csv",
    "to_excel",
    "to_json",
    "to_parquet",
    "unlink",
    "write_bytes",
    "write_text",
}


def _non_empty_string(value: Any, field: str, *, maximum: int = 500) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    normalized = value.strip()
    if len(normalized) > maximum:
        raise ValueError(f"{field} exceeds {maximum} characters")
    return normalized


def _string_list(value: Any, field: str, *, allow_empty: bool) -> list[str]:
    if not isinstance(value, list) or (not allow_empty and not value):
        suffix = "" if allow_empty else " and cannot be empty"
        raise ValueError(f"{field} must be an array of strings{suffix}")
    return [_non_empty_string(item, f"{field} item") for item in value]


def _object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object, got {type(value).__name__}")
    return dict(value)


def _schema_type_name(types: str | list[str]) -> str:
    names = {
        "array": "an array",
        "boolean": "a boolean",
        "integer": "an integer",
        "null": "null",
        "number": "a number",
        "object": "an object",
        "string": "a string",
    }
    values = [types] if isinstance(types, str) else types
    return " or ".join(names[item] for item in values)


def _matches_schema_type(value: Any, expected: str) -> bool:
    return {
        "array": lambda: isinstance(value, list),
        "boolean": lambda: isinstance(value, bool),
        "integer": lambda: isinstance(value, int) and not isinstance(value, bool),
        "null": lambda: value is None,
        "number": lambda: isinstance(value, (int, float)) and not isinstance(value, bool),
        "object": lambda: isinstance(value, Mapping),
        "string": lambda: isinstance(value, str),
    }[expected]()


def _validate_schema(value: Any, schema: Mapping[str, Any], field: str) -> None:
    expected = schema.get("type")
    if expected is not None:
        types = [expected] if isinstance(expected, str) else expected
        if not any(_matches_schema_type(value, item) for item in types):
            raise ValueError(
                f"{field} must be {_schema_type_name(expected)}, "
                f"got {type(value).__name__}"
            )

    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{field} must use string keys")
        properties = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in value:
                raise ValueError(f"{field}.{name} is required")
        unexpected = set(value) - set(properties)
        additional = schema.get("additionalProperties", True)
        if unexpected and additional is False:
            raise ValueError(
                f"{field} contains unexpected field(s): {', '.join(sorted(unexpected))}"
            )
        for name, item in value.items():
            child_schema = properties.get(name)
            if child_schema is not None:
                _validate_schema(item, child_schema, f"{field}.{name}")
            elif isinstance(additional, Mapping):
                _validate_schema(item, additional, f"{field}.{name}")

    if isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value):
            _validate_schema(item, schema["items"], f"{field}[{index}]")

    if isinstance(value, str) and len(value) < schema.get("minLength", 0):
        raise ValueError(f"{field} must be a non-empty string")


def _validate_json_value(value: Any, field: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise ValueError(f"{field} must be finite or null")
        return
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{field} must use string keys")
        for key, item in value.items():
            _validate_json_value(item, f"{field}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{field}[{index}]")
        return
    raise ValueError(
        f"{field} must contain only JSON-compatible values, "
        f"got {type(value).__name__}"
    )


def validate_experiment_spec(value: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Validate the planner-facing experiment contract."""
    if not isinstance(value, Mapping) or set(value) != EXPERIMENT_SPEC_FIELDS:
        return None, f"experiment spec must contain exactly {sorted(EXPERIMENT_SPEC_FIELDS)}"
    try:
        spec = {
            "objective": _non_empty_string(value["objective"], "spec.objective"),
            "method": _non_empty_string(value["method"], "spec.method"),
            "inputs": _object(value["inputs"], "spec.inputs"),
            "assumptions": _string_list(
                value["assumptions"], "spec.assumptions", allow_empty=True
            ),
            "outputs": _string_list(value["outputs"], "spec.outputs", allow_empty=False),
        }
        json.dumps(spec["inputs"], allow_nan=False)
    except (TypeError, ValueError) as exc:
        return None, str(exc)
    return spec, None


def validate_experiment_result(value: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Validate the isolated executor result against the authoring manifest."""
    try:
        _validate_schema(value, EXPERIMENT_RESULT_SCHEMA, "result")
        for field in EXPERIMENT_RESULT_SCHEMA["required"]:
            _validate_json_value(value[field], f"result.{field}")
        result = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        return None, str(exc)
    return result, None


def validate_experiment_provenance(
    value: Any,
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate provenance retained even when execution is unavailable."""
    if not isinstance(value, Mapping) or set(value) != EXPERIMENT_PROVENANCE_FIELDS:
        return None, (
            "experiment provenance must contain exactly "
            f"{sorted(EXPERIMENT_PROVENANCE_FIELDS)}"
        )
    source_sha = value["source_sha256"]
    if source_sha is not None and (
        not isinstance(source_sha, str)
        or len(source_sha) != 64
        or any(char not in "0123456789abcdef" for char in source_sha)
    ):
        return None, "provenance.source_sha256 must be a lowercase SHA-256 or null"
    bounds = value["actual_data_bounds"]
    if bounds is not None and (
        not isinstance(bounds, Mapping) or set(bounds) != {"start", "end"}
    ):
        return None, "provenance.actual_data_bounds must contain start/end or be null"
    if bounds is not None and any(
        item is not None and not isinstance(item, str) for item in bounds.values()
    ):
        return None, "provenance actual data bounds must be strings or null"
    validation = value["validation_status"]
    if not isinstance(validation, Mapping) or set(validation) != VALIDATION_STATUS_FIELDS:
        return None, (
            "provenance.validation_status must contain exactly "
            f"{sorted(VALIDATION_STATUS_FIELDS)}"
        )
    if any(state not in VALIDATION_STATES for state in validation.values()):
        return None, "provenance.validation_status contains an invalid state"
    for field in ("manifest_version", "repo_revision"):
        if not isinstance(value[field], str) or not value[field]:
            return None, f"provenance.{field} must be a non-empty string"
    return dict(value), None


def _repo_revision() -> str:
    git_dir = REPO_ROOT / ".git"
    try:
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref: "):
            return head
        ref = head[5:]
        ref_path = git_dir / ref
        if ref_path.exists():
            return ref_path.read_text(encoding="utf-8").strip()
        for line in (git_dir / "packed-refs").read_text(encoding="utf-8").splitlines():
            if line and not line.startswith(("#", "^")):
                revision, name = line.split(" ", 1)
                if name == ref:
                    return revision
    except (OSError, ValueError):
        pass
    return "unknown"


def _validate_program_source(program: Any) -> str | None:
    if not isinstance(program, str) or not program.strip():
        return "authoring program must be a non-empty string"
    if len(program) > MAX_PROGRAM_CHARS:
        return f"authoring program exceeds {MAX_PROGRAM_CHARS} characters"
    try:
        tree = ast.parse(program, mode="exec")
    except SyntaxError as exc:
        return f"authoring program is not valid Python: {exc}"
    run_functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
    ]
    if len(run_functions) != 1 or isinstance(run_functions[0], ast.AsyncFunctionDef):
        return "authoring program must define exactly one synchronous run() function"
    args = run_functions[0].args
    if args.posonlyargs or args.args or args.kwonlyargs or args.vararg or args.kwarg:
        return "authoring run() must not accept arguments"
    disallowed_imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            disallowed_imports.extend(
                f"import {name.name}"
                + (f" as {name.asname}" if name.asname else "")
                for name in node.names
                if name.name not in _ALLOWED_LIBRARY_ALIASES
                or _ALLOWED_LIBRARY_ALIASES[name.name] != name.asname
            )
        if isinstance(node, ast.ImportFrom):
            disallowed_imports.extend(
                f"from {node.module or '<relative>'} import {name.name}"
                for name in node.names
                if node.level
                or node.module is None
                or (node.module, name.name) not in _ALLOWED_IMPORTS
            )
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            return "authoring program may not access dunder names"
        if isinstance(node, ast.Attribute) and (
            node.attr.startswith("__") or node.attr.casefold() in _DENIED_ATTRIBUTES
        ):
            return "authoring program contains a prohibited attribute"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and (
            node.func.id in _DENIED_CALLS
        ):
            return "authoring program contains a prohibited call"
    if disallowed_imports:
        return "authoring program contains disallowed imports: " + "; ".join(
            sorted(set(disallowed_imports))
        )
    return None


def _source_error_type(error: str) -> str:
    if "disallowed imports" in error:
        return "invalid_experiment_source"
    policy_markers = ("dunder", "prohibited")
    return (
        "experiment_policy_violation"
        if any(marker in error for marker in policy_markers)
        else "invalid_experiment_source"
    )


def _response_text(response: Any) -> str:
    text = response.get("output_text", "") if isinstance(response, Mapping) else getattr(
        response, "output_text", ""
    )
    return text if isinstance(text, str) else ""


def _authored_payload(response: Any) -> tuple[Any, str | None]:
    def field(value: Any, name: str, default: Any = None) -> Any:
        return value.get(name, default) if isinstance(value, Mapping) else getattr(
            value, name, default
        )

    calls = [
        item
        for item in field(response, "output", []) or []
        if field(item, "type") == "function_call"
    ]
    if calls:
        if len(calls) != 1 or field(calls[0], "name") != AUTHORING_TOOL_NAME:
            return None, "authoring response must contain one submit_research_program call"
        arguments = field(calls[0], "arguments", "")
        if isinstance(arguments, Mapping):
            return dict(arguments), None
        try:
            return json.loads(arguments), None
        except (TypeError, json.JSONDecodeError) as exc:
            return None, f"authoring function arguments are not valid JSON: {exc}"

    try:
        return json.loads(_response_text(response).strip()), None
    except json.JSONDecodeError as exc:
        return None, f"authoring response is not valid JSON: {exc}"


def build_authoring_payload(
    user_request: str,
    spec: Mapping[str, Any],
    *,
    model: str = "",
    repair_feedback: str | None = None,
) -> dict[str, Any]:
    request = {
        "user_request": user_request,
        "experiment_spec": dict(spec),
        "capability_manifest": CAPABILITY_MANIFEST,
    }
    if repair_feedback:
        request["repair_feedback"] = repair_feedback[:2000]
    return {
        "model": model,
        "instructions": AUTHORING_PROMPT,
        "input": json.dumps(
            request,
            ensure_ascii=False,
        ),
        "tools": [AUTHORING_TOOL_SCHEMA],
        "tool_choice": "required",
        "parallel_tool_calls": False,
    }


def author_experiment(
    user_request: str,
    spec: Mapping[str, Any],
    *,
    client: Any,
    model: str = "",
    repair_feedback: str | None = None,
) -> dict[str, Any]:
    normalized_spec, error = validate_experiment_spec(spec)
    if error:
        return {
            "status": "error",
            "program": None,
            "provenance": None,
            "error_type": "invalid_experiment_spec",
            "error": error,
        }
    try:
        response = client.create(build_authoring_payload(
            user_request,
            normalized_spec,
            model=model,
            repair_feedback=repair_feedback,
        ))
    except Exception as exc:
        return {
            "status": "error",
            "program": None,
            "provenance": None,
            "error_type": "provider_error",
            "error": f"{type(exc).__name__}: {exc}",
        }
    authored, error = _authored_payload(response)
    if not isinstance(authored, dict) or set(authored) != {"program"}:
        error = error or "authoring response must contain exactly program"
    source_error = None
    if error is None:
        source_error = _validate_program_source(authored["program"])
        error = source_error
    if error:
        return {
            "status": "error",
            "program": None,
            "provenance": None,
            "error_type": (
                _source_error_type(source_error) if source_error else "malformed_response"
            ),
            "error": error,
        }

    program = authored["program"].rstrip() + "\n"
    provenance = {
        "source_sha256": hashlib.sha256(program.encode("utf-8")).hexdigest(),
        "manifest_version": CAPABILITY_MANIFEST["version"],
        "repo_revision": _repo_revision(),
        "actual_data_bounds": None,
        "validation_status": {
            "authoring": "passed",
            "source": "passed",
            "result": "not_run",
            "execution": "unavailable",
        },
    }
    return {
        "status": "ok",
        "program": program,
        "provenance": provenance,
        "error_type": None,
        "error": "",
    }


def _isolation_config(root: Path, data_path: Path) -> dict[str, Any]:
    python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    venv_lib = Path(sys.prefix) / "lib"
    return {
        "root": str(root),
        "work_path": str(root.parent / "workspace"),
        "usr_lib": "/usr/lib",
        "venv_lib": str(venv_lib),
        "research_path": str(REPO_ROOT / "research"),
        "utils_path": str(REPO_ROOT / "utils"),
        "data_path": str(data_path),
        "data_entries": [
            path.name
            for path in data_path.iterdir()
            if not path.name.startswith(".env")
        ],
        "python_path": [
            "/app",
            f"/usr/lib/{python_version}",
            f"/usr/lib/{python_version}/lib-dynload",
            str(venv_lib / python_version / "site-packages"),
        ],
        "environment": {
            "DATA_PATH": "/data",
            "HOME": "/work",
            "TMPDIR": "/work",
            "PYTHONNOUSERSITE": "1",
            "MPLBACKEND": "Agg",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
        "limits": {
            "cpu_seconds": CPU_SECONDS,
            "memory_bytes": MEMORY_BYTES,
            "processes": MAX_PROCESSES,
            "open_files": MAX_OPEN_FILES,
            "output_bytes": MAX_OUTPUT_BYTES,
        },
    }


def _prepare_root(root: Path, venv_lib: Path) -> None:
    for relative in ("app/research", "app/utils", "data", "proc", "usr/lib", "work"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    (root / venv_lib.as_posix().lstrip("/")).mkdir(parents=True, exist_ok=True)
    (root / "lib").symlink_to("usr/lib")
    (root.parent / "workspace").mkdir()


def _execute_isolated_source(
    program: str,
    *,
    data_path: str | os.PathLike[str] | None = None,
    wall_timeout: float = WALL_TIMEOUT_SECONDS,
    memory_bytes: int = MEMORY_BYTES,
    cpu_seconds: int = CPU_SECONDS,
) -> dict[str, Any]:
    """Run source across the native Linux isolation boundary."""
    unshare = shutil.which("unshare")
    worker = REPO_ROOT / "agent/tools/experiment_worker.py"
    raw_data_path = data_path or os.environ.get("DATA_PATH")
    selected_data = Path(raw_data_path).expanduser() if raw_data_path else None
    if (
        not unshare
        or not worker.is_file()
        or selected_data is None
        or not selected_data.is_dir()
    ):
        return {
            "status": "error",
            "error_type": "experiment_unavailable",
            "message": "unshare isolation or canonical DATA_PATH is unavailable",
        }
    if any(path.parent != selected_data for path in selected_data.rglob(".env*")):
        return {
            "status": "error",
            "error_type": "experiment_unavailable",
            "message": "nested dotenv files cannot be safely mounted",
        }

    with tempfile.TemporaryDirectory(prefix="tradar-experiment-") as directory:
        run_dir = Path(directory)
        root = run_dir / "root"
        venv_lib = Path(sys.prefix) / "lib"
        _prepare_root(root, venv_lib)
        (run_dir / "workspace/experiment.py").write_text(program, encoding="utf-8")
        config = _isolation_config(root, selected_data.resolve())
        config["limits"]["memory_bytes"] = memory_bytes
        config["limits"]["cpu_seconds"] = cpu_seconds
        config_path = run_dir / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        stdout_path = run_dir / "stdout"
        stderr_path = run_dir / "stderr"
        command = [
            unshare,
            "--user",
            "--map-root-user",
            "--mount",
            "--net",
            "--pid",
            "--ipc",
            "--uts",
            "--fork",
            "--kill-child=KILL",
            "--propagation",
            "private",
            sys.executable,
            str(worker),
            str(config_path),
        ]
        try:
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    env={"PYTHONNOUSERSITE": "1"},
                    shell=False,
                    start_new_session=True,
                )
                try:
                    return_code = process.wait(timeout=wall_timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    return {
                        "status": "error",
                        "error_type": "experiment_timeout",
                        "message": f"experiment exceeded {wall_timeout:g}s wall-time limit",
                    }
        except OSError as exc:
            return {
                "status": "error",
                "error_type": "experiment_unavailable",
                "message": f"could not start isolated executor: {type(exc).__name__}: {exc}",
            }

        stdout = stdout_path.read_bytes()
        stderr = stderr_path.read_bytes()
        if len(stdout) > MAX_OUTPUT_BYTES or len(stderr) > MAX_OUTPUT_BYTES:
            return {
                "status": "error",
                "error_type": "experiment_resource_limit",
                "message": "experiment exceeded output limit",
            }
        try:
            payload = json.loads(stdout.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            if return_code < 0 or return_code in {137, 152}:
                error_type = "experiment_resource_limit"
                message = "experiment was terminated by a resource limit"
            else:
                error_type = "experiment_sandbox_violation"
                message = "isolated worker returned invalid output"
            return {"status": "error", "error_type": error_type, "message": message}
        if not isinstance(payload, dict) or payload.get("status") not in {"success", "error"}:
            return {
                "status": "error",
                "error_type": "experiment_sandbox_violation",
                "message": "isolated worker returned an invalid envelope",
            }
        if return_code and payload.get("status") == "success":
            return {
                "status": "error",
                "error_type": "experiment_sandbox_violation",
                "message": "isolated worker exit status did not match its envelope",
            }
        return payload


def run_research_experiment(
    spec: Mapping[str, Any],
    *,
    authored_program: str | None = None,
    authoring_provenance: Mapping[str, Any] | None = None,
    run_id: str | None = None,
    **unsupported: Any,
) -> ToolResult:
    """Execute one authored program inside the native Linux sandbox."""
    started = time.perf_counter()
    normalized_spec, spec_error = validate_experiment_spec(spec)
    normalized_args = {"spec": normalized_spec if normalized_spec is not None else spec}
    if unsupported:
        spec_error = f"unsupported argument(s): {', '.join(sorted(unsupported))}"
    if spec_error:
        return ToolResult.error(
            "run_research_experiment",
            normalized_args,
            "invalid_experiment_spec",
            spec_error,
            run_id=run_id,
            provenance={"module": "agent.tools.experiment"},
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    provenance = dict(authoring_provenance or {
        "source_sha256": None,
        "manifest_version": CAPABILITY_MANIFEST["version"],
        "repo_revision": _repo_revision(),
        "actual_data_bounds": None,
        "validation_status": {
            "authoring": "not_run",
            "source": "not_run",
            "result": "not_run",
            "execution": "unavailable",
        },
    })
    provenance_error = (
        _validate_program_source(authored_program)
        if authored_program is not None
        else "authored program is required"
    )
    if authored_program is not None and provenance.get("source_sha256") != hashlib.sha256(
        authored_program.encode("utf-8")
    ).hexdigest():
        provenance_error = "authored program does not match provenance.source_sha256"
    _, contract_error = validate_experiment_provenance(provenance)
    provenance_error = provenance_error or contract_error
    if provenance_error:
        return ToolResult.error(
            "run_research_experiment",
            normalized_args,
            "invalid_experiment_provenance",
            provenance_error,
            run_id=run_id,
            provenance={"module": "agent.tools.experiment"},
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    execution = _execute_isolated_source(authored_program)
    if execution["status"] == "error":
        provenance["validation_status"] = {
            **provenance["validation_status"],
            "result": "not_run",
            "execution": (
                "unavailable"
                if execution["error_type"] == "experiment_unavailable"
                else "failed"
            ),
        }
        return ToolResult.error(
            "run_research_experiment",
            normalized_args,
            execution["error_type"],
            execution.get("message", "experiment execution failed"),
            run_id=run_id,
            provenance={
                "module": "agent.tools.experiment",
                "executor": "unshare-chroot-v1",
                **provenance,
            },
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    result, result_error = validate_experiment_result(execution.get("result"))
    if result_error:
        provenance["validation_status"] = {
            **provenance["validation_status"],
            "result": "failed",
            "execution": "passed",
        }
        return ToolResult.error(
            "run_research_experiment",
            normalized_args,
            "experiment_invalid_result",
            result_error,
            run_id=run_id,
            provenance={
                "module": "agent.tools.experiment",
                "executor": "unshare-chroot-v1",
                **provenance,
            },
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    provenance["actual_data_bounds"] = {
        "start": result["data_coverage"].get("actual_start"),
        "end": result["data_coverage"].get("actual_end"),
    }
    provenance["validation_status"] = {
        "authoring": "passed",
        "source": "passed",
        "result": "passed",
        "execution": "passed",
    }
    return ToolResult(
        tool_name="run_research_experiment",
        **({"run_id": run_id} if run_id is not None else {}),
        status="success",
        normalized_args=normalized_args,
        result=result,
        warnings=[
            {"code": "experiment_warning", "message": warning}
            for warning in result["warnings"]
        ],
        provenance={
            "module": "agent.tools.experiment",
            "executor": "unshare-chroot-v1",
            **provenance,
        },
        timing={"elapsed_ms": round((time.perf_counter() - started) * 1000, 3)},
    )


__all__ = [
    "AUTHORING_PROMPT",
    "AUTHORING_TOOL_SCHEMA",
    "CAPABILITY_MANIFEST",
    "author_experiment",
    "build_authoring_payload",
    "_execute_isolated_source",
    "run_research_experiment",
    "validate_experiment_provenance",
    "validate_experiment_result",
    "validate_experiment_spec",
]
