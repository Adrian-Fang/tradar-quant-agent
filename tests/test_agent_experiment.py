from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sysconfig
import tempfile
import unittest
from unittest.mock import patch

import duckdb
import numpy as np
import pandas as pd

import agent.tools.experiment as experiment_module
from agent.tools import experiment_worker
from agent.agent import run_agent
from agent.core.contracts import ToolResult
from agent.tools.calling import TOOL_SCHEMAS
from agent.tools.experiment import (
    CAPABILITY_MANIFEST,
    AUTHORING_TOOL_SCHEMA,
    _execute_isolated_source,
    author_experiment,
    run_research_experiment,
    validate_experiment_provenance,
    validate_experiment_result,
    validate_experiment_spec,
)


def response(value):
    return {"output_text": json.dumps(value, ensure_ascii=False)}


def spec():
    return {
        "objective": "Measure forward A-share returns after broad-market down days.",
        "method": "event study",
        "inputs": {
            "start_date": "2025-01-01",
            "end_date": "2026-09-23",
            "market_proxy": "000300",
            "horizons": [1, 3, 5, 10, 20],
        },
        "assumptions": ["Use 000300 as the delegated broad-market proxy."],
        "outputs": ["forward return summary", "event and observation counts"],
    }


def _sandbox_namespaces_available():
    try:
        return subprocess.run(
            [
                "unshare", "--user", "--map-root-user", "--mount", "--net",
                "--pid", "--ipc", "--uts", "--fork", "--kill-child=KILL",
                "--propagation", "private", "true",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        ).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


PROGRAM = (
    "def run():\n"
    "    return {'assumptions': ['fixture'], 'method': {'type': 'fixture'}, "
    "'data_coverage': {'actual_start': '2025-01-02', 'actual_end': '2025-01-03'}, "
    "'metrics': {'mean': 0.01}, 'sample_counts': {'rows': 2}, 'warnings': []}\n"
)

EVENT_PROGRAM = (
    "import pandas as pd\n"
    "import numpy as np\n\n"
    "def run():\n"
    "    return {\n"
    "        'assumptions': ['Use CSI 300 as the market proxy.'],\n"
    "        'method': {'type': 'event_study', 'horizons': np.array([1, 3, 5, 10, 20])},\n"
    "        'data_coverage': {\n"
    "            'requested_start': '2025-01-01',\n"
    "            'requested_end': '2026-09-23',\n"
    "            'actual_start': pd.Timestamp('2025-01-02'),\n"
    "            'actual_end': pd.Timestamp('2025-01-03'),\n"
    "        },\n"
    "        'metrics': {\n"
    "            'mean_forward_return': pd.Series([0.01, 0.02], index=[1, 3]),\n"
    "            'yearly_stability': {2025: {'mean': np.float64(0.01)}},\n"
    "            'summary': pd.DataFrame([{'horizon': 1, 'mean': np.float64(0.01)}]),\n"
    "        },\n"
    "        'sample_counts': {'event_days': np.int64(2)},\n"
    "        'warnings': pd.Index([]),\n"
    "    }\n"
)


def isolated_program(statements):
    return (
        f"{statements}\n\n"
        "def run():\n"
        "    return {'assumptions': [], 'method': {'type': 'sandbox_probe'}, "
        "'data_coverage': {'actual_start': None, 'actual_end': None}, "
        "'metrics': probe(), 'sample_counts': {'rows': 0}, 'warnings': []}\n"
    )


class Client:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def create(self, payload):
        self.calls.append(payload)
        if isinstance(self.output, dict) and self.output.get("status") == "ready":
            if len(self.calls) > 1:
                return {"output_text": "Research completed."}
            step = self.output["steps"][0]
            return {"output": [{"type": "function_call", "call_id": "experiment-fixture",
                                "name": step["name"], "arguments": json.dumps(step["arguments"])}]}
        return response(self.output)


class ExperimentArchitectureTests(unittest.TestCase):
    def setUp(self):
        artifact_temp = tempfile.TemporaryDirectory()
        self.addCleanup(artifact_temp.cleanup)
        self.artifact_dir = Path(artifact_temp.name) / "experiments"
        artifact_patch = patch.object(
            experiment_module, "EXPERIMENT_ARTIFACT_DIR", self.artifact_dir
        )
        artifact_patch.start()
        self.addCleanup(artifact_patch.stop)

    def test_isolation_python_paths_follow_running_interpreter(self):
        config = experiment_module._isolation_config(Path("/tmp/root"), Path("/tmp/data"))
        expected = {
            path
            for path in (
                sysconfig.get_path("stdlib"),
                sysconfig.get_path("platstdlib"),
                sysconfig.get_config_var("DESTSHARED"),
                sysconfig.get_path("purelib"),
                sysconfig.get_path("platlib"),
            )
            if path
        }

        self.assertTrue(expected.issubset(config["python_library_paths"]))
        self.assertTrue(expected.issubset(config["python_path"]))
        self.assertEqual(config["environment"]["TRADAR_EXPERIMENT_SANDBOX"], "1")

    def test_cpu_wall_budgets_are_consistent_in_authoring_and_worker(self):
        payload = experiment_module.build_authoring_payload("request", spec())
        budget = json.loads(payload["input"])["execution_budget"]
        config = experiment_module._isolation_config(Path("/tmp/root"), Path("/tmp/data"))
        self.assertEqual(budget["cpu_seconds"], 80)
        self.assertEqual(budget["wall_seconds"], 90)
        self.assertEqual(experiment_module.CPU_SECONDS, budget["cpu_seconds"])
        self.assertEqual(experiment_module.WALL_TIMEOUT_SECONDS, budget["wall_seconds"])
        self.assertLess(budget["cpu_seconds"], budget["wall_seconds"])
        self.assertEqual(config["limits"]["cpu_seconds"], budget["cpu_seconds"])
        with patch.object(experiment_worker.resource, "setrlimit") as setrlimit:
            experiment_worker._apply_limits(config["limits"])
        setrlimit.assert_any_call(experiment_worker.resource.RLIMIT_CPU, (80, 80))

    def test_cpu_limit_classification_handles_unshare_exit_status(self):
        self.assertTrue(experiment_module._resource_limit_termination(1, 1.0, 1))
        self.assertFalse(experiment_module._resource_limit_termination(1, 0.1, 1))
        self.assertFalse(
            experiment_module._resource_limit_termination(-signal.SIGSEGV, 1.0, 1)
        )
        self.assertFalse(experiment_module._resource_limit_termination(2, 0.1, 1))
        self.assertFalse(experiment_module._resource_limit_termination(1, 60, experiment_module.CPU_SECONDS))
        self.assertTrue(experiment_module._resource_limit_termination(1, 80, experiment_module.CPU_SECONDS))
        for sig in (signal.SIGKILL, signal.SIGXCPU, signal.SIGXFSZ):
            for return_code in (-int(sig), 128 + int(sig)):
                with self.subTest(return_code=return_code):
                    self.assertTrue(experiment_module._resource_limit_termination(
                        return_code, 0, experiment_module.CPU_SECONDS,
                    ))

    def test_duckdb_settings_are_worker_only_and_cannot_be_overridden_at_connect(self):
        config = experiment_module._isolation_config(Path("/tmp/root"), Path("/tmp/data"))
        settings = config["duckdb"]
        self.assertEqual(settings["threads"], 1)
        self.assertEqual(settings["temp_directory"], "/work/cache/duckdb")
        self.assertLess(experiment_module.DUCKDB_MEMORY_BYTES, config["limits"]["memory_bytes"])
        self.assertLess(experiment_module.DUCKDB_SPILL_BYTES, config["limits"]["workspace_bytes"])
        from utils import duckdb_manager
        original_connect = duckdb.connect
        with patch.object(duckdb, "connect") as native_connect:
            experiment_worker._configure_duckdb(settings)
            duckdb_manager.get_conn()
            call = native_connect.call_args
            self.assertTrue(call.kwargs["read_only"])
            self.assertEqual(call.kwargs["config"], settings)
            native_connect.return_value.execute.assert_called_once_with(
                "SET max_temp_directory_size = ?", [settings["max_temp_directory_size"]]
            )
            duckdb.connect(config={"threads": 8, "memory_limit": "8GB"})
            self.assertEqual(native_connect.call_args.kwargs["config"], settings)
        self.assertIs(duckdb.connect, original_connect)
        # Shared production connections still pass no experiment config at all.
        with patch.object(duckdb, "connect") as shared_connect:
            duckdb_manager.get_conn()
        shared_connect.assert_called_once_with(duckdb_manager.DB_PATH, read_only=True)

    def test_duckdb_connection_is_closed_if_spill_configuration_fails(self):
        settings = experiment_module._isolation_config(Path("/tmp/root"), Path("/tmp/data"))["duckdb"]
        with patch.object(duckdb, "connect") as native_connect:
            native_connect.return_value.execute.side_effect = RuntimeError("configuration failed")
            experiment_worker._configure_duckdb(settings)
            with self.assertRaisesRegex(RuntimeError, "configuration failed"):
                duckdb.connect()
            native_connect.return_value.close.assert_called_once_with()

    def test_duckdb_oom_is_a_controlled_resource_error(self):
        error = duckdb.OutOfMemoryException("fixture allocation failed")
        self.assertEqual(experiment_worker._error_type(error), "experiment_resource_limit")

    def test_resource_failure_remains_a_structured_tool_error(self):
        authored = author_experiment("request", spec(), client=Client({"program": PROGRAM}))
        with patch.object(experiment_module, "_execute_isolated_source", return_value={
            "status": "error", "error_type": "experiment_resource_limit",
            "message": "OutOfMemoryException: bounded DuckDB allocation failed",
        }):
            result = run_research_experiment(spec(), authored_program=authored["program"],
                                            authoring_provenance=authored["provenance"])
        self.assertEqual(result.status, "error")
        self.assertEqual(result.errors, [{"code": "experiment_resource_limit", "message": "OutOfMemoryException: bounded DuckDB allocation failed"}])
        self.assertEqual(result.provenance["validation_status"]["execution"], "failed")

    def test_schema_uses_structured_spec_without_source(self):
        schema = next(item for item in TOOL_SCHEMAS if item["name"] == "run_research_experiment")

        self.assertEqual(schema["parameters"]["required"], ["spec"])
        encoded = json.dumps(schema, ensure_ascii=False)
        self.assertNotIn('"program"', encoded)
        self.assertEqual(
            set(schema["parameters"]["properties"]["spec"]["required"]),
            {"objective", "method", "inputs", "assumptions", "outputs"},
        )

    def test_authoring_receives_request_spec_and_versioned_manifest(self):
        client = Client({"program": PROGRAM})

        authored = author_experiment("研究大盘下跌事件。", spec(), client=client)
        payload = json.loads(client.calls[0]["input"])

        self.assertEqual(authored["status"], "ok")
        self.assertEqual(payload["experiment_spec"], spec())
        self.assertEqual(payload["capability_manifest"], CAPABILITY_MANIFEST)
        self.assertEqual(payload["capability_manifest"]["version"], "1.5.3")
        self.assertEqual(payload["execution_budget"]["process_memory_bytes"], experiment_module.MEMORY_BYTES)
        self.assertEqual(payload["execution_budget"]["duckdb_threads"], 1)
        self.assertLess(payload["execution_budget"]["duckdb_memory_bytes"], payload["execution_budget"]["process_memory_bytes"])
        self.assertEqual(set(payload["capability_manifest"]["authoring_constraints"]), {
            "data_loading", "panel_lifetime", "tradability", "failure_handling", "canonical_capabilities",
        })
        schema = payload["capability_manifest"]["result_schema"]
        self.assertEqual(set(schema["required"]), set(schema["properties"]))
        self.assertEqual(
            [item["import"] for item in payload["capability_manifest"]["safe_libraries"]],
            ["import pandas as pd", "import numpy as np"],
        )
        self.assertEqual(
            authored["provenance"]["source_sha256"],
            hashlib.sha256(authored["program"].encode()).hexdigest(),
        )
        self.assertEqual(client.calls[0]["tools"], [AUTHORING_TOOL_SCHEMA])
        self.assertEqual(client.calls[0]["tool_choice"], "required")

    def test_authoring_guidance_bounds_field_and_horizon_lifetimes_without_changing_method(self):
        payload = experiment_module.build_authoring_payload("request", spec())
        constraints = json.loads(payload["input"])["capability_manifest"]["authoring_constraints"]
        self.assertIn("unused high/low", constraints["data_loading"])
        self.assertIn("never change adjustment", constraints["data_loading"])
        self.assertIn("one forward-return horizon at a time", constraints["panel_lifetime"])
        self.assertIn("yearly diagnostics", constraints["panel_lifetime"])
        self.assertIn("do not retain a dictionary/list", payload["instructions"])

    def test_authoring_accepts_normalized_tool_call_with_empty_text(self):
        class ToolCallClient(Client):
            def create(self, payload):
                self.calls.append(payload)
                return {
                    "output_text": "",
                    "output": [{
                        "type": "function_call",
                        "name": "submit_research_program",
                        "arguments": json.dumps({"program": PROGRAM}),
                    }],
                }

        authored = author_experiment("request", spec(), client=ToolCallClient(None))

        self.assertEqual(authored["status"], "ok")
        self.assertEqual(authored["program"], PROGRAM)

    def test_malformed_authoring_is_controlled(self):
        for output in (
            "not-json",
            {"source": PROGRAM},
        ):
            with self.subTest(output=output):
                client = Client(output)
                if isinstance(output, str):
                    client.create = lambda _payload, text=output: {"output_text": text}
                authored = author_experiment("request", spec(), client=client)
                self.assertEqual(authored["status"], "error")
                self.assertEqual(authored["error_type"], "malformed_response")

    def test_invalid_source_and_policy_failures_are_distinct(self):
        syntax = author_experiment(
            "request", spec(), client=Client({"program": "def broken("})
        )
        disallowed_import = author_experiment(
            "request",
            spec(),
            client=Client({"program": "import os\ndef run():\n    return {}"}),
        )
        policy = author_experiment(
            "request",
            spec(),
            client=Client({"program": "def run():\n    open('/tmp/x', 'w')\n"}),
        )

        self.assertEqual(syntax["error_type"], "invalid_experiment_source")
        self.assertEqual(disallowed_import["error_type"], "invalid_experiment_source")
        self.assertIn("import os", disallowed_import["error"])
        self.assertEqual(policy["error_type"], "experiment_policy_violation")

    def test_manifested_dataframe_imports_are_allowed(self):
        program = (
            "import pandas as pd\n"
            "import numpy as np\n\n"
            "def run():\n"
            "    values = pd.Series([1.0, np.nan]).dropna()\n"
            "    return {'assumptions': [], 'method': {}, "
            "'data_coverage': {'actual_start': None, 'actual_end': None}, "
            "'metrics': {'mean': float(values.mean())}, "
            "'sample_counts': {'rows': len(values)}, 'warnings': []}\n"
        )

        authored = author_experiment("request", spec(), client=Client({"program": program}))

        self.assertEqual(authored["status"], "ok")

    def test_spec_required_canonical_calls_cannot_be_replaced_by_proxies(self):
        for module, name in (("research.panel", "price_limit_pct_panel"), ("utils.loader", "load_index")):
            with self.subTest(api=name):
                required_spec = {**spec(), "method": f"Use `{module}.{name}` for the requested method."}
                for prefix in ("", f"from {module} import {name}\n", f"# {name}()\n"):
                    fallback = author_experiment("request", required_spec, client=Client({"program": prefix + PROGRAM}))
                    self.assertEqual(fallback["error_type"], "invalid_experiment_source")
                    self.assertIn(f"must call spec-required canonical APIs: {module}.{name}", fallback["error"])
                    self.assertIsNone(fallback["program"])
                call_args = "'2025-01-01', '2025-01-02'" if name == "price_limit_pct_panel" else "'000300', '2025-01-01', '2025-01-02'"
                program = f"from {module} import {name} as canonical\n" + PROGRAM.replace(
                    "def run():", f"def run():\n    canonical({call_args})"
                )
                accepted = author_experiment("request", required_spec, client=Client({"program": program}))
                self.assertEqual(accepted["status"], "ok", accepted)
                chinese_spec = {**required_spec, "method": f"必须使用{module}.{name}计算"}
                rejected = author_experiment("request", chinese_spec, client=Client({"program": PROGRAM}))
                self.assertEqual(rejected["error_type"], "invalid_experiment_source")
                self.assertIn(f"{module}.{name}", rejected["error"])
        payload = experiment_module.build_authoring_payload("request", required_spec)
        self.assertIn("do not substitute an approximate proxy", payload["instructions"])
        self.assertIn("required calls, not suggestions", CAPABILITY_MANIFEST["authoring_constraints"]["canonical_capabilities"])

    def test_source_artifact_failure_stops_before_execution(self):
        authored = author_experiment("request", spec(), client=Client({"program": PROGRAM}))
        with tempfile.TemporaryDirectory() as data_path, patch.dict(
            os.environ, {"DATA_PATH": data_path}
        ), patch.object(
            experiment_module, "_persist_program", side_effect=OSError("read-only")
        ), patch.object(experiment_module, "_execute_isolated_source") as execute:
            result = run_research_experiment(
                spec(),
                authored_program=authored["program"],
                authoring_provenance=authored["provenance"],
            )

        self.assertEqual(result.status, "error")
        self.assertEqual(result.errors[0]["code"], "experiment_artifact_unavailable")
        execute.assert_not_called()

    def test_dirty_research_source_changes_worktree_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "research/panel.py"
            source.parent.mkdir(parents=True)
            source.write_text("def value(): return 1\n", encoding="utf-8")
            before = experiment_module._worktree_fingerprint(root)
            source.write_text("def value(): return 2\n", encoding="utf-8")
            after = experiment_module._worktree_fingerprint(root)

        self.assertNotEqual(before, after)

    def test_result_and_provenance_contracts_validate(self):
        normalized_spec, spec_error = validate_experiment_spec(spec())
        self.assertIsNone(spec_error)
        self.assertEqual(normalized_spec, spec())

        result = {
            "assumptions": ["proxy=000001"],
            "method": {"type": "event_study"},
            "data_coverage": {
                "requested_start": "2025-01-01",
                "requested_end": "2026-09-23",
                "actual_start": "2025-01-02",
                "actual_end": "2026-09-23",
            },
            "metrics": {"h1_mean": 0.01},
            "sample_counts": {"events": 12},
            "warnings": [],
        }
        normalized_result, result_error = validate_experiment_result(result)
        self.assertIsNone(result_error)
        self.assertEqual(normalized_result, result)

        authored = author_experiment("request", spec(), client=Client({"program": PROGRAM}))
        normalized_provenance, provenance_error = validate_experiment_provenance(
            authored["provenance"]
        )
        self.assertIsNone(provenance_error)
        self.assertEqual(normalized_provenance, authored["provenance"])
        self.assertIsNotNone(validate_experiment_result({})[1])
        self.assertIsNotNone(validate_experiment_provenance({})[1])

    def test_result_field_errors_name_every_field_and_type(self):
        valid = {
            "assumptions": [],
            "method": {"type": "event_study"},
            "data_coverage": {"actual_start": None, "actual_end": None},
            "metrics": {},
            "sample_counts": {},
            "warnings": [],
        }
        for field, invalid, expected in (
            ("assumptions", {}, "an array"),
            ("method", "event_study", "an object"),
            ("data_coverage", [], "an object"),
            ("metrics", "mean=1%", "an object"),
            ("sample_counts", [1, 2], "an object"),
            ("warnings", {}, "an array"),
        ):
            with self.subTest(field=field):
                _, error = validate_experiment_result({**valid, field: invalid})
                self.assertEqual(
                    error,
                    f"result.{field} must be {expected}, got {type(invalid).__name__}",
                )

        _, error = validate_experiment_result({**valid, "method": {}})
        self.assertEqual(error, "result.method.type is required")

    def test_result_normalization_accepts_nested_integer_year_keys(self):
        raw = {
            "assumptions": [],
            "method": {"type": "event_study"},
            "data_coverage": {"actual_start": None, "actual_end": None},
            "metrics": {"yearly_stability": {
                2021: {np.int64(5): np.float64(0.01)},
                "2022": [{2023: 0.02}],
            }},
            "sample_counts": {np.int64(2021): np.int64(12)},
            "warnings": [],
        }
        normalized = experiment_worker._normalize_json(raw)
        self.assertEqual(normalized["metrics"]["yearly_stability"], {
            "2021": {"5": 0.01}, "2022": [{"2023": 0.02}],
        })
        self.assertEqual(normalized["sample_counts"], {"2021": 12})
        validated, error = validate_experiment_result(normalized)
        self.assertIsNone(error)
        self.assertEqual(validated, normalized)

    def test_series_and_mapping_share_scalar_key_normalization(self):
        for key, expected in (
            ("2021", "2021"), (2021, "2021"), (np.int64(2021), "2021"),
            (True, "True"), (1.5, "1.5"),
            (pd.Timestamp("2021-01-01"), "2021-01-01T00:00:00"),
        ):
            for value in ({key: 0.01}, pd.Series([0.01], index=[key])):
                with self.subTest(key=key, container=type(value).__name__):
                    self.assertEqual(experiment_worker._normalize_json(value), {expected: 0.01})

    def test_series_and_mapping_reject_collisions_after_key_normalization(self):
        for key, string_key in ((2021, "2021"), (np.int64(2021), "2021"), (True, "True"), (1.5, "1.5")):
            for value in ({key: 1, string_key: 2}, pd.Series([1, 2], index=[key, string_key])):
                with self.subTest(key=key, container=type(value).__name__):
                    with self.assertRaisesRegex(TypeError, "result.metrics.yearly_stability.*duplicate keys.*JSON") as caught:
                        experiment_worker._normalize_json({"metrics": {"yearly_stability": value}})
                    self.assertEqual(experiment_worker._error_type(caught.exception), "experiment_invalid_result")
                    self.assertIn(string_key, str(caught.exception))
        # Series duplicate labels must not disappear through a to_dict() conversion.
        with self.assertRaisesRegex(TypeError, "duplicate keys.*JSON"):
            experiment_worker._normalize_json(pd.Series([1, 2], index=[2021, 2021]))

    def test_series_and_mapping_still_reject_invalid_keys(self):
        for key in (None, (2021, "group"), b"2021", object(), float("inf"), float("nan"), pd.NA, pd.NaT):
            for value in ({key: 1}, pd.Series([1], index=[key])):
                with self.subTest(key=key, container=type(value).__name__):
                    with self.assertRaisesRegex(TypeError, "result.metrics.*JSON") as caught:
                        experiment_worker._normalize_json({"metrics": value})
                    self.assertEqual(experiment_worker._error_type(caught.exception), "experiment_invalid_result")

    def test_key_normalization_does_not_repair_invalid_result_field_shapes(self):
        raw = {
            "assumptions": [], "method": "event_study",
            "data_coverage": {"actual_start": None, "actual_end": None},
            "metrics": {2021: 0.01}, "sample_counts": {}, "warnings": [],
        }
        _, error = validate_experiment_result(experiment_worker._normalize_json(raw))
        self.assertEqual(error, "result.method must be an object, got str")

    def test_exhausted_runtime_repair_propagates_the_final_execution_error(self):
        planner = Client({
            "status": "ready",
            "steps": [{"name": "run_research_experiment", "arguments": {"spec": spec()}}],
            "reason": "custom analysis",
        })
        authoring = Client({"program": PROGRAM})
        errors = [
            ToolResult.error("run_research_experiment", {"spec": spec()}, "experiment_runtime_error",
                             message, run_id="failed-experiment")
            for message in ("Initial runtime failure.", "DuckDB OutOfMemoryException: failed to allocate memory.")
        ]
        with patch("agent.agent.run_research_experiment", side_effect=errors) as execute:
            result = run_agent(
                "研究大盘下跌事件。", planner_client=planner, experiment_authoring_client=authoring,
                hitl_client=Client({"decision": "proceed", "approval_request": None, "reason": "safe"}),
                run_id="failed-experiment",
            )
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["observed"]["outcome"], {"status": "error"})
        self.assertEqual(result["error_stage"], "execution")
        self.assertEqual(result["error_type"], "experiment_runtime_error")
        self.assertEqual(result["error"], errors[-1].errors[0]["message"])
        self.assertEqual(result["research_run"].steps[0]["errors"], errors[-1].errors)
        self.assertEqual(result["research_run"].status, "failed")
        self.assertEqual(result["research_run"].steps[0]["provenance"]["repair_attempts"], 1)
        self.assertEqual(len(planner.calls), 1)
        self.assertEqual(len(authoring.calls), 2)
        self.assertEqual(execute.call_count, 2)
        self.assertIsNone(result["answer"])

    def test_policy_violation_is_terminal_without_repair(self):
        planner = Client({
            "status": "ready",
            "steps": [{"name": "run_research_experiment", "arguments": {"spec": spec()}}],
            "reason": "custom analysis",
        })
        authoring = Client({
            "program": "def run():\n    open('/tmp/forbidden', 'w')\n    return {}"
        })
        result = run_agent(
            "研究大盘下跌事件。",
            planner_client=planner,
            experiment_authoring_client=authoring,
            hitl_client=Client({
                "decision": "proceed", "approval_request": None, "reason": "safe"
            }),
        )

        self.assertEqual(len(authoring.calls), 1)
        self.assertEqual(result["research_run"].status, "failed")
        self.assertEqual(
            result["research_run"].steps[0]["errors"][0]["code"],
            "experiment_policy_violation",
        )




if __name__ == "__main__":
    unittest.main()
