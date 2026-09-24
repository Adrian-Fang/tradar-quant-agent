from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import duckdb

import agent.tools.experiment as experiment_module
from agent.agent import run_agent
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


_SANDBOX_NAMESPACES_AVAILABLE = _sandbox_namespaces_available()


def _requires_sandbox(test):
    return unittest.skipUnless(
        _SANDBOX_NAMESPACES_AVAILABLE,
        "runner does not permit the Linux namespaces required by experiment isolation",
    )(test)


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
        self.assertEqual(payload["capability_manifest"]["version"], "1.4.0")
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

    @_requires_sandbox
    def test_execution_returns_validated_result_and_provenance(self):
        authored = author_experiment("request", spec(), client=Client({"program": PROGRAM}))
        with tempfile.TemporaryDirectory() as data_path, patch.dict(
            os.environ, {"DATA_PATH": data_path}
        ):
            result = run_research_experiment(
                spec(),
                authored_program=authored["program"],
                authoring_provenance=authored["provenance"],
                run_id="run-1",
            )

        self.assertEqual(result.status, "success")
        self.assertEqual(result.result["sample_counts"], {"rows": 2})
        self.assertEqual(result.provenance["executor"], "unshare-chroot-v1")
        self.assertEqual(
            result.provenance["actual_data_bounds"],
            {"start": "2025-01-02", "end": "2025-01-03"},
        )
        self.assertEqual(result.provenance["validation_status"]["result"], "passed")
        self.assertNotIn("program", json.dumps(result.normalized_args))
        artifact_id = result.artifacts[0]["id"]
        artifact_path = self.artifact_dir / f"{result.provenance['source_sha256']}.py"
        self.assertEqual(artifact_id, f"sha256:{result.provenance['source_sha256']}")
        self.assertEqual(artifact_path.read_text(encoding="utf-8"), authored["program"])
        self.assertEqual(stat.S_IMODE(artifact_path.stat().st_mode), 0o600)
        self.assertNotIn(authored["program"], result.to_json())

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

    @_requires_sandbox
    def test_sandbox_exposes_canonical_loader_but_not_other_data_files(self):
        source = (
            "import pandas as pd\n"
            "from utils.loader import load_index, load_prices\n\n"
            "def run():\n"
            "    try:\n"
            "        pd.read_csv('/data/credentials.csv')\n"
            "        secret_file_visible = True\n"
            "    except FileNotFoundError:\n"
            "        secret_file_visible = False\n"
            "    index = load_index('000300', '2025-01-02', '2025-01-02')\n"
            "    prices = load_prices('2025-01-02', '2025-01-02', "
            "adjust='backward', symbols=['000001'], fields=['close'])\n"
            "    return {'assumptions': [], 'method': {'type': 'read_surface'}, "
            "'data_coverage': {'actual_start': None, 'actual_end': None}, "
            "'metrics': {'secret_file_visible': secret_file_visible, "
            "'index_close': float(index.iloc[0]), "
            "'stock_close': float(prices.iloc[0]['close'])}, "
            "'sample_counts': {'prices': len(prices)}, 'warnings': []}\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory)
            (data_path / "credentials.csv").write_text(
                "secret\nnot-for-agent\n", encoding="utf-8"
            )
            connection = duckdb.connect(str(data_path / "tradar.duckdb"))
            try:
                connection.execute(
                    "CREATE TABLE market_index(symbol VARCHAR, date DATE, close DOUBLE)"
                )
                connection.execute(
                    "INSERT INTO market_index VALUES ('000300', '2025-01-02', 3000)"
                )
                connection.execute(
                    "CREATE TABLE history(symbol VARCHAR, date DATE, open DOUBLE, "
                    "high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE, amount DOUBLE)"
                )
                connection.execute(
                    "INSERT INTO history VALUES "
                    "('000001', '2025-01-02', 10, 10, 10, 10, 100, 1000)"
                )
                connection.execute(
                    "CREATE TABLE ex_factors(symbol VARCHAR, date DATE, ex_factor DOUBLE)"
                )
                connection.execute(
                    "INSERT INTO ex_factors VALUES ('000001', '2025-01-02', 1)"
                )
            finally:
                connection.close()

            result = _execute_isolated_source(source, data_path=data_path)

            self.assertFalse((data_path / "cache").exists())

        self.assertEqual(result["status"], "success", result)
        self.assertEqual(result["result"]["metrics"], {
            "secret_file_visible": False,
            "index_close": 3000.0,
            "stock_close": 10.0,
        })

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

    @_requires_sandbox
    def test_runtime_authors_executes_and_does_not_leak_source(self):
        planner = Client({
            "status": "ready",
            "steps": [{"name": "run_research_experiment", "arguments": {"spec": spec()}}],
            "reason": "custom analysis",
        })
        hitl = Client({"decision": "proceed", "approval_request": None, "reason": "safe"})
        authoring = Client({"program": EVENT_PROGRAM})
        synthesis = Client({
            "status": "success",
            "answer": "Fixture completed.",
            "evidence_ids": ["step-1-run_research_experiment"],
        })
        grounding = Client({
            "answer": "ignored",
            "claims": [{
                "claim": "Fixture completed.",
                "evidence_ids": ["step-1-run_research_experiment"],
                "grounding": "supported",
            }],
        })

        with tempfile.TemporaryDirectory() as data_path, patch.dict(
            os.environ, {"DATA_PATH": data_path}
        ):
            result = run_agent(
                "研究大盘下跌事件。",
                planner_client=planner,
                experiment_authoring_client=authoring,
                hitl_client=hitl,
                synthesis_client=synthesis,
                grounding_client=grounding,
            )

        self.assertEqual(len(authoring.calls), 1)
        self.assertEqual(len(planner.calls), 1)
        self.assertEqual(result["observed"]["loop"]["iterations"], 1)
        self.assertEqual(result["research_run"].status, "completed")
        self.assertEqual(result["research_run"].steps[0]["status"], "success")
        self.assertEqual(
            result["research_run"].steps[0]["provenance"]["source_sha256"],
            hashlib.sha256(EVENT_PROGRAM.encode()).hexdigest(),
        )
        self.assertEqual(
            result["research_run"].steps[0]["artifacts"],
            [{
                "kind": "experiment_source",
                "id": f"sha256:{hashlib.sha256(EVENT_PROGRAM.encode()).hexdigest()}",
            }],
        )
        public = json.dumps(result, ensure_ascii=False, default=str)
        self.assertNotIn("def run", public)
        self.assertNotIn("program", json.dumps(result["plan"], ensure_ascii=False))
        evidence = json.loads(result["evidence"][0]["text"])
        self.assertEqual(evidence["result"]["sample_counts"], {"event_days": 2})
        self.assertEqual(
            evidence["result"]["metrics"]["mean_forward_return"],
            {"1": 0.01, "3": 0.02},
        )
        self.assertEqual(
            evidence["provenance"]["actual_data_bounds"],
            {"start": "2025-01-02T00:00:00", "end": "2025-01-03T00:00:00"},
        )
        self.assertEqual(evidence["provenance"]["validation_status"]["result"], "passed")
        self.assertIn(
            "experiment_authoring",
            [call["stage"] for call in result["telemetry"]["calls"]],
        )

    @_requires_sandbox
    def test_runtime_repairs_once_after_runtime_failure(self):
        planner = Client({
            "status": "ready",
            "steps": [{"name": "run_research_experiment", "arguments": {"spec": spec()}}],
            "reason": "custom analysis",
        })
        hitl = Client({"decision": "proceed", "approval_request": None, "reason": "safe"})

        class RepairClient(Client):
            def __init__(self):
                super().__init__(None)

            def create(self, payload):
                self.calls.append(payload)
                program = (
                    "def run():\n    raise RuntimeError('broken')\n"
                    if len(self.calls) == 1 else PROGRAM
                )
                return response({"program": program})

        authoring = RepairClient()
        with tempfile.TemporaryDirectory() as data_path, patch.dict(
            os.environ, {"DATA_PATH": data_path}
        ):
            result = run_agent(
                "研究大盘下跌事件。",
                planner_client=planner,
                experiment_authoring_client=authoring,
                hitl_client=hitl,
                answer="Fixture completed.",
                evidence=[{"id": "step-1-run_research_experiment", "text": "fixture"}],
                grounding_client=Client({
                    "answer": "ignored",
                    "claims": [{
                        "claim": "Fixture completed.",
                        "evidence_ids": ["step-1-run_research_experiment"],
                        "grounding": "supported",
                    }],
                }),
            )

        self.assertEqual(result["research_run"].status, "completed")
        self.assertEqual(len(authoring.calls), 2)
        self.assertIn(
            "repair_feedback", json.loads(authoring.calls[1]["input"])
        )
        self.assertEqual(
            result["research_run"].steps[0]["provenance"]["repair_attempts"], 1
        )

    @_requires_sandbox
    def test_runtime_repairs_once_after_disallowed_import(self):
        planner = Client({
            "status": "ready",
            "steps": [{"name": "run_research_experiment", "arguments": {"spec": spec()}}],
            "reason": "custom analysis",
        })

        class RepairClient(Client):
            def __init__(self):
                super().__init__(None)

            def create(self, payload):
                self.calls.append(payload)
                program = "import os\n" + PROGRAM if len(self.calls) == 1 else PROGRAM
                return response({"program": program})

        authoring = RepairClient()
        with tempfile.TemporaryDirectory() as data_path, patch.dict(
            os.environ, {"DATA_PATH": data_path}
        ):
            result = run_agent(
                "研究大盘下跌事件。",
                planner_client=planner,
                experiment_authoring_client=authoring,
                hitl_client=Client({
                    "decision": "proceed", "approval_request": None, "reason": "safe"
                }),
                answer="Fixture completed.",
                evidence=[{"id": "step-1-run_research_experiment", "text": "fixture"}],
                grounding_client=Client({
                    "answer": "ignored",
                    "claims": [{
                        "claim": "Fixture completed.",
                        "evidence_ids": ["step-1-run_research_experiment"],
                        "grounding": "supported",
                    }],
                }),
            )

        self.assertEqual(result["research_run"].status, "completed")
        self.assertEqual(len(authoring.calls), 2)
        self.assertIn(
            "import os", json.loads(authoring.calls[1]["input"])["repair_feedback"]
        )
        self.assertEqual(
            result["research_run"].steps[0]["provenance"]["repair_attempts"], 1
        )

    @_requires_sandbox
    def test_runtime_repairs_once_after_malformed_authoring(self):
        planner = Client({
            "status": "ready",
            "steps": [{"name": "run_research_experiment", "arguments": {"spec": spec()}}],
            "reason": "custom analysis",
        })

        class RepairClient(Client):
            def __init__(self):
                super().__init__(None)

            def create(self, payload):
                self.calls.append(payload)
                if len(self.calls) == 1:
                    return {"output_text": "```json\n{}\n```"}
                return response({"program": PROGRAM})

        authoring = RepairClient()
        with tempfile.TemporaryDirectory() as data_path, patch.dict(
            os.environ, {"DATA_PATH": data_path}
        ):
            result = run_agent(
                "研究大盘下跌事件。",
                planner_client=planner,
                experiment_authoring_client=authoring,
                hitl_client=Client({
                    "decision": "proceed", "approval_request": None, "reason": "safe"
                }),
                answer="Fixture completed.",
                evidence=[{"id": "step-1-run_research_experiment", "text": "fixture"}],
                grounding_client=Client({
                    "answer": "ignored",
                    "claims": [{
                        "claim": "Fixture completed.",
                        "evidence_ids": ["step-1-run_research_experiment"],
                        "grounding": "supported",
                    }],
                }),
            )

        self.assertEqual(result["research_run"].status, "completed")
        self.assertEqual(len(authoring.calls), 2)
        self.assertIn(
            "not valid JSON", json.loads(authoring.calls[1]["input"])["repair_feedback"]
        )
        self.assertEqual(
            result["research_run"].steps[0]["provenance"]["repair_attempts"], 1
        )

    @_requires_sandbox
    def test_runtime_repairs_once_after_invalid_result_shape(self):
        planner = Client({
            "status": "ready",
            "steps": [{"name": "run_research_experiment", "arguments": {"spec": spec()}}],
            "reason": "custom analysis",
        })
        invalid_program = (
            "def run():\n"
            "    return {'assumptions': [], 'method': 'event_study', "
            "'data_coverage': {'actual_start': None, 'actual_end': None}, "
            "'metrics': {}, 'sample_counts': {}, 'warnings': []}\n"
        )

        class RepairClient(Client):
            def __init__(self):
                super().__init__(None)

            def create(self, payload):
                self.calls.append(payload)
                return response({
                    "program": invalid_program if len(self.calls) == 1 else PROGRAM
                })

        authoring = RepairClient()
        with tempfile.TemporaryDirectory() as data_path, patch.dict(
            os.environ, {"DATA_PATH": data_path}
        ):
            result = run_agent(
                "研究大盘下跌事件。",
                planner_client=planner,
                experiment_authoring_client=authoring,
                hitl_client=Client({
                    "decision": "proceed", "approval_request": None, "reason": "safe"
                }),
                answer="Fixture completed.",
                evidence=[{"id": "step-1-run_research_experiment", "text": "fixture"}],
                grounding_client=Client({
                    "answer": "ignored",
                    "claims": [{
                        "claim": "Fixture completed.",
                        "evidence_ids": ["step-1-run_research_experiment"],
                        "grounding": "supported",
                    }],
                }),
            )

        self.assertEqual(result["research_run"].status, "completed")
        self.assertEqual(len(authoring.calls), 2)
        self.assertIn(
            "result.method must be an object, got str",
            json.loads(authoring.calls[1]["input"])["repair_feedback"],
        )
        self.assertEqual(
            result["research_run"].steps[0]["provenance"]["repair_attempts"], 1
        )

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


@_requires_sandbox
class ExperimentSandboxTests(unittest.TestCase):
    def execute(self, source, **kwargs):
        with tempfile.TemporaryDirectory() as data_path:
            return _execute_isolated_source(source, data_path=data_path, **kwargs)

    def test_host_secrets_and_dotenv_are_not_visible(self):
        source = isolated_program(
            "import os\n"
            "def probe():\n"
            "    return {\n"
            "        'secret': os.environ.get('TRADAR_TEST_SECRET'),\n"
            "        'repo_dotenv': os.path.exists('/app/.env'),\n"
            "        'host_dotenv': os.path.exists('/home/ubuntu/tradar-quant-agent/.env'),\n"
            "    }"
        )
        with patch.dict(os.environ, {"TRADAR_TEST_SECRET": "must-not-leak"}):
            result = self.execute(source)

        self.assertEqual(result["status"], "success")
        self.assertEqual(
            result["result"]["metrics"],
            {"secret": None, "repo_dotenv": False, "host_dotenv": False},
        )

    def test_repo_and_data_are_read_only_and_workspace_is_writable(self):
        source = isolated_program(
            "def probe():\n"
            "    blocked = []\n"
            "    for path in ('/sandbox-probe', '/app/research/sandbox-probe', "
            "'/data/sandbox-probe'):\n"
            "        try:\n"
            "            open(path, 'w').write('bad')\n"
            "        except OSError:\n"
            "            blocked.append(path)\n"
            "    open('/work/allowed', 'w').write('ok')\n"
            "    return {'blocked': blocked, 'workspace': open('/work/allowed').read()}"
        )
        probe = Path("research/sandbox-probe")
        self.assertFalse(probe.exists())
        result = self.execute(source)

        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["result"]["metrics"]["blocked"]), 3)
        self.assertEqual(result["result"]["metrics"]["workspace"], "ok")
        self.assertFalse(probe.exists())

    def test_network_namespace_has_no_network(self):
        source = isolated_program(
            "import socket\n"
            "def probe():\n"
            "    sock = None\n"
            "    try:\n"
            "        sock = socket.socket()\n"
            "        sock.settimeout(0.2)\n"
            "        sock.connect(('1.1.1.1', 53))\n"
            "        return {'blocked': False}\n"
            "    except OSError:\n"
            "        return {'blocked': True}\n"
            "    finally:\n"
            "        if sock is not None:\n"
            "            sock.close()"
        )
        result = self.execute(source)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["result"]["metrics"], {"blocked": True})

    def test_uncaught_network_attempt_is_reported_as_sandbox_violation(self):
        source = isolated_program(
            "import socket\n"
            "def probe():\n"
            "    sock = socket.socket()\n"
            "    sock.settimeout(0.2)\n"
            "    sock.connect(('1.1.1.1', 53))\n"
            "    sock.close()\n"
            "    return {'blocked': False}"
        )
        result = self.execute(source)

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "experiment_sandbox_violation")

    def test_process_limit_blocks_fork(self):
        result = self.execute(
            "import os\n\ndef run():\n    os.fork()\n    return {}\n"
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "experiment_sandbox_violation")

    def test_open_file_limit_is_enforced(self):
        source = (
            "def run():\n"
            "    files = [open('/work/file-' + str(i), 'w') for i in range(100)]\n"
            "    return {}\n"
        )
        result = self.execute(source)

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "experiment_sandbox_violation")

    def test_wall_timeout_kills_the_namespace(self):
        result = self.execute(
            "def run():\n    while True:\n        pass\n",
            wall_timeout=2,
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "experiment_timeout")

    def test_cpu_limit_is_enforced(self):
        result = self.execute(
            "def run():\n    while True:\n        pass\n",
            wall_timeout=10,
            cpu_seconds=1,
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "experiment_resource_limit")

    def test_memory_limit_is_enforced(self):
        result = self.execute(
            "def run():\n    value = bytearray(256 * 1024 * 1024)\n    return value\n",
            memory_bytes=128 * 1024**2,
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "experiment_resource_limit")

    def test_output_limit_is_enforced(self):
        result = self.execute(
            "def run():\n    print('x' * 70000)\n    return {}\n"
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "experiment_resource_limit")

    def test_missing_isolation_fails_closed(self):
        with patch("agent.tools.experiment.shutil.which", return_value=None):
            result = self.execute(PROGRAM)

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "experiment_unavailable")


if __name__ == "__main__":
    unittest.main()
