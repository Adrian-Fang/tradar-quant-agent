"""Opt-in real isolation/resource regressions; excluded from default pytest collection."""

import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

import duckdb

import agent.tools.experiment as experiment_module
from agent.agent import run_agent
from agent.tools.experiment import (
    CAPABILITY_MANIFEST, _execute_isolated_source, author_experiment,
    run_research_experiment, validate_experiment_provenance, validate_experiment_result,
)
from tests import test_agent_experiment as fixtures
from tests.test_agent_experiment import Client, EVENT_PROGRAM, PROGRAM, isolated_program, response, spec

_SANDBOX_NAMESPACES_AVAILABLE = fixtures._sandbox_namespaces_available()


def _requires_sandbox(test):
    return unittest.skipUnless(
        _SANDBOX_NAMESPACES_AVAILABLE,
        "runner does not permit the Linux namespaces required by experiment isolation",
    )(test)


class ExperimentIntegrationTests(unittest.TestCase):
    # Module-qualified reuse does not expose the original TestCase to collection.
    setUp = fixtures.ExperimentArchitectureTests.setUp
    @_requires_sandbox
    def test_local_momentum_and_low_vol_breakout_contract_paths(self):
        prefix = (
            "import numpy as np\nimport pandas as pd\n"
            "from research.panel import price_panel, eligible_universe_mask, buyable_mask\n"
            "def run():\n"
            "    close = price_panel('2021-01-01', '2021-12-31', field='close')\n"
            "    eligible = eligible_universe_mask('2021-01-01', '2021-12-31')\n"
        )
        programs = {
            "momentum": prefix +
                "    values = (close / close.shift(20) - 1).where(eligible)\n",
            "low_vol_breakout": prefix +
                "    volume = price_panel('2021-01-01', '2021-12-31', field='volume')\n"
                "    low = close.pct_change().rolling(60).std().rank(axis=1, pct=True).le(1/3)\n"
                "    breakout = (close > close.rolling(20).max().shift(1)) & (volume > volume.rolling(20).mean() * 1.5)\n"
                "    buy = buyable_mask('2021-01-01', '2021-12-31')\n"
                "    values = (close.shift(-20) / close - 1).where(eligible & low & breakout & buy)\n",
        }
        with tempfile.TemporaryDirectory() as directory:
            db = duckdb.connect(str(Path(directory) / "tradar.duckdb"))
            db.execute("""CREATE TABLE history AS
                SELECT lpad(s::VARCHAR,6,'0') symbol, DATE '2021-01-01'+d::INTEGER date,
                       (10 + d * 0.2 + s * 0.01)::DOUBLE AS close,
                       (CASE WHEN d%25=0 THEN 400 ELSE 100 END)::DOUBLE volume
                FROM range(120) stocks(s), range(200) days(d)""")
            db.execute("CREATE TABLE ex_factors(symbol VARCHAR, date DATE, ex_factor DOUBLE)")
            db.execute("INSERT INTO ex_factors SELECT DISTINCT symbol,DATE '2000-01-01',1.2 FROM history")
            db.execute("""CREATE TABLE hist_ext AS SELECT symbol,date,false is_st,
                true is_trading,2.0 turnover_rate,0.0 change_pct FROM history""")
            db.execute("CREATE TABLE universe AS SELECT DISTINCT symbol, DATE '2000-01-01' list_date,'SH' exchange FROM history")
            db.close()
            with patch.dict(os.environ, {"DATA_PATH": directory}):
                for name, source in programs.items():
                    with self.subTest(method=name):
                        source += (
                            "    return {'assumptions': ['fixture universe'], 'method': {'type': '" + name + "'}, "
                            "'data_coverage': {'actual_start': str(close.index.min().date()), 'actual_end': str(close.index.max().date())}, "
                            "'metrics': {'mean': float(values.mean().mean())}, 'sample_counts': {'observations': int(values.count().sum())}, 'warnings': []}\n"
                        )
                        authored = author_experiment("local fixture", spec(), client=Client({"program": source}))
                        self.assertEqual(authored["status"], "ok", authored)
                        result = run_research_experiment(spec(), authored_program=authored["program"],
                                                        authoring_provenance=authored["provenance"])
                        self.assertEqual(result.status, "success", result.errors)
                        self.assertGreater(result.result["sample_counts"]["observations"], 0)
                        self.assertIsNone(validate_experiment_result(result.result)[1])
                        provenance = {key: result.provenance[key] for key in experiment_module.EXPERIMENT_PROVENANCE_FIELDS}
                        self.assertIsNone(validate_experiment_provenance(provenance)[1])
                        self.assertEqual(result.provenance["validation_status"]["execution"], "passed")
                        self.assertNotIn("def run", result.to_json())
                self.assertFalse((Path(directory) / "cache").exists())

    @_requires_sandbox
    def test_authored_price_limit_panel_uses_canonical_board_rules_and_nan(self):
        required_spec = {**spec(), "method": "Use research.panel.price_limit_pct_panel for daily thresholds."}
        program = (
            "from research.panel import price_limit_pct_panel as canonical_limits\n"
            "def run():\n"
            "    limits = canonical_limits('2025-01-02', '2025-01-02')\n"
            "    return {'assumptions': [], 'method': {'type': 'canonical_price_limits'}, "
            "'data_coverage': {'actual_start': '2025-01-02', 'actual_end': '2025-01-02'}, "
            "'metrics': {'limits': limits.iloc[0].to_dict()}, "
            "'sample_counts': {'known_limits': int(limits.notna().sum().sum())}, "
            "'warnings': ['NaN limits are not replaced by a fixed proxy']}\n"
        )
        api = next(item for item in CAPABILITY_MANIFEST["apis"] if item["name"] == "price_limit_pct_panel")
        self.assertEqual(api["signature"], "price_limit_pct_panel(start, end)")
        self.assertIn("10.0 means 10%", api["returns"])
        self.assertIn("NaN", api["notes"])
        authored = author_experiment("local fixture", required_spec, client=Client({"program": program}))
        self.assertEqual(authored["status"], "ok", authored)
        with tempfile.TemporaryDirectory() as directory:
            connection = duckdb.connect(str(Path(directory) / "tradar.duckdb"))
            connection.execute("CREATE TABLE history(date DATE)")
            connection.execute("INSERT INTO history VALUES ('2025-01-02')")
            connection.execute("CREATE TABLE universe(symbol VARCHAR, list_date DATE, exchange VARCHAR)")
            connection.execute("""INSERT INTO universe VALUES
                ('600001','2000-01-01','SH'),('600002','2000-01-01','SH'),
                ('300001','2000-01-01','SZ'),('688001','2000-01-01','SH'),
                ('800001','2000-01-01','BJ'),('600003','2025-01-02','SH'),
                ('000001','2000-01-01','SZ')""")
            connection.execute("""CREATE TABLE hist_ext AS SELECT symbol, DATE '2025-01-02' date,
                symbol='600002' is_st, symbol!='000001' is_trading FROM universe""")
            connection.close()
            with patch.dict(os.environ, {"DATA_PATH": directory}):
                result = run_research_experiment(required_spec, authored_program=authored["program"],
                                                authoring_provenance=authored["provenance"])
        self.assertEqual(result.status, "success", result.errors)
        self.assertEqual(result.result["metrics"]["limits"], {
            "600001": 10.0, "600002": 5.0, "300001": 20.0, "688001": 20.0,
            "800001": 30.0, "600003": None, "000001": None,
        })
        self.assertEqual(result.result["sample_counts"]["known_limits"], 5)

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

    @_requires_sandbox
    def test_sandbox_exposes_canonical_loader_but_not_other_data_files(self):
        source = (
            "import pandas as pd\n"
            "import duckdb\n"
            "settings = []\n"
            "worker_connect = duckdb.connect\n"
            "def capture_connect(*args, **kwargs):\n"
            "    c = worker_connect(*args, **kwargs)\n"
            "    c.register('resource_probe', pd.DataFrame({'value': [1]}))\n"
            "    settings.append(dict(c.execute(\"SELECT name,value FROM duckdb_settings() WHERE name IN "
            "('threads','memory_limit','temp_directory','max_temp_directory_size')\").fetchall()))\n"
            "    return c\n"
            "duckdb.connect = capture_connect\n"
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
            "    load_prices('2025-01-02', '2025-01-02', adjust='forward', fields=['close'])\n"
            "    return {'assumptions': [], 'method': {'type': 'read_surface'}, "
            "'data_coverage': {'actual_start': None, 'actual_end': None}, "
            "'metrics': {'secret_file_visible': secret_file_visible, "
            "'index_close': float(index.iloc[0]), "
            "'stock_close': float(prices.iloc[0]['close']), 'connection_settings': settings}, "
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
        metrics = result["result"]["metrics"]
        self.assertEqual(metrics.pop("connection_settings"), [{
            "threads": "1", "memory_limit": "256.0 MiB",
            "temp_directory": "/work/cache/duckdb", "max_temp_directory_size": "256.0 MiB",
        }] * 3)
        self.assertEqual(metrics, {
            "secret_file_visible": False,
            "index_close": 3000.0,
            "stock_close": 10.0,
        })

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
                client=planner,
                experiment_authoring_client=authoring,
                hitl_client=hitl,
                synthesis_client=synthesis,
                grounding_client=grounding,
            )

        self.assertEqual(len(authoring.calls), 1)
        self.assertEqual(len(planner.calls), 2)
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
        self.assertEqual(evidence["result"]["metrics"]["yearly_stability"], {"2025": {"mean": 0.01}})
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
                client=planner,
                experiment_authoring_client=authoring,
                hitl_client=hitl,
                synthesis_client=Client({
                    "status": "success", "answer": "Fixture completed.",
                    "evidence_ids": ["step-1-run_research_experiment"],
                }),
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
                client=planner,
                experiment_authoring_client=authoring,
                hitl_client=Client({
                    "decision": "proceed", "approval_request": None, "reason": "safe"
                }),
                synthesis_client=Client({
                    "status": "success", "answer": "Fixture completed.",
                    "evidence_ids": ["step-1-run_research_experiment"],
                }),
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
                client=planner,
                experiment_authoring_client=authoring,
                hitl_client=Client({
                    "decision": "proceed", "approval_request": None, "reason": "safe"
                }),
                synthesis_client=Client({
                    "status": "success", "answer": "Fixture completed.",
                    "evidence_ids": ["step-1-run_research_experiment"],
                }),
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
                client=planner,
                experiment_authoring_client=authoring,
                hitl_client=Client({
                    "decision": "proceed", "approval_request": None, "reason": "safe"
                }),
                synthesis_client=Client({
                    "status": "success", "answer": "Fixture completed.",
                    "evidence_ids": ["step-1-run_research_experiment"],
                }),
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

@_requires_sandbox
class ExperimentSandboxTests(unittest.TestCase):
    def execute(self, source, **kwargs):
        with tempfile.TemporaryDirectory() as data_path:
            return _execute_isolated_source(source, data_path=data_path, **kwargs)

    def test_duckdb_spills_only_under_bounded_workspace(self):
        source = isolated_program(
            "import duckdb, os\n"
            "def probe():\n"
            "    c = duckdb.connect()\n"
            "    settings = dict(c.execute(\"SELECT name, value FROM duckdb_settings() WHERE name IN "
            "('threads','memory_limit','temp_directory','max_temp_directory_size')\").fetchall())\n"
            "    c.execute(\"SET memory_limit='16MiB'\")\n"
            "    n = c.execute(\"SELECT count(*) FROM (SELECT i, md5(i::VARCHAR) s "
            "FROM range(300000) t(i) ORDER BY s)\").fetchone()[0]\n"
            "    spill = os.listdir('/work/cache/duckdb')\n"
            "    c.close()\n"
            "    fs = os.statvfs('/work')\n"
            "    return {'settings': settings, 'sorted_rows': n, 'spill_files': spill, "
            "'workspace_bytes': fs.f_blocks * fs.f_frsize}"
        )
        result = self.execute(source)
        self.assertEqual(result["status"], "success", result)
        metrics = result["result"]["metrics"]
        self.assertEqual(metrics["settings"]["threads"], "1")
        self.assertEqual(metrics["settings"]["temp_directory"], "/work/cache/duckdb")
        self.assertEqual(metrics["settings"]["memory_limit"], "256.0 MiB")
        self.assertEqual(metrics["settings"]["max_temp_directory_size"], "256.0 MiB")
        self.assertEqual(metrics["sorted_rows"], 300000)
        self.assertTrue(metrics["spill_files"])
        self.assertEqual(metrics["workspace_bytes"], experiment_module.WORKSPACE_BYTES)

    def test_canonical_connection_enforces_spill_limit_not_just_reports_it(self):
        source = (
            "from utils.duckdb_manager import get_conn\n"
            "def run():\n"
            "    c = get_conn()\n"
            "    c.execute(\"SET memory_limit='32MiB'\")\n"
            "    c.execute(\"SELECT count(*) FROM (SELECT i, md5(i::VARCHAR) s "
            "FROM range(2000000) t(i) ORDER BY s)\").fetchone()\n"
            "    return {}\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            duckdb.connect(str(Path(directory) / "tradar.duckdb")).close()
            # A lower test-only bound makes enforcement deterministic and cheap.
            with patch.object(experiment_module, "DUCKDB_SPILL_BYTES", 8 * 1024**2):
                result = _execute_isolated_source(source, data_path=directory)
        self.assertEqual(result["status"], "error", result)
        self.assertEqual(result["error_type"], "experiment_resource_limit", result)
        self.assertIn("failed to offload data block", result["message"])
        self.assertIn("/8.0 MiB used", result["message"])

    def test_workspace_capacity_is_enforced(self):
        result = self.execute(
            "def run():\n"
            "    with open('/work/full', 'wb') as f:\n"
            "        block = b'x' * (1024 * 1024)\n"
            "        for i in range(513):\n"
            "            f.write(block)\n"
            "    return {}\n"
        )
        self.assertEqual(result["status"], "error", result)
        self.assertEqual(result["error_type"], "experiment_resource_limit")

    def test_native_stderr_output_is_bounded(self):
        result = self.execute("import os\ndef run():\n    os.write(2, b'x' * 70000)\n    return {}\n")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "experiment_resource_limit")

    def test_host_secrets_and_dotenv_are_not_visible(self):
        with tempfile.TemporaryDirectory() as host_directory:
            host_dotenv = Path(host_directory) / ".env"
            host_dotenv.write_text("TRADAR_TEST_SECRET=must-not-leak\n")
            self.assertTrue(host_dotenv.is_file())
            source = isolated_program(
                "import os\n"
                "def probe():\n"
                "    return {\n"
                "        'secret': os.environ.get('TRADAR_TEST_SECRET'),\n"
                "        'repo_dotenv': os.path.exists('/app/.env'),\n"
                f"        'host_dotenv': os.path.exists({str(host_dotenv)!r}),\n"
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
