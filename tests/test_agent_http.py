from __future__ import annotations

import httpx
from pathlib import Path
import unittest
from unittest.mock import patch

from agent.http import app, _request_lock


class AgentHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def call_in_test(self, function, *args, **kwargs):
        return function(*args, **kwargs)

    async def request(self, method, path, **kwargs):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.request(method, path, **kwargs)

    async def test_healthz_does_not_call_runtime(self):
        with patch("agent.http.run_request") as run:
            response = await self.request("GET", "/healthz")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
        run.assert_not_called()

    async def test_research_delegates_to_run_request_and_returns_compact_view(self):
        result = {
            "status": "ok",
            "answer": "A concise answer.",
            "research_run": type("Run", (), {"run_id": "run-1"})(),
            "observed": {
                "outcome": {"status": "success"},
                "steps": [{"name": "inspect_universe", "status": "success"}],
                "loop": {"iterations": 1},
                "grounding": {"fully_grounded": True},
            },
            "telemetry": {"summary": {"calls": 4}},
            "error_type": None,
            "error_stage": None,
            "error": "",
        }
        with patch("agent.http.run_request", return_value=result) as run, patch(
            "agent.http.asyncio.to_thread", new=self.call_in_test
        ):
            response = await self.request(
                "POST", "/v1/research", json={"message": "Inspect the universe."}
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "status": "ok",
            "outcome": "success",
            "answer": "A concise answer.",
            "run_id": "run-1",
            "steps": [{"name": "inspect_universe", "status": "success"}],
            "loop": {"iterations": 1},
            "grounding": {"fully_grounded": True},
            "telemetry": {"calls": 4},
            "error_type": None,
            "error_stage": None,
            "error": "",
        })
        run.assert_called_once_with(
            "Inspect the universe.", provider="deepseek", retrieval="none"
        )

    async def test_missing_or_invalid_message_is_rejected(self):
        for payload in ({}, {"message": "   "}, {"message": 42}, []):
            with self.subTest(payload=payload):
                response = await self.request("POST", "/v1/research", json=payload)
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.json()["error_type"], "invalid_request")

    async def test_agent_error_is_returned_as_structured_runtime_error(self):
        result = {
            "status": "error",
            "answer": None,
            "research_run": None,
            "observed": {"outcome": {"status": "error"}, "steps": []},
            "telemetry": {"summary": {"calls": 1}},
            "error_type": "provider_error",
            "error_stage": "planning",
            "error": "offline",
        }
        with patch("agent.http.run_request", return_value=result), patch(
            "agent.http.asyncio.to_thread", new=self.call_in_test
        ):
            response = await self.request(
                "POST", "/v1/research", json={"message": "Research this."}
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "error")
        self.assertEqual(body["error_type"], "provider_error")
        self.assertEqual(body["error_stage"], "planning")

    async def test_busy_request_fails_fast(self):
        self.assertTrue(_request_lock.acquire(blocking=False))
        try:
            response = await self.request(
                "POST", "/v1/research", json={"message": "Research this."}
            )
        finally:
            _request_lock.release()

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error_type"], "busy")

    def test_service_config_is_localhost_single_worker(self):
        service = Path(__file__).parents[1] / "deploy" / "tradar-agent.service"
        text = service.read_text(encoding="utf-8")
        self.assertIn("agent.http:app", text)
        self.assertIn("--host 127.0.0.1", text)
        self.assertIn("--port 8891", text)
        self.assertIn("--workers 1", text)


if __name__ == "__main__":
    unittest.main()
