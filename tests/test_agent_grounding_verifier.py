from __future__ import annotations

import json
import unittest

from agent.grounding.eval import CASES, FixtureClient, run_case
from agent.grounding.verifier import verify_answer_grounding


EVIDENCE = [{"id": "e1", "text": "The measured result is positive."}]


class FakeClient:
    def __init__(self, output=None, error=None):
        self.output = output
        self.error = error
        self.calls = []

    def create(self, payload):
        self.calls.append(payload)
        if self.error:
            raise self.error
        text = self.output if isinstance(self.output, str) else json.dumps(self.output)
        return {"output_text": text}


def output(*claims, answer="model answer"):
    return {"answer": answer, "claims": list(claims)}


def claim(label, evidence_ids=("e1",), text="claim"):
    return {"claim": text, "evidence_ids": list(evidence_ids), "grounding": label}


class GroundingVerifierTests(unittest.TestCase):
    def test_fully_supported_single_claim_uses_original_answer(self):
        client = FakeClient(output=output(claim("supported")))
        result = verify_answer_grounding("original answer", EVIDENCE, client=client)

        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["assessment"]["fully_grounded"])
        self.assertEqual(result["assessment"]["answer"], "original answer")
        self.assertEqual(len(client.calls), 1)

    def test_supported_multiple_claims_are_fully_grounded(self):
        evidence = [{"id": "e1", "text": "one"}, {"id": "e2", "text": "two"}]
        result = verify_answer_grounding(
            "answer",
            evidence,
            client=FakeClient(output=output(
                claim("supported", ("e1",)), claim("supported", ("e2",)),
            )),
        )

        self.assertTrue(result["assessment"]["fully_grounded"])

    def test_non_supported_labels_complete_successfully_but_are_not_fully_grounded(self):
        for label in ("unsupported", "contradicted", "unverifiable"):
            result = verify_answer_grounding(
                "answer",
                EVIDENCE,
                client=FakeClient(output=output(claim(label))),
            )
            self.assertEqual(result["status"], "ok")
            self.assertFalse(result["assessment"]["fully_grounded"])

    def test_mixed_labels_are_not_fully_grounded(self):
        result = verify_answer_grounding(
            "answer",
            [{"id": "e1", "text": "one"}, {"id": "e2", "text": "two"}],
            client=FakeClient(output=output(
                claim("supported", ("e1",)), claim("unverifiable", ("e2",)),
            )),
        )

        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["assessment"]["fully_grounded"])

    def test_provider_error_is_returned_without_retry(self):
        client = FakeClient(error=RuntimeError("offline"))
        result = verify_answer_grounding("answer", EVIDENCE, client=client)

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "provider_error")
        self.assertIsNone(result["assessment"])
        self.assertEqual(len(client.calls), 1)

    def test_malformed_shape_and_unknown_evidence_are_malformed_response(self):
        cases = [
            "not json",
            output({"claim": "x", "evidence_ids": ["e1"]}),
            output(claim("supported", ("missing",))),
            output(claim("other",)),
        ]
        for bad in cases:
            client = FakeClient(output=bad)
            result = verify_answer_grounding("answer", EVIDENCE, client=client)
            self.assertEqual(result["status"], "error")
            self.assertEqual(result["error_type"], "malformed_response")

    def test_prompt_contains_only_runtime_answer_and_evidence(self):
        client = FakeClient(output=output(claim("supported")))
        verify_answer_grounding("answer", EVIDENCE, client=client)
        body = json.loads(client.calls[0]["input"])

        self.assertEqual(set(body), {"answer", "evidence"})
        self.assertNotIn("expected_label", client.calls[0]["input"])
        self.assertNotIn("expected_claims", client.calls[0]["input"])
        self.assertNotIn("slice", client.calls[0]["input"])

    def test_input_validation_rejects_empty_or_duplicate_evidence(self):
        with self.assertRaises(ValueError):
            verify_answer_grounding("", EVIDENCE, client=FakeClient())
        with self.assertRaises(ValueError):
            verify_answer_grounding("answer", [], client=FakeClient())
        with self.assertRaises(ValueError):
            verify_answer_grounding(
                "answer",
                [{"id": "e1", "text": "one"}, {"id": "e1", "text": "two"}],
                client=FakeClient(),
            )

    def test_eval_uses_shared_runtime_parser(self):
        case = CASES[0]
        outcome = run_case(case, client=FixtureClient(case), model="fixture")

        self.assertIsNone(outcome["parse_error"])
        self.assertIsNone(outcome["structure_error"])
        self.assertEqual(outcome["parsed"]["claims"][0]["grounding"], "supported")


if __name__ == "__main__":
    unittest.main()
