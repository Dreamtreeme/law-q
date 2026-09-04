from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import verify_model_revisions  # noqa: E402


class FakeApi:
    def __init__(self, resolved: str) -> None:
        self.resolved = resolved
        self.calls: list[tuple[str, str]] = []

    def model_info(self, repo_id: str, revision: str, token: str | None) -> object:
        self.calls.append((repo_id, revision))
        return SimpleNamespace(sha=self.resolved)


def config(revision: str) -> dict[str, object]:
    return {
        "models": [
            {
                "name": "model-a",
                "hf_repo": "owner/model-a",
                "revision": revision,
            }
        ]
    }


class VerifyModelRevisionsTest(unittest.TestCase):
    def test_exact_sha_is_verified(self) -> None:
        sha = "a" * 40
        report = verify_model_revisions.verify_revisions(config(sha), api=FakeApi(sha))
        self.assertEqual(report["status"], "success")
        self.assertEqual(report["verified_models"], 1)

    def test_moving_revision_is_rejected_before_network(self) -> None:
        api = FakeApi("a" * 40)
        report = verify_model_revisions.verify_revisions(config("main"), api=api)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(api.calls, [])

    def test_resolved_sha_must_match_requested_sha(self) -> None:
        report = verify_model_revisions.verify_revisions(
            config("a" * 40), api=FakeApi("b" * 40)
        )
        self.assertEqual(report["status"], "failed")
        self.assertIn("다릅니다", report["models"][0]["error"])


if __name__ == "__main__":
    unittest.main()
