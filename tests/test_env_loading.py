from __future__ import annotations

import importlib
import unittest
from unittest.mock import patch

import agent.core
from agent.core.resources import REPO_ROOT


class EnvironmentLoadingTests(unittest.TestCase):
    def test_repo_env_is_loaded_without_overriding_process_environment(self):
        with patch("dotenv.load_dotenv") as load_dotenv:
            importlib.reload(agent.core)

        load_dotenv.assert_called_once_with(REPO_ROOT / ".env", override=False)


if __name__ == "__main__":
    unittest.main()
