
import os
import unittest
from unittest.mock import patch

from api_headers import api_headers


class ApiHeadersTest(unittest.TestCase):
    def test_without_token_has_no_authorization(self):
        with patch.dict(os.environ, {"RAG_API_TOKEN": ""}, clear=False):
            self.assertEqual(api_headers(), {})

    def test_with_token_adds_bearer_header(self):
        with patch.dict(os.environ, {"RAG_API_TOKEN": "secret-token"}, clear=False):
            self.assertEqual(
                api_headers(),
                {"Authorization": "Bearer secret-token"},
            )


if __name__ == "__main__":
    unittest.main()
