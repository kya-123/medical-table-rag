
import os
import unittest
from unittest.mock import patch

from api_headers import admin_headers, api_headers


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

    def test_admin_header_prefers_admin_token(self):
        with patch.dict(
            os.environ,
            {
                "RAG_API_TOKEN": "api-token",
                "RAG_ADMIN_TOKEN": "admin-token",
            },
            clear=False,
        ):
            self.assertEqual(
                admin_headers(),
                {"Authorization": "Bearer admin-token"},
            )


if __name__ == "__main__":
    unittest.main()
