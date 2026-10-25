import os
import unittest

from security import (
    SlidingWindowRateLimiter,
    configured_rate_limit,
    extract_bearer_token,
    is_local_client,
    is_public_path,
    path_requires_admin,
    token_matches,
    validate_network_security,
)


class SecurityTest(unittest.TestCase):
    def test_extract_bearer_token_is_case_insensitive(self):
        self.assertEqual(
            extract_bearer_token("bearer example-token"),
            "example-token",
        )
        self.assertEqual(extract_bearer_token("Basic example-token"), "")

    def test_token_comparison_rejects_empty_values(self):
        self.assertTrue(token_matches("same-token", "same-token"))
        self.assertFalse(token_matches("", ""))
        self.assertFalse(token_matches("wrong-token", "same-token"))

    def test_admin_and_public_paths(self):
        self.assertTrue(path_requires_admin("/upload"))
        self.assertTrue(path_requires_admin("/delete"))
        self.assertTrue(path_requires_admin("/profiles"))
        self.assertFalse(path_requires_admin("/ask"))
        self.assertTrue(is_public_path("/health"))
        self.assertTrue(is_public_path("/docs"))
        self.assertFalse(is_public_path("/ask"))

    def test_external_bind_requires_distinct_tokens(self):
        validate_network_security("127.0.0.1", "", "")
        with self.assertRaises(ValueError):
            validate_network_security("0.0.0.0", "", "")
        with self.assertRaises(ValueError):
            validate_network_security("0.0.0.0", "api-token", "")
        with self.assertRaises(ValueError):
            validate_network_security("0.0.0.0", "same-token", "same-token")
        validate_network_security("0.0.0.0", "api-token", "admin-token")

    def test_local_client_detection(self):
        self.assertTrue(is_local_client("127.0.0.1"))
        self.assertTrue(is_local_client("::1"))
        self.assertTrue(is_local_client("testclient"))
        self.assertFalse(is_local_client("192.168.1.20"))

    def test_rate_limiter_blocks_after_limit(self):
        limiter = SlidingWindowRateLimiter(window_seconds=60)
        self.assertEqual(limiter.allow("client", 2), (True, 0))
        self.assertEqual(limiter.allow("client", 2), (True, 0))
        allowed, retry_after = limiter.allow("client", 2)
        self.assertFalse(allowed)
        self.assertGreater(retry_after, 0)

    def test_configured_rate_limit_uses_environment(self):
        old = os.environ.get("RAG_RATE_LIMIT_PER_MINUTE")
        try:
            os.environ["RAG_RATE_LIMIT_PER_MINUTE"] = "7"
            self.assertEqual(configured_rate_limit(False), 7)
        finally:
            if old is None:
                os.environ.pop("RAG_RATE_LIMIT_PER_MINUTE", None)
            else:
                os.environ["RAG_RATE_LIMIT_PER_MINUTE"] = old


if __name__ == "__main__":
    unittest.main()
