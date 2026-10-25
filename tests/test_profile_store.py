import os
import sqlite3
import unittest
from contextlib import closing

from profile_store import (
    ProfileAccessDenied,
    delete_profile_record,
    load_profile_record,
    save_profile_record,
)


FIELDS = [
    ("age", "年龄", "int"),
    ("meds", "药物", "str"),
]


class ProfileStoreTest(unittest.TestCase):
    def setUp(self):
        test_dir = os.path.dirname(os.path.abspath(__file__))
        self.db_path = os.path.join(
            test_dir,
            "_profile_store_test_%d_%d.db" % (os.getpid(), id(self)),
        )

    def tearDown(self):
        for suffix in ("", "-wal", "-shm"):
            path = self.db_path + suffix
            if os.path.exists(path):
                os.remove(path)

    def test_new_profile_requires_returned_token(self):
        token, created, generated = save_profile_record(
            self.db_path,
            FIELDS,
            "user-a",
            {"age": 58, "meds": "二甲双胍"},
        )
        self.assertTrue(created)
        self.assertTrue(generated)
        self.assertGreaterEqual(len(token), 32)
        with self.assertRaises(ProfileAccessDenied):
            load_profile_record(self.db_path, FIELDS, "user-a")
        profile = load_profile_record(
            self.db_path,
            FIELDS,
            "user-a",
            token,
        )
        self.assertEqual(profile["age"], 58)
        self.assertNotIn("access_token_hash", profile)

    def test_wrong_token_cannot_read_update_or_delete(self):
        token, _created, _generated = save_profile_record(
            self.db_path,
            FIELDS,
            "user-b",
            {"age": 64},
        )
        self.assertTrue(token)
        with self.assertRaises(ProfileAccessDenied):
            load_profile_record(
                self.db_path,
                FIELDS,
                "user-b",
                "wrong-token",
            )
        with self.assertRaises(ProfileAccessDenied):
            save_profile_record(
                self.db_path,
                FIELDS,
                "user-b",
                {"age": 65},
                "wrong-token",
            )
        with self.assertRaises(ProfileAccessDenied):
            delete_profile_record(
                self.db_path,
                FIELDS,
                "user-b",
                "wrong-token",
            )

    def test_correct_token_can_update_and_delete(self):
        token, _created, _generated = save_profile_record(
            self.db_path,
            FIELDS,
            "user-c",
            {"age": 70},
        )
        _token, created, generated = save_profile_record(
            self.db_path,
            FIELDS,
            "user-c",
            {"age": 71},
            token,
        )
        self.assertFalse(created)
        self.assertFalse(generated)
        profile = load_profile_record(
            self.db_path,
            FIELDS,
            "user-c",
            token,
        )
        self.assertEqual(profile["age"], 71)
        self.assertTrue(
            delete_profile_record(
                self.db_path,
                FIELDS,
                "user-c",
                token,
            )
        )
        self.assertEqual(
            load_profile_record(
                self.db_path,
                FIELDS,
                "user-c",
                token,
            ),
            {},
        )

    def test_legacy_profile_without_token_is_denied_by_default(self):
        with closing(sqlite3.connect(self.db_path)) as conn:
            with conn:
                conn.execute(
                    "CREATE TABLE profiles ("
                    "user_id TEXT PRIMARY KEY, age INTEGER, meds TEXT, "
                    "updated_at TEXT)"
                )
                conn.execute(
                    "INSERT INTO profiles VALUES (?, ?, ?, ?)",
                    ("legacy", 60, "二甲双胍", "2026-10-25"),
                )
        with self.assertRaises(ProfileAccessDenied):
            load_profile_record(self.db_path, FIELDS, "legacy", "any-token")


if __name__ == "__main__":
    unittest.main()
