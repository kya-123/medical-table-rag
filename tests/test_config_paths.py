
import os
import unittest
from unittest.mock import patch

import config


class ChromaPathTest(unittest.TestCase):
    def test_space_path_uses_safe_fallback(self):
        with patch.object(config, "BASE_DIR", r"C:\Users\John Doe\RAG Project"), patch.dict(
            os.environ,
            {"LOCALAPPDATA": r"C:\Users\John Doe\AppData\Local", "TEMP": r"C:\Temp"},
            clear=False,
        ):
            result = config._default_chroma_dir()
        self.assertNotIn(" ", result)
        self.assertIn("MedRAG", result)

    def test_ascii_path_stays_in_project(self):
        with patch.object(config, "BASE_DIR", r"C:\work\rag"), patch.dict(
            os.environ,
            {"LOCALAPPDATA": r"C:\Users\John Doe\AppData\Local"},
            clear=False,
        ):
            result = config._default_chroma_dir()
        self.assertEqual(result, r"C:\work\rag\chroma_db")

    def test_both_fallbacks_unsafe_uses_system_drive(self):
        with patch.object(config, "BASE_DIR", r"C:\Users\John Doe\RAG Project"), patch.dict(
            os.environ,
            {
                "LOCALAPPDATA": r"C:\Users\John Doe\AppData\Local",
                "TEMP": r"C:\Users\John Doe\AppData\Local\Temp",
                "SystemDrive": "C:",
            },
            clear=False,
        ):
            result = config._default_chroma_dir()
        self.assertEqual(result, r"C:\MedRAG\chroma_db")
        self.assertNotIn(" ", result)


if __name__ == "__main__":
    unittest.main()
