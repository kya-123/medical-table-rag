import ast
import os
import unittest


class ApiStructureTest(unittest.TestCase):
    def test_ask_calls_retrieval_once(self):
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "api.py")
        with open(path, encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=path)
        count = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name) and func.id == "_retrieve_with_assessment":
                count += 1
        self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
