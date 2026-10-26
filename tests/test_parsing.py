import unittest

from table_text import row_cell_values, split_table_line


class FakeCell:
    def __init__(self, text, identity):
        self.text = text
        self._tc = identity


class FakeRow:
    def __init__(self, cells):
        self.cells = cells


class TableParsingTest(unittest.TestCase):
    def test_equal_values_in_distinct_cells_are_preserved(self):
        row = FakeRow([
            FakeCell("空腹", object()),
            FakeCell("5.6", object()),
            FakeCell("餐后", object()),
            FakeCell("5.6", object()),
        ])
        self.assertEqual(
            row_cell_values(row),
            ["空腹", "5.6", "餐后", "5.6"],
        )

    def test_repeated_merged_cell_is_removed(self):
        merged = object()
        row = FakeRow([
            FakeCell("合并项", merged),
            FakeCell("合并项", merged),
            FakeCell("其他", object()),
        ])
        self.assertEqual(row_cell_values(row), ["合并项", "其他"])

    def test_wide_table_line_is_split(self):
        line = "【表·宽表】" + "；".join(
            "字段%d：%s" % (index, "值" * 40)
            for index in range(20)
        )
        parts = split_table_line(line, max_chars=360)
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(len(part) <= 360 for part in parts))
        self.assertTrue(all(part.startswith("【表·宽表】") for part in parts))


if __name__ == "__main__":
    unittest.main()
