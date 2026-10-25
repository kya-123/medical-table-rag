import ast
import io
import tokenize
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def strip_hash_comments(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    spans = {}
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type != tokenize.COMMENT:
                continue
            if token.start[0] == 1 and token.string.startswith("#!"):
                continue
            spans.setdefault(token.start[0], []).append(
                (token.start[1], token.end[1])
            )
    except tokenize.TokenError:
        return False
    if not spans:
        return False
    lines = source.splitlines(keepends=True)
    for line_number, ranges in spans.items():
        index = line_number - 1
        line = lines[index]
        newline = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
        body = line[:-len(newline)] if newline else line
        for start, end in reversed(ranges):
            body = body[:start] + body[end:]
        lines[index] = body.rstrip() + newline if body.strip() else newline
    cleaned = "".join(lines)
    path.write_text(cleaned, encoding="utf-8", newline="")
    return cleaned != source


def strip_docstrings(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    ranges = []
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if not isinstance(first, ast.Expr):
            continue
        value = first.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            ranges.append((first.lineno, first.end_lineno))
    if not ranges:
        return False
    lines = source.splitlines(keepends=True)
    for start, end in sorted(set(ranges), reverse=True):
        del lines[start - 1:end]
    cleaned = "".join(lines)
    path.write_text(cleaned, encoding="utf-8", newline="")
    return cleaned != source


def main() -> int:
    changed = 0
    for path in ROOT.rglob("*.py"):
        if any(part in {".git", ".venv", "venv", "model", "chroma_db"} for part in path.parts):
            continue
        changed_here = strip_hash_comments(path)
        changed_here = strip_docstrings(path) or changed_here
        if changed_here:
            changed += 1
    print(f"stripped {changed} python files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
