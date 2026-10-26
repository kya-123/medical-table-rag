import re


DEFAULT_MAX_TABLE_ROW_CHARS = 360


def row_cell_values(row) -> list[str]:
    """按底层单元格身份去重，保留独立但文本相同的值。"""
    values = []
    seen_cells = set()
    for cell in row.cells:
        identity = id(getattr(cell, "_tc", cell))
        if identity in seen_cells:
            continue
        seen_cells.add(identity)
        values.append(cell.text.strip().replace("\n", " "))
    return values


def split_table_line(
    line: str,
    max_chars: int = DEFAULT_MAX_TABLE_ROW_CHARS,
) -> list[str]:
    """把过宽表格行拆成多个带表名的片段。"""
    text = (line or "").strip()
    if len(text) <= max_chars:
        return [text] if text else []

    match = re.match(r"^(【[^】]+】)", text)
    prefix = match.group(1) if match else ""
    body = text[len(prefix):]
    separator = "；" if "；" in body else (" | " if " | " in body else "")
    pieces = body.split(separator) if separator else [body]

    result = []
    current = ""

    def flush():
        nonlocal current
        if current:
            result.append(prefix + current)
            current = ""

    for raw_piece in pieces:
        piece = raw_piece.strip() if separator else raw_piece
        if not piece:
            continue
        while len(prefix) + len(piece) > max_chars:
            flush()
            room = max(1, max_chars - len(prefix))
            result.append(prefix + piece[:room])
            piece = piece[room:]
        candidate = separator.join([current, piece]) if current else piece
        if current and len(prefix) + len(candidate) > max_chars:
            flush()
            current = piece
        else:
            current = candidate
    flush()
    return result or ([text] if text else [])
