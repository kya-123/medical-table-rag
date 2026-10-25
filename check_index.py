import os
import sqlite3
from config import CHROMA_DB_DIR

db_path = os.path.join(CHROMA_DB_DIR, "chroma.sqlite3")
con = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
cur = con.cursor()

def count(where_sql):
    return cur.execute(
        'SELECT COUNT(*) FROM embedding_metadata WHERE key="chroma:document" AND ' + where_sql
    ).fetchone()[0]

total = count('1=1')
tag_open = count("string_value LIKE '%[表格]%'")
tag_close = count("string_value LIKE '%[/表格]%'")
table_rows = count("string_value LIKE '【表%'")

print(f'片段总数            : {total}')
print(f'含 [表格] 标记      : {tag_open}')
print(f'含 [/表格] 标记     : {tag_close}')
print(f'以【表 开头的表格行 : {table_rows}')
con.close()
