import os

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb
from chromadb.config import Settings
from config import CHROMA_DB_DIR


client = chromadb.PersistentClient(
    path=CHROMA_DB_DIR,
    settings=Settings(anonymized_telemetry=False),
)
collection = client.get_collection('knowledge_base')


data = collection.get()

print(f"📚 知识库共有 {len(data['documents'])} 个片断\n")
print("=" * 60)


for i, doc in enumerate(data['documents']):
    doc_id = data['ids'][i] if i < len(data['ids']) else "未知ID"
    print(f"【片断 {i+1}】ID: {doc_id}")
    print(f"内容: {doc}")
    print("-" * 60)


print("\n📁 按文件分组统计：")
file_stats = {}
for metadata in data.get("metadatas") or []:
    file_name = (metadata or {}).get("source", "(未知来源)")
    file_stats[file_name] = file_stats.get(file_name, 0) + 1

for file_name, count in file_stats.items():
    print(f"  {file_name}: {count} 个片断")
