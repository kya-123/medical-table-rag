# Medical Table RAG

这是一个面向中文文档的本地 RAG 示例，医疗资料是主要演示场景。项目用 FastAPI 提供接口，
检索在本地完成，最后由 DeepSeek 根据检索结果生成答案。

这个仓库适合用来了解 Word 表格处理、混合检索、重排、引用和拒答的实现方式。
它是参考项目，不是通用 RAG 平台，也不是医疗器械。涉及诊断、用药和剂量时，请以医生意见为准。

## 处理流程

```text
问题
  ├─ 多轮问题改写
  ├─ 向量检索 + BM25 + 关键词检索
  ├─ RRF 融合
  ├─ bge-reranker-base 重排
  ├─ 证据充分性判定
  └─ DeepSeek 生成答案和引用
```

## 功能

- Word 表格逐行转成“列名：值”，并保留表名；
- 命中表格某一行时，优先补回同一文档中的整张表；
- 向量检索和 BM25 混合召回，再通过 RRF 合并；
- 使用 Cross-Encoder 对候选片段重排；
- 回答中带【资料N】引用，接口返回 `sources`；
- 资料不足时返回 `partial` 或 `refuse`，不强行补全；
- 支持多轮追问改写、文件上传、删除和用户档案；
- 向量模型和重排模型在本地运行，只有生成阶段访问 DeepSeek。

## 目录

| 路径 | 内容 |
|---|---|
| `parsing.py` | 文档解析、表格处理、切片 |
| `store.py` | ChromaDB、BM25、重排模型和档案数据库 |
| `retrieval.py` | 召回、融合、重排和表格补全 |
| `evidence_sufficiency.py` | 意图、分面覆盖率和三态判定 |
| `generation.py` | 模型调用和多轮改写 |
| `api.py` | FastAPI 路由 |
| `eval/` | 题库和评测脚本 |
| `samples/` | 可公开的示例语料 |
| `tests/` | 自动化测试 |

## 快速开始

建议 Python 3.12 以上，内存至少 8 GB。

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

下载本地模型：

```bash
python scripts/download_models.py
```

默认使用 `hf-mirror.com`。下载后的模型约 2.3 GB，放在 `model/models/`。

配置 DeepSeek Key：

```powershell
$env:DEEPSEEK_API_KEY = "sk-your-key"
```

也可以复制 `config.example.txt` 为 `config.txt` 后填写。环境变量优先于文件。
没有 Key 时，服务仍可启动，但 `/ask` 会返回 503。

启动：

```bash
python rag_system.py
```

默认地址是 `http://127.0.0.1:8001`，接口文档是 `http://127.0.0.1:8001/docs`。

导入文档：

```bash
python batch_upload.py
```

可以把 docx、pdf、txt 放进 `my_knowledge/`。仓库没有附带医学指南原文。
想先验证流程，可以复制示例语料：

```powershell
Copy-Item samples\sample_corpus.txt my_knowledge\
python batch_upload.py
```

## ChromaDB 路径

当前 ChromaDB 版本在中文或空格路径下可能无法正常持久化。项目会自动尝试安全目录，
也可以手动指定：

```powershell
$env:RAG_CHROMA_DIR = "D:\rag_index"
```

## 接口

| 方法 | 路径 | 用途 |
|---|---|---|
| POST | `/ask` | 提问 |
| POST | `/upload` | 上传文档 |
| GET | `/sources` | 查看来源和片段数 |
| POST | `/delete` | 预览或删除文档 |
| POST | `/profile` | 保存档案 |
| GET | `/profile/{user_id}` | 读取档案 |
| DELETE | `/profile/{user_id}` | 删除档案 |
| GET | `/profiles` | 列出档案，默认关闭 |
| GET | `/health` | 健康检查 |

`/ask` 的返回中包含 `answer`、`context`、`sources` 和 `decision`。

`decision` 有三种值：

- `answer`：资料足够；
- `partial`：只能回答一部分；
- `refuse`：资料不足，不生成推测性答案。

## 常用环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `RAG_PORT` | `8001` | 端口 |
| `RAG_HOST` | `127.0.0.1` | 监听地址 |
| `RAG_API_TOKEN` | 空 | Bearer Token |
| `RAG_PROFILE` | `generic` | 预设身份 |
| `RAG_DOMAIN` | `医疗健康` | 领域名称 |
| `RAG_ASSISTANT_NAME` | 自动生成 | 助手名称 |
| `RAG_CHROMA_DIR` | 自动选择 | 索引目录 |
| `RAG_RERANK_POOL` | `40` | 重排候选数 |
| `RAG_MIN_RERANK_SCORE` | `0.3` | 最低整体意图分 |

## 评测

历史 14 题快照：Recall@5 58.0%，Hit@5 52.4%，Answer Recall 70.0%。
这组数据来自旧版本和旧索引。

恢复 21 份资料后，纯检索探针的结果为：

| 阶段 | Recall@5 | Hit@5 |
|---|---:|---:|
| 重排后 | 54.9% | 52.4% |
| 表格补全后 | 56.1% | 52.4% |
| gate 后 | 52.6% | 45.2% |

三态结果为 `10 answer / 3 partial / 1 refuse`。

gate 后的下降主要来自 Q7 被正确拒答：CGM 校准资料不能回答血糖仪校准周期。
DeepSeek 在当前环境连接失败，新一轮 Answer Recall 还没有完成。

纯检索探针：

```powershell
$env:RAG_CHROMA_DIR = "D:\rag_index"
python eval/retrieval_gate_probe.py
```

## 已知限制

- 只支持有文字层的 PDF，扫描件需要先 OCR。
- 没有按 Word 标题层级切片。
- BM25 索引在内存中构建，数据量很大时需要专用搜索服务。
- ChromaDB 是本地单进程使用，不适合多进程同时写。
- 多药问题 Q3 只召回了部分药物资料，目前降级为 `partial`。
- Q5 当前判为 `answer`，但 Context Recall 只有 0.167，需要人工核对最终回答。
- 档案是本地明文 SQLite，不能直接存真实患者数据。
- `/upload` 和 `/delete` 没有用户隔离，外网部署前必须补权限。
- 评测集较小，指标适合版本对比，不适合作为公开榜单。

## 检查

```bash
python -m unittest discover -s tests -v
python scripts/check_secrets.py
```

## License

MIT
