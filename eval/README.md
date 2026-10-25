# eval

这里放题库、评测脚本和结果快照，运行服务本身不需要这个目录。

## 文件

| 文件 | 用途 |
|---|---|
| `gold_standard.json` | 14 道有标准信息点的题 |
| `测试问题.docx` | 24 道业务题，只统计作答和拒答 |
| `eval_rag.py` | 跑 14 题，计算 Recall、Hit 和 Answer Recall |
| `eval_24.py` | 跑 24 题，记录结果和来源数量 |
| `retrieval_gate_probe.py` | 不调用生成模型，测重排、补全和 gate 三阶段 |
| `calibrate_gate.py` | 查看重排分数分布，用于标定阈值 |
| `bench_real.py` | 对比 CPU 和 GPU 的向量化、重排速度 |

## 运行

完整生成评测需要先启动服务：

```bash
python rag_system.py
```

另一个终端执行：

```bash
python eval/eval_rag.py
python eval/eval_24.py
```

如果服务设置了 `RAG_API_TOKEN`，评测脚本会自动读取同名环境变量并添加 Bearer 请求头。

`eval_rag.py` 只有在 14 题全部成功时才覆盖结果文件。部分请求失败时直接退出。

## 纯检索探针

`retrieval_gate_probe.py` 不访问 DeepSeek，但需要已构建的索引：

```powershell
$env:RAG_CHROMA_DIR = "D:\rag_index"
python eval/retrieval_gate_probe.py
```

脚本会记录重排后、表格补全后和 gate 后三个阶段。索引为空时会直接报错。

## 换题库

换成自己的领域时：

1. 按 `gold_standard.json` 的字段写新题；
2. 重新标定 `RAG_MIN_RERANK_SCORE` 和 `RAG_EVIDENCE_*`；
3. 人工检查 answer、partial、refuse 三类样本；
4. 记录索引来源和片段数。
