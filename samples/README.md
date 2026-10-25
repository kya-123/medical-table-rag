# 示例语料

仓库不能附带医学指南原文，所以这里放了一份介绍系统自身的示例文本，用来检查解析、切片、
检索和引用能不能跑通。

这份文本不包含医学结论，通常会被切成 8 个片段。

```powershell
Copy-Item samples\sample_corpus.txt my_knowledge\
python rag_system.py
```

另开一个终端：

```powershell
python batch_upload.py
```

可以试这些问题：

- 切片长度是多少，为什么用这个长度
- 三路召回分别是什么
- 什么时候会拒答
- 文档删除需要什么参数

问“今天天气怎么样”时，系统应该拒答。

用完以后可以把示例文本删掉，换成自己的 docx、pdf 或 txt。
