# 知识库目录

把要检索的原始文档放在这里。仓库不附带医学指南原文。

支持 `.docx`、有文字层的 `.pdf` 和 `.txt`。

启动服务后运行：

```bash
python batch_upload.py
```

也可以调用上传接口：

```powershell
curl.exe -X POST http://127.0.0.1:8001/upload `
  -H "Authorization: Bearer <RAG_ADMIN_TOKEN>" `
  -F "file=@your_doc.docx"
```

本机未配置 Token 时可以省略 `Authorization`；对外部署时必须使用管理员 Token。

如果文档里有表格，请使用 Word 的真实表格，不要用空格或制表符排版。解析器会尽量保留表名和列名。

标题建议使用 Word 的标题样式。当前版本还没有按标题层级切分。

同一份资料有多个版本时，建议在文件名中写明年份。新版确认无误后，再用 `/delete` 下线旧版。
