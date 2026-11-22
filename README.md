# Finance RAG System

一个基于 Python 和 Ollama 构建的本地 RAG（Retrieval-Augmented Generation，检索增强生成）系统，支持文档切分、向量化、语义检索和大语言模型问答。

## 1. 项目介绍

本项目通过 RAG 技术，让大语言模型能够基于给定文档进行问答，减少模型脱离参考资料生成答案的情况。

项目采用模块化设计，将文本处理、Embedding、检索、Prompt 构建和 LLM 调用拆分为独立模块，并通过 Pipeline 串联完整流程。

## 2. 技术栈

- Python 3.12
- NumPy
- Requests
- Pytest
- Ollama
- Qwen3 Embedding 0.6B
- Qwen3.5 9B

## 3. 项目结构

```text
RAG/
├── src/
│   ├── config.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── retrieval.py
│   ├── prompting.py
│   ├── llm.py
│   └── pipeline.py
├── tests/
│   ├── test_chunking.py
│   ├── test_embeddings.py
│   ├── test_retrieval.py
│   ├── test_prompting.py
│   ├── test_llm.py
│   ├── test_pipeline.py
│   ├── test_embeddings_integration.py
│   └── test_llm_integration.py
├── main.py
├── .gitignore
└── README.md
```

## 4. 核心工作流程

1. 文本切分：将原始文档切分为适合检索的文本块。
2. 向量化：调用 Ollama Embedding 模型生成文本向量。
3. 语义检索：通过余弦相似度计算相关性，筛选 Top-K 参考资料。
4. Prompt 构建：将检索结果与用户问题组合为提示词。
5. LLM 生成：调用 Ollama 大语言模型生成最终回答。

## 5. 环境要求

- Python 3.12+
- Ollama
- 本地下载以下模型：

```bash
ollama pull qwen3-embedding:0.6b
ollama pull qwen3.5:9b
```

## 6. 安装与运行

安装依赖：

```bash
pip install requests numpy pytest
```

确认 Ollama 服务正常运行，并检查 `src/config.py` 中的模型名称、API 地址和超时配置。

运行项目：

```bash
python main.py
```

## 7. 自动化测试

运行全部测试：

```bash
python -m pytest tests/ -v
```

运行 Mock 单元测试与真实模型集成测试时，可分别指定测试文件。

```bash
python -m pytest tests/test_llm.py -v
python -m pytest tests/test_embeddings.py -v
python -m pytest tests/test_llm_integration.py -v
python -m pytest tests/test_embeddings_integration.py -v
```

项目已完成基础模块化及自动化测试，Mock 测试与本地 Ollama 模型集成测试均已通过。

## 8. 后续规划

- 优化文本切分策略
- 建立检索效果评估体系
- 引入 BM25 混合检索
- 引入 Reranker 重排序
- 支持 PDF、DOCX 等文档解析
- 使用 FastAPI 构建后端接口
- 完善日志、配置管理与部署能力

## 9. 项目定位

本项目是一个持续迭代的 RAG 工程实践项目，重点关注检索质量、模块化设计、自动化测试和后端工程化能力。