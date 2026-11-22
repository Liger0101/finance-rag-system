# embedding配置：
EMBEDDING_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "qwen3-embedding:0.6b"

# LLM 配置
LLM_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "qwen3.5:9b"

# 检索配置，包括默认top-k以及相似度标准
TOP_K = 4
SIMILARITY_THRESHOLD = 0.5

# 请求配置
EMBEDDING_TIMEOUT = 120
LLM_TIMEOUT = 180