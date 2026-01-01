# 配置文件目录

## 设置环境变量

### 快速开始（无需配置）

基础功能（向量检索、混合检索）**无需任何配置**即可使用！

### 高级功能配置（可选）

如需使用 Multi-Query、HyDE、重排序等高级功能：

1. 复制示例配置文件到项目根目录：
```bash
cp config/.env.example .env
```

2. 编辑 `.env` 文件，取消注释并填写 API Key：
```bash
# 取消下面这行的注释，并填入你的 API Key
DEEPSEEK_API_KEY=sk-your-api-key-here

# 其他配置使用默认值即可
MILVUS_HOST=localhost
QDRANT_HOST=localhost
```

## 配置说明

详细的配置说明请参考：[../docs/CONFIGURATION_GUIDE.md](../docs/CONFIGURATION_GUIDE.md)
