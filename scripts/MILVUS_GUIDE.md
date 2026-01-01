# Milvus Docker Compose 使用指南

## 🚀 快速启动

### Windows 用户

**启动服务**（推荐）：
```cmd
start_milvus.bat
```

**或手动运行**：
```cmd
cd config
docker-compose up -d
```

### Linux/Mac 用户

```bash
cd config
docker-compose up -d
```

---

## 📋 服务说明

启动后将运行 **3 个容器**：

| 容器名 | 服务 | 端口 | 说明 |
|--------|------|------|------|
| milvus-standalone | Milvus | 19530, 9091 | 向量数据库主服务 |
| milvus-etcd | etcd | 2379 | 配置和元数据存储 |
| milvus-minio | MinIO | 9000, 9001 | 对象存储（向量数据） |

---

## 🎯 常用命令

### 启动服务
```cmd
# Windows
start_milvus.bat

# Linux/Mac
cd config && docker-compose up -d
```

### 停止服务
```cmd
# Windows
stop_milvus.bat

# Linux/Mac  
cd config && docker-compose down
```

### 查看状态
```cmd
# Windows
check_milvus_status.bat

# Linux/Mac
cd config && docker-compose ps
```

### 查看日志
```cmd
cd config

# 查看所有服务日志
docker-compose logs

# 实时跟踪日志
docker-compose logs -f

# 只看 Milvus 日志
docker-compose logs milvus-standalone

# 最近 50 行
docker-compose logs --tail=50
```

### 重启服务
```cmd
cd config
docker-compose restart
```

### 完全清理（删除数据）
```cmd
cd config
docker-compose down -v
```
⚠️ **警告**：这会删除所有 Milvus 数据！

---

## 🔍 验证安装

### 方法 1：检查容器状态
```cmd
cd config
docker-compose ps
```

应该看到 3 个容器都是 "Up" 状态。

### 方法 2：测试连接
创建 `test_milvus_connection.py`：

```python
from pymilvus import connections

try:
    connections.connect(
        alias="default",
        host="localhost",
        port="19530"
    )
    print("✓ Milvus 连接成功！")
    connections.disconnect()
except Exception as e:
    print(f"✗ 连接失败: {e}")
```

运行：
```bash
python test_milvus_connection.py
```

### 方法 3：使用 RAG 系统测试
```python
from src.vectorstores import MilvusVectorStore
from src.rag_engine import AdvancedRAGEngine
from src.core.models import Document

# 创建向量数据库
vector_store = MilvusVectorStore("test_collection")
vector_store.create_collection(dimension=768)

# 创建 RAG 引擎
rag = AdvancedRAGEngine(vector_store)

# 测试文档
docs = [
    Document(id="1", content="测试内容", metadata={})
]

rag.index_documents(docs)
results = rag.search("测试", top_k=1)

print(f"✓ 成功！检索到 {len(results)} 个结果")
```

---

## 🌐 Web 控制台

### MinIO 控制台
- **地址**: http://localhost:9001
- **用户名**: `minioadmin`
- **密码**: `minioadmin`

可以查看 Milvus 存储的向量数据文件。

---

## ⚠️ 常见问题

### 1. 端口被占用
**错误**: `port is already allocated`

**解决**:
```cmd
# 检查占用的端口
netstat -ano | findstr "19530"
netstat -ano | findstr "9091"
netstat -ano | findstr "9000"
netstat -ano | findstr "9001"

# 停止占用端口的程序，或修改 docker-compose.yml 中的端口映射
```

### 2. 服务启动失败
**查看日志**:
```cmd
cd config
docker-compose logs standalone
```

**常见原因**:
- Docker Desktop 内存不足（建议至少 4GB）
- 磁盘空间不足
- 防火墙阻止

### 3. 容器反复重启
**检查健康状态**:
```cmd
cd config
docker-compose ps
```

如果看到 "Restarting"，查看详细日志：
```cmd
docker-compose logs standalone
```

### 4. 数据持久化
数据存储在 `config/volumes/` 目录：
```
config/
├── volumes/
│   ├── etcd/      # etcd 数据
│   ├── minio/     # MinIO 数据（向量文件）
│   └── milvus/    # Milvus 数据
```

**备份数据**：直接复制 `volumes/` 目录

---

## 🎓 推荐工作流

### 开发阶段
```bash
# 1. 启动服务
start_milvus.bat

# 2. 等待 30 秒服务就绪

# 3. 运行你的代码
python your_rag_script.py

# 4. 开发完成后停止（可选）
stop_milvus.bat
```

### 测试阶段
```bash
# 每次测试前清理数据
cd config
docker-compose down -v
docker-compose up -d

# 运行测试
python test_script.py
```

---

## 📊 性能优化

### 增加 Docker 资源
1. 打开 Docker Desktop
2. Settings → Resources
3. 设置:
   - **Memory**: 至少 4GB（推荐 8GB）
   - **CPU**: 至少 2 核心
   - **Disk**: 至少 20GB

### 调整 Milvus 配置
编辑 `docker-compose.yml`，在 `standalone` 服务下添加环境变量：

```yaml
environment:
  ETCD_ENDPOINTS: etcd:2379
  MINIO_ADDRESS: minio:9000
  MILVUS_LOG_LEVEL: info  # 调整日志级别
```

---

## 🔄 升级 Milvus

### 更新到最新版本
```cmd
cd config

# 停止服务
docker-compose down

# 拉取最新镜像
docker-compose pull

# 重新启动
docker-compose up -d
```

### 指定版本
编辑 `docker-compose.yml`：
```yaml
standalone:
  image: milvusdb/milvus:v2.4.0  # 改为需要的版本
```

---

## 💡 提示

### vs. 单容器启动
| 特性 | docker-compose | 单容器 docker run |
|------|---------------|------------------|
| 稳定性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 功能完整 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 配置难度 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 启动速度 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 推荐场景 | 生产、开发 | 快速测试 |

**结论**: docker-compose 更稳定，推荐使用！

---

## 📞 获取帮助

遇到问题？

1. **查看日志**: `docker-compose logs`
2. **检查状态**: `docker-compose ps`  
3. **查看官方文档**: https://milvus.io/docs
4. **或使用 Chroma**: 零配置替代方案

---

**祝你使用愉快！** 🚀
