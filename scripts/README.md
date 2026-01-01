# Scripts 脚本目录

此目录包含用于管理 Milvus 向量数据库的辅助脚本。

## 📋 文件说明

| 文件 | 功能 | 使用场景 |
|------|------|----------|
| `start_milvus.bat` | 启动 Milvus 服务 | Windows 一键启动 |
| `stop_milvus.bat` | 停止 Milvus 服务 | Windows 一键停止 |
| `check_milvus_status.bat` | 检查服务状态 | 查看运行状态 |
| `test_chroma_simple.py` | Chroma 测试脚本 | 测试零配置方案 |
| `test_milvus_simple.py` | Milvus 快速测试 | 测试 Milvus 功能 |
| `MILVUS_GUIDE.md` | 完整使用指南 | 详细文档 |

## 🚀 快速开始

### 启动 Milvus（推荐方式）

**Windows**:
```cmd
cd scripts
start_milvus.bat
```

这将启动完整的 Milvus 服务栈（使用 docker-compose）：
- Milvus 向量数据库
- etcd 配置存储
- MinIO 对象存储

**Linux/Mac**:
```bash
cd config
docker-compose up -d
```

### 停止服务

**Windows**:
```cmd
cd scripts
stop_milvus.bat
```

**Linux/Mac**:
```bash
cd config
docker-compose down
```

### 查看状态

**Windows**:
```cmd
cd scripts
check_milvus_status.bat
```

**Linux/Mac**:
```bash
cd config
docker-compose ps
docker-compose logs
```

## 💡 测试 Chroma（零配置替代方案）

如果 Milvus 启动遇到困难，可以使用 Chroma：

```cmd
cd scripts
python test_chroma_simple.py
```

或直接在项目根目录：
```cmd
cd examples
python quick_start.py
```

## 📖 详细文档

查看完整的 Milvus 使用指南：
```
scripts/MILVUS_GUIDE.md
```

包含：
- 详细启动步骤
- 常见问题排查
- 性能优化建议
- 数据备份方法

## ⚙️ 配置文件

Milvus 的 docker-compose 配置文件位于：
```
config/docker-compose.yml
```

## 🌐 管理界面

启动后可访问：

**MinIO 控制台**:
- URL: http://localhost:9001
- 用户名: `minioadmin`
- 密码: `minioadmin`

## 🎯 常用命令

```bash
# 查看所有容器
docker ps -a

# 查看 Milvus 日志
cd config && docker-compose logs milvus-standalone

# 重启服务
cd config && docker-compose restart

# 完全清理（删除数据）
cd config && docker-compose down -v
```

## 📞 获取帮助

- 查看 `MILVUS_GUIDE.md` 获取详细说明
- 运行 `check_milvus_status.bat` 诊断问题
- 或切换到 Chroma 零配置方案

---

**祝使用愉快！** 🚀
