@echo off
chcp 65001 >nul
echo ========================================
echo 启动 Milvus 完整服务（docker-compose）
echo ========================================
echo.

cd config

echo 正在启动 Milvus 及其依赖服务...
echo   - etcd (配置中心)
echo   - minio (对象存储)
echo   - milvus (向量数据库)
echo.

docker-compose up -d

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo ✓ Milvus 服务启动成功！
    echo ========================================
    echo.
    echo 等待服务完全就绪（约 30 秒）...
    timeout /t 30 /nobreak >nul
    echo.
    
    echo 服务状态:
    docker-compose ps
    
    echo.
    echo ========================================
    echo 连接信息:
    echo   Milvus: localhost:19530
    echo   MinIO 控制台: http://localhost:9001
    echo     用户名: minioadmin
    echo     密码: minioadmin
    echo ========================================
    echo.
    echo 管理命令:
    echo   查看日志: docker-compose logs -f
    echo   停止服务: docker-compose down
    echo   重启服务: docker-compose restart
    echo.
) else (
    echo.
    echo ✗ 启动失败！
    echo.
    echo 可能原因:
    echo   1. docker-compose 未安装
    echo   2. 端口被占用 (19530, 9091, 9000, 9001)
    echo   3. Docker Desktop 未运行
    echo.
)

cd ..
pause
