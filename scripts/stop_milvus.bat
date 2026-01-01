@echo off
chcp 65001 >nul
echo ========================================
echo 停止 Milvus 服务
echo ========================================
echo.

cd config

docker-compose down

if %errorlevel% equ 0 (
    echo.
    echo ✓ Milvus 服务已停止
    echo.
) else (
    echo.
    echo ✗ 停止失败
    echo.
)

cd ..
pause
