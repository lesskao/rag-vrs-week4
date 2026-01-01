@echo off
chcp 65001 >nul
echo ========================================
echo 编辑 .env 配置文件
echo ========================================
echo.

cd /d %~dp0\..

if not exist .env (
    echo 创建新的 .env 文件...
    copy /Y nul .env >nul
)

echo 正在打开 .env 文件...
echo.
echo 请修改以下内容:
echo   DEEPSEEK_API_KEY=your_api_key_here
echo.
echo 改为:
echo   DEEPSEEK_API_KEY=sk-你的真实API密钥
echo.

notepad .env

echo.
echo ========================================
echo 保存后请运行测试脚本
echo ========================================
echo.
pause
