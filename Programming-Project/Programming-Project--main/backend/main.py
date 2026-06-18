"""
Точка входа для запуска сервера.

Запуск:
    uv run python main.py
    или
    uv run uvicorn app.main:app --reload --port 8000
"""

import uvicorn

if __name__ == "__main__":
    # reload=True — сервер перезапускается при изменении файлов (удобно при разработке)
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
