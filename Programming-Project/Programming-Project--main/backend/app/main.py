"""
Главный модуль FastAPI-приложения для предсказания одобрения ипотеки.
Здесь регистрируются все роуты и настраивается CORS (чтобы фронтенд мог
обращаться к бэкенду с другого порта в браузере).
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import model, predict

# Создаём экземпляр приложения с базовым описанием для Swagger UI
app = FastAPI(
    title="Mortgage Approval ML Service",
    description="Сервис для предсказания одобрения ипотечной заявки на основе ML-модели",
    version="1.0.0",
)

# CORS — разрешаем запросы с любых источников.
# В продакшене стоит указать конкретные домены, но для лабы "*" подойдёт.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключаем роутеры из отдельных модулей
app.include_router(model.router)
app.include_router(predict.router)


@app.get("/", tags=["health"])
def root():
    """Простой health-check — проверяем что сервер живой."""
    return {"status": "ok", "message": "Mortgage ML Service is running"}
