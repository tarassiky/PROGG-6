# Mortgage ML Service — Backend

FastAPI-бэкенд для сервиса предсказания одобрения ипотечной заявки.

## Стек

- **Python 3.12**
- **FastAPI** — REST API
- **uvicorn** — ASGI-сервер
- **scikit-learn / joblib** — загрузка ML-модели
- **pandas** — обработка CSV
- **uv** — менеджер зависимостей и виртуального окружения
- **pytest** — тесты

## Структура проекта

```
backend/
├── app/
│   ├── main.py          # Точка входа приложения, CORS, регистрация роутеров
│   ├── state.py         # Глобальное хранилище загруженной модели
│   ├── schemas.py       # Pydantic-схемы для валидации запросов/ответов
│   └── routers/
│       ├── model.py     # POST /upload-model
│       └── predict.py   # POST /predict, POST /predict-from-csv
├── tests/
│   └── test_api.py      # Тесты всех эндпоинтов
├── models/              # Папка для хранения pkl-файлов (гитигнорится)
├── main.py              # Запуск uvicorn
├── pyproject.toml       # Зависимости и конфигурация
└── README.md
```

## Установка и запуск

```bash
# 1. Установить uv (если ещё не установлен)
pip install uv

# 2. Установить зависимости
uv sync

# 3. Запустить сервер разработки
uv run python main.py

# Или через uvicorn напрямую
uv run uvicorn app.main:app --reload --port 8000
```

Swagger UI будет доступен по адресу: http://localhost:8000/docs

## API Endpoints

### `GET /`
Health-check. Возвращает `{"status": "ok"}`.

---

### `POST /upload-model`
Загружает обученную ML-модель (sklearn Pipeline, сериализованный через joblib).

**Формат запроса:** `multipart/form-data`
- `model_file` — файл `.pkl`

**Ответ:**
```json
{"status": "success", "message": "Модель 'pipeline.pkl' успешно загружена"}
```

---

### `POST /predict`
Предсказание для одного или нескольких клиентов.

**Формат запроса:** `application/json`
```json
{
  "person_age": 35,
  "person_gender": "male",
  "person_education": "Bachelor",
  "person_income": 75000,
  "person_emp_exp": 5,
  "person_home_ownership": "rent",
  "loan_amnt": 200000,
  "loan_intent": "homeimprovement",
  "loan_int_rate": 4.5,
  "loan_percent_income": 0.27,
  "cb_person_cred_hist_length": 8,
  "credit_score": 710,
  "previous_loan_defaults_on_file": "No"
}
```
Или массив таких объектов.

**Ответ:** массив объектов с теми же полями + `loan_status: "approved" | "rejected"`

---

### `POST /predict-from-csv`
Пакетное предсказание по CSV-файлу.

**Формат запроса:** `multipart/form-data`
- `file` — CSV-файл (сырой из train_test_split или предобработанный)

**Ответ:**
```json
{
  "roc_auc": 0.9362,
  "data": [ {..., "predicted_loan_status": "approved"}, ... ]
}
```
`roc_auc` возвращается только если в CSV есть колонка `loan_status`.

---

## Коды ошибок

| Код | Причина |
|-----|---------|
| 400 | Неверный формат файла или повреждённый pkl |
| 422 | Некорректные значения полей (валидация Pydantic) |
| 503 | Модель не загружена |
| 500 | Внутренняя ошибка сервера |

## Тесты

```bash
uv run pytest tests/ -v
```
