# Ипотечный консультант — ML-сервис для оценки одобрения ипотеки

Командный проект по курсу «Программирование Python» (III курс, РГПУ им. Герцена).

Сервис предсказывает вероятность одобрения ипотечной заявки на основе данных клиента с использованием машинного обучения.

---

## Структура проекта

```
project/
├── backend/           # FastAPI-сервер (REST API + ML-модель)
├── frontend_2/        # Веб-интерфейс (HTML/CSS/JS)
└── model/             # Код ML-части (обучение модели)
```

---

## Команда и роли

| Роль | Задачи |
|------|--------|
| ML Engineer | Обучение модели, предобработка данных, сериализация pipeline |
| Backend Developer | FastAPI-сервер, интеграция модели, обработка запросов |
| Frontend Developer | Пользовательский интерфейс, интеграция с API |

---

## Стек технологий

- **ML:** scikit-learn, XGBoost, pandas, numpy, joblib
- **Backend:** Python 3.13, FastAPI, uvicorn, pydantic
- **Frontend:** HTML, CSS, JavaScript
- **Менеджер зависимостей:** uv

---

## Быстрый старт

### 1. Установить зависимости бэкенда

```bash
cd backend
pip install fastapi "uvicorn[standard]" python-multipart scikit-learn pandas numpy joblib
```

### 2. Запустить сервер

```bash
cd backend
python main.py
```

Сервер запустится на `http://localhost:8000`

Swagger UI (документация API): `http://localhost:8000/docs`

### 3. Загрузить модель

Перейти на `http://localhost:8000/docs` → `POST /upload-model` → загрузить файл `pipeline.pkl`

### 4. Открыть интерфейс

Открыть файл `frontend_2/index.html` в браузере.

---

## API

### `GET /`
Health-check сервера.

**Ответ:**
```json
{"status": "ok", "message": "Mortgage ML Service is running"}
```

---

### `POST /upload-model`
Загрузка обученной ML-модели (sklearn Pipeline, сериализованной через joblib).

**Формат:** `multipart/form-data`, поле `model_file` — файл `.pkl`

**Ответ:**
```json
{"status": "success", "message": "Модель 'pipeline.pkl' успешно загружена"}
```

---

### `POST /predict`
Предсказание для одного или нескольких клиентов.

**Пример запроса:**
```json
{
  "person_age": 35,
  "person_gender": "male",
  "person_education": "Bachelor",
  "person_income": 60000,
  "person_emp_exp": 5,
  "person_home_ownership": "rent",
  "loan_amnt": 15000,
  "loan_intent": "homeimprovement",
  "loan_int_rate": 4.5,
  "loan_percent_income": 0.25,
  "cb_person_cred_hist_length": 8,
  "credit_score": 720,
  "previous_loan_defaults_on_file": "No"
}
```

**Ответ:**
```json
[{"loan_status": "approved", "person_age": 35, ...}]
```

---

### `POST /predict-from-csv`
Пакетное предсказание по CSV-файлу.

**Формат:** `multipart/form-data`, поле `file` — CSV-файл

**Ответ:**
```json
{
  "roc_auc": 0.8745,
  "data": [{"person_age": 22, ..., "predicted_loan_status": "approved"}, ...]
}
```

> `roc_auc` возвращается только если в CSV есть колонка `loan_status`

---

## Коды ошибок

| Код | Причина |
|-----|---------|
| 400 | Неверный формат файла |
| 422 | Некорректные значения полей (например, возраст < 18) |
| 503 | Модель не загружена |
| 500 | Внутренняя ошибка сервера |

---

## Качество модели

Лучшая модель выбирается автоматически по метрике ROC-AUC среди:
- Logistic Regression
- Random Forest
- XGBoost

ROC-AUC на тестовой выборке: **~0.87**

---

## Тесты

```bash
cd backend
pip install pytest httpx
pytest tests/ -v
```

---

## Датасет

Использован датасет `loan_data.csv` — данные о заявках на кредит с признаками клиента и историей одобрений/отказов.
