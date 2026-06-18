"""
Тесты для FastAPI-приложения.

Используем TestClient из fastapi.testclient — он запускает приложение
«вживую» без реального сетевого соединения, что удобно для юнит-тестов.

Запуск:
    uv run pytest tests/ -v
"""

import io
import pickle
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.state import model_state

# Создаём тестовый клиент один раз для всех тестов
client = TestClient(app)


# ========== Фиктивная модель для тестов ==========

class FakePipeline:
    """
    Заглушка sklearn-пайплайна для тестов.
    Всегда предсказывает 1 (одобрено) — нам не важна точность в тестах,
    важно что API корректно обрабатывает данные.
    """
    def predict(self, X):
        import numpy as np
        return np.ones(len(X), dtype=int)


def make_fake_pkl() -> bytes:
    """Сериализуем фиктивную модель в байты."""
    buf = io.BytesIO()
    pickle.dump(FakePipeline(), buf)
    return buf.getvalue()


VALID_CLIENT = {
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
    "previous_loan_defaults_on_file": "No",
}


# ========== Фикстура ==========

@pytest.fixture(autouse=True)
def reset_model_state():
    """
    Перед каждым тестом сбрасываем состояние модели.
    Это нужно чтобы тесты не зависели от порядка запуска.
    """
    model_state.pipeline = None
    yield
    model_state.pipeline = None


# ========== Тесты health-check ==========

def test_root_returns_ok():
    """GET / должен вернуть статус ok."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ========== Тесты /upload-model ==========

def test_upload_model_success():
    """Загружаем валидный pkl — ожидаем status=success."""
    pkl_bytes = make_fake_pkl()
    resp = client.post(
        "/upload-model",
        files={"model_file": ("pipeline.pkl", pkl_bytes, "application/octet-stream")},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"


def test_upload_model_wrong_extension():
    """Загружаем .txt файл — ожидаем 400."""
    resp = client.post(
        "/upload-model",
        files={"model_file": ("model.txt", b"not a model", "text/plain")},
    )
    assert resp.status_code == 400


def test_upload_model_broken_pkl():
    """Загружаем битый pkl — ожидаем 400."""
    resp = client.post(
        "/upload-model",
        files={"model_file": ("model.pkl", b"this is not valid pickle data", "application/octet-stream")},
    )
    assert resp.status_code == 400


# ========== Тесты /predict ==========

def test_predict_without_model():
    """Предсказание без загруженной модели — ожидаем 503."""
    resp = client.post("/predict", json=VALID_CLIENT)
    assert resp.status_code == 503


def test_predict_single_client():
    """Предсказание для одного клиента — ожидаем массив с одним результатом."""
    model_state.load(FakePipeline())
    resp = client.post("/predict", json=VALID_CLIENT)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["loan_status"] in ("approved", "rejected")


def test_predict_multiple_clients():
    """Предсказание для списка клиентов."""
    model_state.load(FakePipeline())
    resp = client.post("/predict", json=[VALID_CLIENT, VALID_CLIENT])
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2


def test_predict_invalid_age():
    """Возраст < 18 — должен вернуть 422 (ошибка валидации)."""
    bad_client = {**VALID_CLIENT, "person_age": 10}
    model_state.load(FakePipeline())
    resp = client.post("/predict", json=bad_client)
    assert resp.status_code == 422


# ========== Тесты /predict-from-csv ==========

def make_csv_bytes(include_loan_status: bool = True) -> bytes:
    """Генерируем тестовый CSV с несколькими строками."""
    header = "person_age,person_gender,person_education,person_income,person_emp_exp," \
             "person_home_ownership,loan_amnt,loan_intent,loan_int_rate," \
             "loan_percent_income,cb_person_cred_hist_length,credit_score," \
             "previous_loan_defaults_on_file"
    if include_loan_status:
        header += ",loan_status"

    rows = []
    for i in range(3):
        row = f"35,male,Bachelor,75000,5,rent,200000,homeimprovement,4.5,0.27,8,710,No"
        if include_loan_status:
            row += f",{i % 2}"  # чередуем 0 и 1
        rows.append(row)

    csv_content = header + "\n" + "\n".join(rows)
    return csv_content.encode("utf-8")


def test_predict_csv_without_model():
    """CSV-предсказание без модели — ожидаем 503."""
    csv_bytes = make_csv_bytes()
    resp = client.post(
        "/predict-from-csv",
        files={"file": ("test.csv", csv_bytes, "text/csv")},
    )
    assert resp.status_code == 503


def test_predict_csv_with_loan_status():
    """CSV с loan_status — ожидаем roc_auc в ответе."""
    model_state.load(FakePipeline())
    csv_bytes = make_csv_bytes(include_loan_status=True)
    resp = client.post(
        "/predict-from-csv",
        files={"file": ("test.csv", csv_bytes, "text/csv")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "data" in data
    assert len(data["data"]) == 3
    # ROC-AUC может быть None если все метки одинакового класса
    # В нашем тесте 3 строки: 0, 1, 0 — так что должен считаться
    assert "roc_auc" in data


def test_predict_csv_without_loan_status():
    """CSV без loan_status — roc_auc должен быть None."""
    model_state.load(FakePipeline())
    csv_bytes = make_csv_bytes(include_loan_status=False)
    resp = client.post(
        "/predict-from-csv",
        files={"file": ("test.csv", csv_bytes, "text/csv")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["roc_auc"] is None


def test_predict_csv_wrong_extension():
    """Загружаем .txt вместо .csv — ожидаем 400."""
    model_state.load(FakePipeline())
    resp = client.post(
        "/predict-from-csv",
        files={"file": ("data.txt", b"not,a,csv", "text/plain")},
    )
    assert resp.status_code == 400


def test_predicted_status_column_in_response():
    """Убеждаемся что в ответе есть колонка predicted_loan_status."""
    model_state.load(FakePipeline())
    csv_bytes = make_csv_bytes()
    resp = client.post(
        "/predict-from-csv",
        files={"file": ("test.csv", csv_bytes, "text/csv")},
    )
    assert resp.status_code == 200
    first_row = resp.json()["data"][0]
    assert "predicted_loan_status" in first_row
    assert first_row["predicted_loan_status"] in ("approved", "rejected")
