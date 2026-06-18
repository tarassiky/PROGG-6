"""
Роутеры для предсказаний.

POST /predict        — предсказание для одного или нескольких клиентов (JSON)
POST /predict-from-csv — пакетное предсказание по CSV-файлу + ROC-AUC
"""

import io
import logging
from typing import List, Union

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, UploadFile, HTTPException
from sklearn.metrics import roc_auc_score

from app.state import model_state
from app.schemas import ClientData, PredictionResult, CsvPredictionResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["predict"])

# Колонки, которые ожидает модель (до предобработки — пайплайн сам кодирует)
FEATURE_COLUMNS = [
    "person_age",
    "person_gender",
    "person_education",
    "person_income",
    "person_emp_exp",
    "person_home_ownership",
    "loan_amnt",
    "loan_intent",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score",
    "previous_loan_defaults_on_file",
]


def _check_model_loaded():
    """
    Вспомогательная функция — проверяем загружена ли модель.
    Если нет — кидаем 503 (сервис недоступен), как указано в ТЗ.
    """
    if not model_state.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Модель не загружена. Сначала загрузите модель через POST /upload-model"
        )


def _predict_dataframe(df: pd.DataFrame) -> np.ndarray:
    """
    Делаем предсказание для DataFrame с признаками клиентов.
    Возвращает массив меток (0 или 1).
    """
    pipeline = model_state.get()

    # Оставляем только нужные колонки в нужном порядке
    # (на случай если в CSV есть лишние колонки или другой порядок)
    available_cols = [col for col in FEATURE_COLUMNS if col in df.columns]
    df_features = df[available_cols]

    predictions = pipeline.predict(df_features)
    return predictions


def _label_to_str(label) -> str:
    """Конвертируем числовую метку в строку для фронтенда.
    В этой модели метки перевёрнуты: 0 = одобрено, 1 = отказано.
    Это особенность конкретного pipeline.pkl от ML-инженера.
    """
    if label == 0 or label == "0":
        return "approved"
    return "rejected"


@router.post("/predict", response_model=List[PredictionResult])
async def predict(data: Union[ClientData, List[ClientData]]):
    """
    Предсказывает вероятность одобрения ипотеки для одного или нескольких клиентов.

    Принимает JSON: один объект или массив объектов с признаками клиента.
    Возвращает массив с признаками + predicted loan_status.
    """
    _check_model_loaded()

    # Нормализуем ввод — всегда работаем со списком
    if isinstance(data, ClientData):
        records = [data]
    else:
        records = data

    if len(records) == 0:
        raise HTTPException(status_code=400, detail="Список клиентов пустой")

    try:
        # Конвертируем Pydantic-модели в DataFrame для sklearn
        df = pd.DataFrame([r.model_dump() for r in records])

        predictions = _predict_dataframe(df)

        # Формируем результат: исходные признаки + loan_status
        results = []
        for i, record in enumerate(records):
            result_dict = record.model_dump()
            result_dict["loan_status"] = _label_to_str(predictions[i])
            results.append(PredictionResult(**result_dict))

        logger.info(f"Предсказание выполнено для {len(records)} клиентов")
        return results

    except Exception as e:
        logger.error(f"Ошибка при предсказании: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при предсказании: {str(e)}")


@router.post("/predict-from-csv", response_model=CsvPredictionResponse)
async def predict_from_csv(file: UploadFile = File(..., description="CSV-файл с данными клиентов")):
    """
    Принимает CSV-файл (обычный или предобработанный) и возвращает:
    - roc_auc: метрика качества (только если в CSV есть колонка loan_status)
    - data: полный датасет с добавленной колонкой predicted_loan_status

    CSV может быть как сырым (из train_test_split), так и предобработанным.
    """
    _check_model_loaded()

    # Проверяем расширение файла
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Ожидается CSV-файл")

    try:
        contents = await file.read()
        # Пробуем читать как UTF-8, при ошибке пробуем cp1251 (Windows-кодировка)
        try:
            df = pd.read_csv(io.BytesIO(contents), encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(contents), encoding="cp1251")

        logger.info(f"CSV загружен: {df.shape[0]} строк, {df.shape[1]} колонок")

        # Проверяем наличие хоть каких-то нужных признаков
        missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if len(missing_features) == len(FEATURE_COLUMNS):
            raise HTTPException(
                status_code=400,
                detail=f"CSV не содержит ни одного ожидаемого признака. Ожидаются: {FEATURE_COLUMNS}"
            )

        # Если есть колонка loan_status — сохраняем истинные метки для ROC-AUC
        true_labels = None
        if "loan_status" in df.columns:
            true_labels = df["loan_status"].copy()

        # Делаем предсказание
        predictions = _predict_dataframe(df)

        # Считаем ROC-AUC если у нас есть истинные метки
        roc_auc = None
        if true_labels is not None:
            try:
                # Метки могут быть строками ("approved"/"rejected") или числами (0/1)
                # Приводим к числам для sklearn
                if true_labels.dtype == object:
                    # Маппинг строковых меток в числа
                    label_map = {"approved": 1, "rejected": 0, "1": 1, "0": 0, "yes": 1, "no": 0}
                    numeric_labels = true_labels.str.lower().map(label_map)
                else:
                    numeric_labels = true_labels.astype(int)

                # Проверяем что у нас не все одного класса
                if numeric_labels.nunique() > 1:
                    roc_auc = float(roc_auc_score(numeric_labels, predictions))
                    roc_auc = round(roc_auc, 4)
                    logger.info(f"ROC-AUC: {roc_auc}")
            except Exception as e:
                logger.warning(f"Не удалось вычислить ROC-AUC: {e}")

        # Добавляем предсказания в датасет
        df["predicted_loan_status"] = [_label_to_str(p) for p in predictions]

        # Конвертируем в список словарей для JSON-ответа
        # Заменяем NaN на None (JSON не поддерживает NaN)
        result_data = df.where(pd.notnull(df), None).to_dict(orient="records")

        return CsvPredictionResponse(
            roc_auc=roc_auc,
            data=result_data
        )

    except HTTPException:
        # Пробрасываем HTTP-исключения дальше без изменений
        raise
    except Exception as e:
        logger.error(f"Ошибка при обработке CSV: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке CSV: {str(e)}")
