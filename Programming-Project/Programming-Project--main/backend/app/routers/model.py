"""
Роутер для загрузки ML-модели.

Endpoint POST /upload-model принимает .pkl файл с обученным пайплайном
и сохраняет его в памяти приложения через model_state.
"""

import io
import logging

import joblib
from fastapi import APIRouter, File, UploadFile, HTTPException

from app.state import model_state
from app.schemas import UploadModelResponse

# Логгер для записи событий загрузки модели
logger = logging.getLogger(__name__)

router = APIRouter(tags=["model"])


@router.post("/upload-model", response_model=UploadModelResponse)
async def upload_model(model_file: UploadFile = File(..., description="Файл модели в формате .pkl")):
    """
    Загружает обученную ML-модель (Pipeline) из pkl-файла.

    Ожидает multipart/form-data с полем 'model_file'.
    Модель должна быть sklearn Pipeline, сериализованным через joblib.
    """

    # Проверяем расширение — принимаем только pkl
    if not model_file.filename.endswith(".pkl"):
        raise HTTPException(
            status_code=400,
            detail="Неверный формат файла. Ожидается .pkl"
        )

    try:
        # Читаем байты из загруженного файла
        contents = await model_file.read()

        # joblib.load умеет работать с файлоподобным объектом (io.BytesIO)
        # Это безопаснее, чем сохранять на диск и читать обратно
        pipeline = joblib.load(io.BytesIO(contents))

        # Сохраняем модель в глобальное состояние
        model_state.load(pipeline)

        logger.info(f"Модель успешно загружена из файла: {model_file.filename}")

        return UploadModelResponse(
            status="success",
            message=f"Модель '{model_file.filename}' успешно загружена"
        )

    except Exception as e:
        # Если файл битый или не является корректным pkl — возвращаем 400
        logger.error(f"Ошибка при загрузке модели: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Не удалось загрузить модель: {str(e)}"
        )
