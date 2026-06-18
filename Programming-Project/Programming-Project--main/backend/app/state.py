"""
Модуль для хранения состояния модели в памяти приложения.

Используем простой класс-синглтон вместо глобальной переменной —
так проще тестировать и читать код. Модель загружается один раз
через /upload-model и переиспользуется во всех запросах.
"""

from typing import Optional


class ModelState:
    """Хранит текущую загруженную ML-модель."""

    def __init__(self):
        # Здесь будет храниться объект Pipeline из sklearn/joblib
        self.pipeline = None

    def is_loaded(self) -> bool:
        """Проверяем, загружена ли модель."""
        return self.pipeline is not None

    def load(self, pipeline) -> None:
        """Сохраняем модель в состояние."""
        self.pipeline = pipeline

    def get(self):
        """Возвращаем модель или None."""
        return self.pipeline


# Глобальный экземпляр — создаётся один раз при старте приложения
model_state = ModelState()
