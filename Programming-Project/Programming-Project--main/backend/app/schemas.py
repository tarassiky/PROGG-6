"""
Схемы данных (Pydantic-модели) для валидации входящих запросов и формирования ответов.

Pydantic автоматически проверяет типы и диапазоны значений — если фронтенд
отправит строку вместо числа, FastAPI вернёт 422 с понятным сообщением об ошибке.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class ClientData(BaseModel):
    """
    Данные одного клиента для предсказания.
    Поля соответствуют признакам датасета loan_data.csv до предобработки —
    ML-пайплайн сам сделает кодирование и нормализацию.
    """

    person_age: float = Field(..., ge=18, le=100, description="Возраст клиента (лет)")
    person_gender: str = Field(..., description="Пол: 'male' или 'female'")
    person_education: str = Field(..., description="Уровень образования (Bachelor, Master, PhD, ...)")
    person_income: float = Field(..., ge=0, description="Годовой доход в долларах")
    person_emp_exp: int = Field(..., ge=0, description="Стаж работы (лет)")
    person_home_ownership: str = Field(..., description="Статус жилья: rent, own, mortgage, other")
    loan_amnt: float = Field(..., ge=0, description="Запрашиваемая сумма кредита ($)")
    loan_intent: str = Field(..., description="Цель кредита (homeimprovement, debtconsolidation, ...)")
    loan_int_rate: float = Field(..., ge=0, le=100, description="Процентная ставка (%)")
    loan_percent_income: float = Field(..., ge=0, description="Доля кредита от дохода")
    cb_person_cred_hist_length: float = Field(..., ge=0, description="Длина кредитной истории (лет)")
    credit_score: int = Field(..., ge=300, le=850, description="Кредитный рейтинг (300-850)")
    previous_loan_defaults_on_file: str = Field(..., description="Были ли дефолты: 'Yes' или 'No'")

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class PredictionResult(BaseModel):
    """Результат предсказания для одного клиента."""

    loan_status: str = Field(..., description="'approved' или 'rejected'")
    # Дополнительно возвращаем исходные признаки — так удобно отображать в таблице
    person_age: Optional[float] = None
    person_gender: Optional[str] = None
    person_education: Optional[str] = None
    person_income: Optional[float] = None
    person_emp_exp: Optional[int] = None
    person_home_ownership: Optional[str] = None
    loan_amnt: Optional[float] = None
    loan_intent: Optional[str] = None
    loan_int_rate: Optional[float] = None
    loan_percent_income: Optional[float] = None
    cb_person_cred_hist_length: Optional[float] = None
    credit_score: Optional[int] = None
    previous_loan_defaults_on_file: Optional[str] = None


class CsvPredictionResponse(BaseModel):
    """Ответ для /predict-from-csv."""

    roc_auc: Optional[float] = Field(None, description="ROC-AUC (если в CSV был loan_status)")
    data: List[dict] = Field(..., description="Полный датасет с колонкой predicted_loan_status")


class UploadModelResponse(BaseModel):
    """Ответ при загрузке модели."""

    status: str
    message: str
