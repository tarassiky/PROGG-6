import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def clean_data(df):
    """
    Очистка данных от выбросов
    """
    df_clean = df.copy()
    
    # 1. Возраст (нормальный диапазон 18-100)
    df_clean = df_clean[(df_clean['person_age'] >= 18) & (df_clean['person_age'] <= 100)]
    
    # 2. Стаж работы (не может быть больше возраста - 16)
    df_clean = df_clean[df_clean['person_emp_exp'] <= df_clean['person_age'] - 16]
    df_clean = df_clean[df_clean['person_emp_exp'] >= 0]
    
    # 3. Процентная ставка (не может быть 0 или отрицательной)
    df_clean = df_clean[df_clean['loan_int_rate'] > 0]
    
    # 4. Кредитный рейтинг (нормальный диапазон 300-850)
    df_clean = df_clean[(df_clean['credit_score'] >= 300) & (df_clean['credit_score'] <= 850)]
    
    # 5. Доход (не может быть отрицательным)
    df_clean = df_clean[df_clean['person_income'] > 0]
    
    # 6. Сумма кредита (не может быть отрицательной)
    df_clean = df_clean[df_clean['loan_amnt'] > 0]
    
    print(f"После очистки: {len(df_clean)} строк (удалено {len(df) - len(df_clean)})")
    
    return df_clean


def create_preprocessing_pipeline():
    """
    Создает предобработчик для данных
    """
    # Числовые признаки
    numeric_features = [
        'person_age', 'person_income', 'person_emp_exp', 'loan_amnt',
        'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
        'credit_score'
    ]
    
    # Категориальные признаки
    categorical_features = [
        'person_gender', 'person_education', 'person_home_ownership', 'loan_intent'
    ]
    
    # Трансформеры
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    
    # Объединяем
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    return preprocessor, numeric_features, categorical_features


def encode_target(df):
    """
    Кодирует целевые признаки
    """
    df_encoded = df.copy()
    
    # Преобразуем Yes/No в 1/0
    df_encoded['previous_defaults'] = df_encoded['previous_loan_defaults_on_file'].map({'Yes': 1, 'No': 0})
    
    # Удаляем старую колонку
    df_encoded = df_encoded.drop('previous_loan_defaults_on_file', axis=1)
    
    return df_encoded


def get_feature_columns():
    """
    Возвращает список всех признаков для модели
    """
    return [
        'person_age',
        'person_gender',
        'person_education',
        'person_income',
        'person_emp_exp',
        'person_home_ownership',
        'loan_amnt',
        'loan_intent',
        'loan_int_rate',
        'loan_percent_income',
        'cb_person_cred_hist_length',
        'credit_score',
        'previous_defaults'
    ]