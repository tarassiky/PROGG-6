import pandas as pd
import joblib
import json
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline  # ← ЭТОТ ИМПОРТ БЫЛ ПРОПУЩЕН
from data_preprocessing import clean_data, create_preprocessing_pipeline, encode_target, get_feature_columns


def train_model():
    """
    Обучает модели и сохраняет лучшую
    """
    # Загрузка данных
    print("1. Загрузка данных...")
    df = pd.read_csv('data/loan_data.csv')
    print(f"   Загружено {len(df)} строк")
    
    # Очистка
    print("\n2. Очистка данных...")
    df_clean = clean_data(df)
    
    # Кодирование
    print("\n3. Кодирование категориальных признаков...")
    df_encoded = encode_target(df_clean)
    
    # Подготовка X и y
    feature_cols = get_feature_columns()
    X = df_encoded[feature_cols]
    y = df_encoded['loan_status']
    
    print(f"\n4. Размерность X: {X.shape}")
    print(f"   Доля одобрений: {y.mean():.2%}")
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n5. Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Создание pipeline предобработки
    preprocessor, numeric_features, categorical_features = create_preprocessing_pipeline()
    
    # Обучаем модели
    print("\n6. Обучение моделей...")
    results = {}
    
    # Logistic Regression
    print("\n   → Logistic Regression...")
    lr_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    lr_pipeline.fit(X_train, y_train)
    y_pred_lr = lr_pipeline.predict_proba(X_test)[:, 1]
    lr_auc = roc_auc_score(y_test, y_pred_lr)
    results['LogisticRegression'] = {'pipeline': lr_pipeline, 'auc': lr_auc}
    print(f"      AUC: {lr_auc:.4f}")
    
    # Random Forest
    print("\n   → Random Forest...")
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    rf_pipeline.fit(X_train, y_train)
    y_pred_rf = rf_pipeline.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test, y_pred_rf)
    results['RandomForest'] = {'pipeline': rf_pipeline, 'auc': rf_auc}
    print(f"      AUC: {rf_auc:.4f}")
    
    # XGBoost
    print("\n   → XGBoost...")
    xgb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss'))
    ])
    xgb_pipeline.fit(X_train, y_train)
    y_pred_xgb = xgb_pipeline.predict_proba(X_test)[:, 1]
    xgb_auc = roc_auc_score(y_test, y_pred_xgb)
    results['XGBoost'] = {'pipeline': xgb_pipeline, 'auc': xgb_auc}
    print(f"      AUC: {xgb_auc:.4f}")
    
    # Выбор лучшей модели
    print("\n7. Выбор лучшей модели...")
    best_model_name = max(results, key=lambda x: results[x]['auc'])
    best_model = results[best_model_name]['pipeline']
    best_auc = results[best_model_name]['auc']
    
    print(f"\n   ✅ Лучшая модель: {best_model_name}")
    print(f"   ✅ ROC-AUC на тесте: {best_auc:.4f}")
    
    # Кросс-валидация
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='roc_auc')
    print(f"\n   Cross-validation AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Сохранение модели
    print("\n8. Сохранение модели...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/pipeline.pkl')
    print("   ✅ Модель сохранена в models/pipeline.pkl")
    
    # Сохранение метаданных
    metadata = {
        'model_type': best_model_name,
        'auc_score': float(best_auc),
        'cv_auc_mean': float(cv_scores.mean()),
        'cv_auc_std': float(cv_scores.std()),
        'features': feature_cols,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'approval_rate': float(y.mean()),
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    os.makedirs('config', exist_ok=True)
    with open('config/features_config.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("   ✅ Метаданные сохранены в config/features_config.json")
    
    return best_model


if __name__ == "__main__":
    train_model()