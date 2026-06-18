import pandas as pd
import joblib

class MortgagePredictor:
    def __init__(self, model_path='models/pipeline.pkl'):
        self.model = joblib.load(model_path)
    
    def predict(self, input_data):
        """Принимает dict или list of dicts, возвращает предсказания"""
        df = pd.DataFrame(input_data)
        predictions = self.model.predict_proba(df)[:, 1]
        return predictions.tolist()
    
    def predict_with_status(self, input_data):
        """Возвращает 0/1 предсказания"""
        probs = self.predict(input_data)
        return [1 if p >= 0.5 else 0 for p in probs]

# Пример использования
if __name__ == "__main__":
    predictor = MortgagePredictor()
    
    sample = [{
        'person_age': 35,
        'person_gender': 'male',
        'person_education': 'Bachelor',
        'person_income': 75000,
        'person_emp_exp': 8,
        'person_home_ownership': 'RENT',
        'loan_amnt': 250000,
        'loan_intent': 'PERSONAL',
        'loan_int_rate': 11.14,
        'loan_percent_income': 0.33,
        'cb_person_cred_hist_length': 3,
        'credit_score': 720,
        'previous_loan_defaults_on_file': 'No'
    }]
    
    result = predictor.predict(sample)
    print(f"Вероятность одобрения: {result[0]:.2%}")