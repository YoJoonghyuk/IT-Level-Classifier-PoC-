import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from utils.age_parser import extract_age
from utils.city_parser import extract_city
from utils.experience_parser import extract_experience
from utils.helpers import find_column_name
from utils.it_filter import is_it_developer
from utils.level_classifier import classify_developer_level
from utils.salary_parser import extract_salary
from utils.transformer_utils import get_fitted_transformer  # Наша логика здесь
from .base import Handler

class FeatureExtractionHandler(Handler):
    """Извлекает признаки и размечает уровни IT-специалистов (Jr/Mid/Sr)."""
    RES_DIR = "resources"
    LEVEL_MAP = {'junior': 0, 'middle': 1, 'senior': 2}

    def __init__(self, is_training: bool = True):
        self.is_training = is_training

    def handle(self, df: pd.DataFrame) -> dict:
        """Трансформирует данные, обучая или используя готовые веса."""
        pos_col, last_col = find_column_name(df, 'Ищет работу'), find_column_name(df, 'должность')
        data = pd.DataFrame({
            'sal': df[find_column_name(df, 'ЗП')].apply(extract_salary),
            'age': df[find_column_name(df, 'Пол, возраст')].apply(extract_age),
            'exp': df[find_column_name(df, 'Опыт')].apply(extract_experience),
            'male': df[find_column_name(df, 'Пол, возраст')].apply(lambda x: 1 if 'Мужчина' in str(x) else 0),
            'city': df[find_column_name(df, 'Город')].apply(extract_city),
            'pos': (df[pos_col].fillna('') + ' ' + df[last_col].fillna('')).str.lower()
        }).dropna(subset=['sal', 'age', 'exp'])  # Удаляем пропуски без inplace

        data = data[data['pos'].apply(is_it_developer)]

        y_data = None
        if self.is_training:
            y_data = data.apply(lambda r: self.LEVEL_MAP[classify_developer_level(r['exp'], r['pos'])], axis=1).values

        vec = get_fitted_transformer(TfidfVectorizer(max_features=500),
                                     os.path.join(self.RES_DIR, "vectorizer.pkl"),
                                     data['pos'] + " " + data['city'], self.is_training)

        scaler = get_fitted_transformer(StandardScaler(),
                                        os.path.join(self.RES_DIR, "scaler.pkl"),
                                        data[['sal', 'age', 'exp']], self.is_training)

        x_data = np.hstack([data[['male']].values, scaler.transform(data[['sal', 'age', 'exp']]),
                            vec.transform(data['pos'] + " " + data['city']).toarray()]).astype(np.float32)

        return super().handle({'x': x_data, 'y': y_data})