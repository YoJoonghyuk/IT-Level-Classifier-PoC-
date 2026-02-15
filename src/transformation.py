import pandas as pd
import numpy as np
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from .base import Handler
from utils.parsers import (extract_age, extract_experience, extract_salary,
                           extract_city, extract_position_level_keywords,
                           classify_developer_level, is_it_developer)
from utils.helpers import find_column_name


class FeatureExtractionHandler(Handler):
    """Обработчик для извлечения признаков и фильтрации IT-специалистов."""

    RESOURCES_DIR = "resources"
    VECTORIZER_PATH = os.path.join(RESOURCES_DIR, "vectorizer.pkl")
    SCALER_PATH = os.path.join(RESOURCES_DIR, "scaler.pkl")
    LEVEL_MAPPING = {'junior': 0, 'middle': 1, 'senior': 2}

    def __init__(self, is_training: bool = True):
        self.is_training = is_training
        os.makedirs(self.RESOURCES_DIR, exist_ok=True)

    def _process_text(self, text_data: pd.Series) -> np.ndarray:
        """Векторизация текста через TF-IDF."""
        if os.path.exists(self.VECTORIZER_PATH) and not self.is_training:
            vectorizer = joblib.load(self.VECTORIZER_PATH)
        else:
            vectorizer = TfidfVectorizer(max_features=500, min_df=2)
            if self.is_training:
                vectorizer.fit(text_data)
                joblib.dump(vectorizer, self.VECTORIZER_PATH)
        return vectorizer.transform(text_data).toarray()

    def _process_numeric(self, numeric_data: pd.DataFrame) -> np.ndarray:
        """Масштабирование числовых признаков."""
        if os.path.exists(self.SCALER_PATH) and not self.is_training:
            scaler = joblib.load(self.SCALER_PATH)
        else:
            scaler = StandardScaler()
            if self.is_training:
                scaler.fit(numeric_data)
                joblib.dump(scaler, self.SCALER_PATH)
        return scaler.transform(numeric_data)

    def handle(self, df: pd.DataFrame) -> dict:
        """Основной метод обработки данных."""
        col_personal = find_column_name(df, 'Пол, возраст')
        col_salary = find_column_name(df, 'ЗП')
        col_pos_seek = find_column_name(df, 'Ищет работу')
        col_city = find_column_name(df, 'Город')
        col_exp = find_column_name(df, 'Опыт')
        col_last_pos = find_column_name(df, 'должность')

        salary = df[col_salary].apply(extract_salary)
        age = df[col_personal].apply(extract_age)
        exp_months = df[col_exp].apply(extract_experience)
        is_male = df[col_personal].apply(lambda x: 1 if 'Мужчина' in str(x) else 0)
        city = df[col_city].apply(extract_city)
        pos_text = df[col_pos_seek].fillna('') + ' ' + df[col_last_pos].fillna('')
        pos_text = pos_text.str.lower()

        it_mask = pos_text.apply(is_it_developer)
        processed = pd.DataFrame({
            'salary': salary[it_mask], 'age': age[it_mask],
            'exp': exp_months[it_mask], 'is_male': is_male[it_mask],
            'city': city[it_mask], 'pos': pos_text[it_mask]
        }).dropna(subset=['salary', 'age', 'exp'])

        y_data = None
        if self.is_training:
            levels = processed.apply(
                lambda r: classify_developer_level(r['exp'], extract_position_level_keywords(r['pos'])),
                axis=1
            )
            y_data = levels.map(self.LEVEL_MAPPING).values.astype(np.int32)

        text_feats = self._process_text(processed['pos'] + " " + processed['city'])
        num_feats = self._process_numeric(processed[['salary', 'age', 'exp']])
        x_data = np.hstack([processed[['is_male']].values, num_feats, text_feats]).astype(np.float32)

        return super().handle({'x': x_data, 'y': y_data})