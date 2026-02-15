import re
import numpy as np

def extract_age(text: str) -> float:
    """Извлекает возраст из строки."""
    if not isinstance(text, str):
        return np.nan
    match = re.search(r'(\d+)\s+(?:год|года|лет)', text, re.IGNORECASE)
    return float(match.group(1)) if match else np.nan

def extract_experience(text: str) -> float:
    """Извлекает опыт работы в месяцах."""
    s = str(text).lower()
    if 'не указано' in s or s == 'nan' or not isinstance(text, str):
        return np.nan
    years_match = re.search(r'(\d+)\s+(?:год|года|лет)', s)
    months_match = re.search(r'(\d+)\s+(?:месяц|месяца|месяцев)', s)
    years = int(years_match.group(1)) if years_match else 0
    months = int(months_match.group(1)) if months_match else 0
    if years > 100:
        years = 0
    return float(years * 12 + months)

def extract_salary(text: str) -> float:
    """Извлекает зарплату и конвертирует в рубли."""
    if not isinstance(text, str):
        return np.nan
    num_match = re.sub(r'[^\d]', '', text)
    if not num_match:
        return np.nan
    amount = float(num_match)
    text_upper = text.upper()
    if 'USD' in text_upper:
        amount *= 90.0
    elif 'EUR' in text_upper:
        amount *= 98.0
    elif 'KZT' in text_upper:
        amount *= 0.20
    elif 'ГРН' in text_upper or 'UAH' in text_upper:
        amount *= 2.5
    return amount

def extract_city(text: str) -> str:
    """Извлекает название города."""
    if not isinstance(text, str):
        return ""
    return text.split(',')[0].strip()

def is_it_developer(position_text: str) -> bool:
    """Проверяет, относится ли должность к IT-разработке."""
    if not isinstance(position_text, str):
        return False
    s = position_text.lower()
    it_keywords = [
        'разработчик', 'developer', 'программист', 'engineer', 'инженер',
        'backend', 'frontend', 'fullstack', 'qa', 'тестировщик', 'devops',
        'python', 'java', 'c++', 'javascript', 'php', 'go', 'data scientist'
    ]
    return any(kw in s for kw in it_keywords)

def extract_position_level_keywords(text: str) -> str:
    """Определяет уровень по ключевым словам в должности."""
    if not isinstance(text, str):
        return 'unknown'
    s = text.lower()
    if any(kw in s for kw in ['senior', 'ведущий', 'lead', 'главный', 'architect', 'архитектор']):
        return 'senior'
    if any(kw in s for kw in ['junior', 'младший', 'стажер', 'intern', 'trainee']):
        return 'junior'
    return 'middle'

def classify_developer_level(experience_months: float, pos_level: str) -> str:
    """Классифицирует уровень на основе опыта и ключевых слов."""
    if pos_level in ['senior', 'junior']:
        return pos_level
    if np.isnan(experience_months):
        return 'junior'
    if experience_months < 12:
        return 'junior'
    elif 12 <= experience_months <= 36:
        return 'middle'
    return 'senior'