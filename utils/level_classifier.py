import numpy as np

def extract_level_from_text(text: str) -> str:
    """
    Определяет квалификационный уровень по ключевым словам в названии должности.

    Args:
        text (str): Текст названия должности (например, 'Senior Python Developer').

    Returns:
        str: Одна из категорий: 'senior', 'junior' или 'middle' (по умолчанию).
    """
    s = str(text).lower()
    if any(kw in s for kw in ['senior', 'ведущий', 'lead', 'architect', 'главный']):
        return 'senior'
    if any(kw in s for kw in ['junior', 'младший', 'стажер', 'intern', 'trainee']):
        return 'junior'
    return 'middle'


def classify_developer_level(experience_months: float, pos_text: str) -> str:
    """
    Классифицирует разработчика по уровням (Junior/Middle/Senior) на основе
    комбинации опыта работы и ключевых слов в должности.

    Алгоритм: если в названии явно указан уровень, используется он.
    В противном случае уровень определяется по количеству месяцев опыта.

    Args:
        experience_months (float): Общий стаж работы в месяцах.
        pos_text (str): Текст названия должности для лингвистического анализа.

    Returns:
        str: Финальный уровень специалиста: 'junior', 'middle' или 'senior'.
    """
    pos_level = extract_level_from_text(pos_text)

    # Если в названии уже есть Junior или Senior — доверяем названию
    if pos_level in ['senior', 'junior']:
        return pos_level

    # Иначе классифицируем по опыту (принятая в PoC эвристика)
    if np.isnan(experience_months) or experience_months < 12:
        return 'junior'
    if 12 <= experience_months <= 36:
        return 'middle'

    return 'senior'