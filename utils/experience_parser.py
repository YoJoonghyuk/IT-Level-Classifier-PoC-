import re

def extract_experience(text: str) -> int:
    """
    Извлекает опыт работы из текстовой строки и конвертирует его в общее количество месяцев.
    Парсит числа перед словами 'год/лет/года' и 'месяц/месяца/месяцев'.
    Включает эвристику для фильтрации дат (например, '2019 год' вместо '2019 лет').
    Возвращает 0, если опыт не указан или не может быть извлечен.

    Args:
        text: Входная строка с информацией об опыте работы (например, '5 лет 2 месяца').

    Returns:
        Общее количество месяцев опыта (int).
    """
    s = str(text).lower()
    if s == 'nan' or 'не указано' in s:
        return 0

    years_match = re.search(r'(\d+)\s+(?:год|года|лет)', s)
    months_match = re.search(r'(\d+)\s+(?:месяц|месяца|месяцев)', s)

    years = int(years_match.group(1)) if years_match else 0
    months = int(months_match.group(1)) if months_match else 0

    # Эвристика: если число лет выглядит как год (например, 2019), сбрасываем его
    if years > 100:
        years = 0

    return years * 12 + months