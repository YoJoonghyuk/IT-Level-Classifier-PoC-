import numpy as np

def extract_city(text: str) -> str:
    """
    Извлекает название города из текстовой строки до первой запятой.

    Args:
        text: Входная строка (например, 'Москва, готов к переезду').

    Returns:
        Название города (str) или пустую строку.
    """
    if not isinstance(text, str):
        return ""
    return text.split(',')[0].strip()