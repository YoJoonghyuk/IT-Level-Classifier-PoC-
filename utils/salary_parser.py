import re

import numpy as np

def extract_salary(text: str) -> float:
    """
    Извлекает числовое значение зарплаты и конвертирует в рубли по фиксированным курсам.
    Поддерживает RUB, USD, EUR, KZT, UAH.
    Возвращает np.nan, если зарплата не может быть извлечена.

    Args:
        text: Входная строка (например, '200 USD').

    Returns:
        Числовое значение зарплаты в рублях (float) или np.nan.
    """
    if not isinstance(text, str):
        return np.nan

    num_match = re.sub(r'[^\d]', '', text)
    if not num_match:
        return np.nan

    amount = float(num_match)
    text_upper = text.upper()

    # Фиксированные курсы для PoC
    rates = {'USD': 90.0, 'EUR': 98.0, 'KZT': 0.20, 'ГРН': 2.5, 'UAH': 2.5}
    for currency, rate in rates.items():
        if currency in text_upper:
            amount *= rate
            break

    return amount