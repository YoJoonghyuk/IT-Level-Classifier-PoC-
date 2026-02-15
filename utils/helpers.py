import pandas as pd

def find_column_name(df: pd.DataFrame, keyword: str) -> str:
    """
    Осуществляет поиск названия колонки в DataFrame по ключевому слову.
    Поиск выполняется без учета регистра и частичным совпадением.

    Args:
        df: pandas.DataFrame, в котором производится поиск.
        keyword: Ключевое слово, по которому осуществляется поиск колонки.

    Returns:
        Точное название колонки (str), содержащей ключевое слово.

    Raises:
        KeyError: Если колонка, содержащая указанное ключевое слово, не найдена.
    """
    for col in df.columns:
        if keyword.lower() in str(col).lower():
            return col
    raise KeyError(f"Колонка с ключевым словом '{keyword}' не найдена в данных.")