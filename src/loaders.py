import os

import pandas as pd

from .base import Handler

class DataLoaderHandler(Handler):
    """
    Обработчик, отвечающий за загрузку данных из CSV-файла.
    Автоматически определяет разделитель и пытается подобрать кодировку
    (utf-8, cp1251) для устойчивости к различным форматам файлов.
    """
    def handle(self, path: str) -> pd.DataFrame:
        """
        Загружает данные из CSV-файла по указанному пути.
        Пытается загрузить файл с кодировкой 'utf-8', если не удается,
        переключается на 'cp1251'. Использует engine='python' и sep=None
        для автоматического определения разделителя.

        Args:
            path: Абсолютный или относительный путь к CSV-файлу.

        Returns:
            Загруженный pandas.DataFrame.

        Raises:
            FileNotFoundError: Если файл по указанному пути не найден.
            Exception: В случае других ошибок при чтении файла.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Ошибка: Файл '{path}' не найден.")

        try:
            df = pd.read_csv(path, sep=None, engine='python', encoding='utf-8')
        except UnicodeDecodeError:
            print(f"[{self.__class__.__name__}] Не удалось загрузить с utf-8, пробуем cp1251...")
            df = pd.read_csv(path, sep=None, engine='python', encoding='cp1251')
        except Exception as e:
            raise Exception(f"Произошла ошибка при чтении файла CSV: {e}")

        print(f"[{self.__class__.__name__}] Файл '{os.path.basename(path)}' загружен. Строк: {df.shape[0]}")
        return super().handle(df)