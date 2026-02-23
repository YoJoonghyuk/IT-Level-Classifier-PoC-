import os

import numpy as np

from .base import Handler

class NpySaveHandler(Handler):
    """
    Обработчик, отвечающий за сохранение результирующих массивов NumPy
    (матрицы признаков X и вектора целевой переменной y) в файлы .npy.
    """
    def __init__(self, output_dir: str = 'data'):
        """
        Инициализирует NpySaveHandler.

        Args:
            output_dir: Директория, куда будут сохранены файлы x_data.npy и y_data.npy.
                        По умолчанию 'data'.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"[{self.__class__.__name__}] Файлы .npy будут сохранены в: {self.output_dir}")
    def handle(self, data: dict) -> str:
        """
        Сохраняет матрицу признаков 'x_data.npy' и вектор целевой переменной 'y_data.npy'.

        Args:
            data: Словарь, содержащий два ключа:
                  'x': numpy.ndarray - матрица признаков.
                  'y': numpy.ndarray - вектор целевой переменной.

        Returns:
            Строковое сообщение об успешном сохранении и пути к файлам.
        """
        x_path = os.path.join(self.output_dir, 'x_data.npy')
        y_path = os.path.join(self.output_dir, 'y_data.npy')

        np.save(x_path, data['x'])
        np.save(y_path, data['y'])

        message = f"[{self.__class__.__name__}] Успех! Файлы x_data.npy и y_data.npy созданы в: {self.output_dir}"
        print(message)
        return super().handle(message)