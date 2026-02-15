import argparse
import os
import sys

from src.loaders import DataLoaderHandler
from src.transformation import FeatureExtractionHandler
from src.output import NpySaveHandler

DATA_DIR = 'data'

def parse_data_pipeline(csv_path: str) -> None:
    """
    Запускает пайплайн обработки CSV-файла: загрузка -> трансформация -> сохранение .npy.

    Этот скрипт читает исходный CSV-файл, извлекает необходимые признаки (пол, возраст,
    опыт, город, должность, зарплата), фильтрует IT-разработчиков, формирует целевую
    переменную (уровень специалиста), масштабирует числовые признаки, векторизует
    текстовые и сохраняет результат в 'data/x_data.npy' (признаки) и 'data/y_data.npy'
    (целевые значения). Также будут сохранены обученные 'scaler.pkl' и 'vectorizer.pkl'
    в папку 'resources/'.

    Args:
        csv_path: Путь к исходному CSV-файлу.
    """
    print(f"\n--- Запуск пайплайна парсинга данных из '{os.path.basename(csv_path)}' ---")

    # Проверка существования CSV-файла
    if not os.path.exists(csv_path):
        print(f"Ошибка: CSV-файл '{csv_path}' не найден. Пожалуйста, укажите корректный путь.")
        sys.exit(1)

    loader = DataLoaderHandler()
    # Передаем is_training=True, чтобы трансформаторы были обучены и сохранены
    transformer = FeatureExtractionHandler(is_training=True)
    saver = NpySaveHandler(output_dir=DATA_DIR)

    # Создание цепочки обработчиков
    loader.set_next(transformer).set_next(saver)

    try:
        # Запуск обработки
        loader.handle(csv_path)
        print(f"Парсинг успешно завершен. Данные сохранены в '{DATA_DIR}'.")
    except Exception as e:
        print(f"Ошибка во время парсинга: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HH Developer Level Predictor: Обработка исходного CSV-файла в формат .npy."
    )
    parser.add_argument(
        "csv_path",
        help="Путь к исходному CSV-файлу резюме HeadHunter (например, 'data/hh.csv')."
    )
    args = parser.parse_args()
    parse_data_pipeline(args.csv_path)