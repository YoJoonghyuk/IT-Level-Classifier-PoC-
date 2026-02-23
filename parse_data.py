import argparse
import os
import sys

from src.loaders import DataLoaderHandler
from src.output import NpySaveHandler
from src.transformation import FeatureExtractionHandler

DATA_DIR = 'data'

def parse_data_pipeline(csv_path: str) -> None:
    """
    Запускает пайплайн обработки CSV-файла: загрузка -> трансформация -> сохранение .npy.

    Скрипт читает CSV, извлекает признаки, фильтрует IT-разработчиков,
    формирует целевую переменную (уровень), масштабирует числовые признаки,
    векторизует текст и сохраняет результат в .npy файлы.

    Args:
        csv_path (str): Путь к исходному CSV-файлу резюме.
    """
    print(f"\n--- Запуск пайплайна парсинга данных из '{os.path.basename(csv_path)}' ---")

    if not os.path.exists(csv_path):
        print(f"Ошибка: CSV-файл '{csv_path}' не найден.")
        sys.exit(1)

    loader = DataLoaderHandler()
    # is_training=True обучает и сохраняет трансформеры в resources/
    transformer = FeatureExtractionHandler(is_training=True)
    saver = NpySaveHandler(output_dir=DATA_DIR)

    loader.set_next(transformer).set_next(saver)

    try:
        loader.handle(csv_path)
        print(f"Парсинг успешно завершен. Данные сохранены в '{DATA_DIR}'.")
    except Exception as e:
        print(f"Ошибка во время парсинга: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HH Developer Level Predictor: Обработка CSV в формат .npy."
    )
    parser.add_argument(
        "csv_path",
        help="Путь к исходному CSV-файлу резюме HeadHunter."
    )
    args = parser.parse_args()
    parse_data_pipeline(args.csv_path)