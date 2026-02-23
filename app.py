import argparse
import os
import sys
from typing import List

import joblib
import numpy as np

MODEL_PATH = os.path.join('resources', 'classifier_rf.pkl')
LEVEL_MAP = {0: 'junior', 1: 'middle', 2: 'senior'}

def predict_levels(npy_path: str) -> List[str]:
    """
    Выполняет предсказание уровней квалификации.

    Args:
        npy_path (str): путь к файлу с признаками.
    Returns:
        List[str]: список предсказанных уровней.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Модель {MODEL_PATH} не найдена. Обучите её через train_classifier.py")

    try:
        x_data = np.load(npy_path)
        model = joblib.load(MODEL_PATH)
        preds = model.predict(x_data)
        return [LEVEL_MAP[int(p)] for p in preds]
    except Exception as e:
        raise RuntimeError(f"Ошибка в процессе инференса: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HH Developer Level Predictor")
    parser.add_argument("path", help="Путь к x_data.npy")
    args = parser.parse_args()

    try:
        results = predict_levels(args.path)
        print(f"Обработано образцов: {len(results)}")
        print("Результаты (первые 10):", results[:10])
    except (FileNotFoundError, RuntimeError) as err:
        print(f"Ошибка: {err}")
        sys.exit(1)