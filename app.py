import argparse
import os
import sys
import joblib
import numpy as np
from typing import List

MODEL_PATH = os.path.join('resources', 'classifier_rf.pkl')
MAP = {0: 'junior', 1: 'middle', 2: 'senior'}

def predict(npy_path: str) -> List[str]:
    """Предсказание уровней на основе .npy файла."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Модель не найдена. Запустите train_classifier.py")

    x_data = np.load(npy_path)
    model = joblib.load(MODEL_PATH)
    preds = model.predict(x_data)
    return [MAP[int(p)] for p in preds]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HH Level Predictor")
    parser.add_argument("path", help="Путь к x_data.npy")
    args = parser.parse_args()

    try:
        results = predict(args.path)
        print(f"Предсказано уровней: {len(results)}")
        print(results[:10], "..." if len(results) > 10 else "")
    except Exception as e:
        print(f"Ошибка: {e}")