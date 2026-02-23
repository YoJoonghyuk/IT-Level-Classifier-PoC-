import os
import sys

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from utils.visualizer import save_class_balance_plot

DATA_DIR, RES_DIR, DOCS_DIR = 'data', 'resources', 'docs'
MAP = {0: 'junior', 1: 'middle', 2: 'senior'}

def train_and_evaluate(model, x_train, x_test, y_train, y_test, name: str):
    """Обучает модель и выводит отчет."""
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    print(f"\n--- Отчет {name} ---")
    print(classification_report(y_test, preds, target_names=list(MAP.values())))
    joblib.dump(model, os.path.join(RES_DIR, f'classifier_{name.lower()}.pkl'))

def main():
    try:
        x = np.load(os.path.join(DATA_DIR, 'x_data.npy'))
        y = np.load(os.path.join(DATA_DIR, 'y_data.npy'))

        save_class_balance_plot(y, MAP, DOCS_DIR)

        x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

        # Обучаем две модели (PoC)
        train_and_evaluate(LogisticRegression(max_iter=1000, class_weight='balanced'),
                           x_tr, x_te, y_tr, y_te, "LR")
        train_and_evaluate(RandomForestClassifier(class_weight='balanced', n_jobs=-1),
                           x_tr, x_te, y_tr, y_te, "RF")

    except FileNotFoundError:
        print("Данные не найдены. Сначала запустите parse_data.py")
        sys.exit(1)
    except Exception as e:
        print(f"Ошибка при обучении: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()