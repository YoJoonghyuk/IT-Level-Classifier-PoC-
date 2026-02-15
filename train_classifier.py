import os
import sys
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

DATA_DIR, RES_DIR = 'data', 'resources'
X_PATH, Y_PATH = os.path.join(DATA_DIR, 'x_data.npy'), os.path.join(DATA_DIR, 'y_data.npy')
MAP = {0: 'junior', 1: 'middle', 2: 'senior'}

def plot_balance(y):
    """Сохранение графика баланса классов."""
    counts = Counter(y)
    sorted_c = sorted(counts.items())
    labels = [MAP[int(c[0])] for c in sorted_c]
    vals = [c[1] for c in sorted_c]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=labels, y=vals, palette='magma')
    plt.title("Распределение уровней IT-специалистов")
    plt.savefig(os.path.join(RES_DIR, 'class_balance.png'))
    plt.close()

def train_model(model, X_train, X_test, y_train, y_test, name, path):
    """Обучение и оценка модели."""
    print(f"Обучение {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    report = classification_report(y_test, preds, target_names=list(MAP.values()))
    print(f"\nОтчет {name}:\n{report}")
    joblib.dump(model, path)
    return report

def main():
    """Запуск процесса обучения."""
    if not os.path.exists(X_PATH):
        sys.exit("Данные не найдены. Запустите parse_data.py")

    X, y = np.load(X_PATH), np.load(Y_PATH)
    os.makedirs(RES_DIR, exist_ok=True)
    plot_balance(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    train_model(LogisticRegression(max_iter=1000, class_weight='balanced'),
                X_train, X_test, y_train, y_test, "LogReg", os.path.join(RES_DIR, 'classifier_lr.pkl'))

    train_model(RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1),
                X_train, X_test, y_train, y_test, "RandomForest", os.path.join(RES_DIR, 'classifier_rf.pkl'))

if __name__ == "__main__":
    main()