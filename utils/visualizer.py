import os
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns

def save_class_balance_plot(y, mapping, docs_dir: str = 'docs'):
    """
    Строит и сохраняет график распределения классов.

    Args:
        y (np.ndarray): массив целевых меток.
        mapping (dict): словарь для перевода кодов в названия (0 -> 'junior').
        res_dir (str): папка для сохранения графика.
    """
    counts = Counter(y)
    sorted_items = sorted(counts.items())
    labels = [mapping[int(item[0])] for item in sorted_items]
    vals = [item[1] for item in sorted_items]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=labels, y=vals, palette='viridis')
    plt.title("Распределение уровней IT-специалистов")
    plt.ylabel("Количество резюме")

    os.makedirs(docs_dir, exist_ok=True)
    # Сохраняем именно в docs/
    save_path = os.path.join(docs_dir, 'class_balance.png')
    plt.savefig(save_path)
    plt.close()
    print(f"[Visualizer] Отчетный график сохранен в {save_path}")