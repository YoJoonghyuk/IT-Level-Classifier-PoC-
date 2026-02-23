# IT-Level Classifier (PoC)

##  Структура Репозитория

* salary_classification/
* ├── app.py                # Предсказание уровня 
* ├── parse_data.py         # CSV -> Масштабирование -> .npy
* ├── train_classifier.py    # Обучение моделей (LR, RF) и оценка
* ├── README.md             # Описание проекта и инструкции
* ├── .gitignore            # Исключения для Git
* ├── resources/            # Папка для весов .pkl и графиков
* ├── src/                  # Папка с логикой хендлеров
* │   ├── base.py           # Базовый класс Handler
* │   ├── loaders.py        # Загрузка CSV
* │   ├── output.py         # Сохранение .npy
* │   └── transformation.py # Обработка признаков (Feature Engineering)
* └── utils/                # Папка с утилитами (SRP)
*  ├── age_parser.py
*  ├── city_parser.py
*  ├── experience_parser.py
*  ├── helpers.py
*  ├── it_filter.py      # Фильтрация только IT-специалистов
*  ├── level_classifier.py # Логика разметки Junior/Middle/Senior
*  ├── salary_parser.py
*  ├── transformer_utils.py # Логика Fit/Load для Scaler/Vectorizer
*  └── visualizer.py     # Построение графиков


## Инструкция по запуску

1. **Подготовка данных**: Положите `hh.csv` в папку `data/`. Установите необходимые библиотеки, используя pip: 

`pip install pandas numpy scikit-learn joblib`
2. **Парсинг и обработка**: 

`python parse_data.py data/hh.csv`


3. **Обучение моделей**

`python train_classifier.py`


4. **Предсказание**

`python app.py data/x_data.npy
   `

