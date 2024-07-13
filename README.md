# Прогнозирование стоимости с использованием машинного обучения
### Ссылка на проект Kaggle - https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview

Этот проект представляет собой решение задачи машинного обучения по прогнозированию стоимости недвижимости. Задача состоит в предсказании стоимости на основе набора данных, содержащего различные признаки, которые могут влиять на цену.  

## Особенности проекта:
- **Машинное обучение**: В рамках работы над проектом было использовано 5 моделей LinearRegression, Randomforest, LightGBM, XGBoost, CatBoost), тесты показали наименьшую метрику с моделью CatBoost, которая и была использована далее.
- **Оптимизация гиперпараметров**: Используется библиотека Optuna для автоматической оптимизации гиперпараметров моделей.
- **Предобработка данных**: Проект включает в себя этапы предобработки данных, такие как заполнение пропущенных значений, кодирование категориальных признаков и масштабирование признаков.
- **Оценка моделей**: Оценка качества моделей проводится с использованием кросс-валидации и метрики RMSLE.  
- **Визуализация результатов**: Используются библиотеки для визуализации данных, чтобы наглядно представить результаты анализа и обучения моделей.

## Структура проекта:
- Jupyter notebooks с кодом для исследования данных, построения моделей и оценки их качества.
- Набор train и test данных
- Сохраненная модель после обучения для последующего использования или развертывания.
- **requirements.txt**: Файл с перечислением всех зависимостей и библиотек, необходимых для воспроизведения проекта.
- **README.md**: Основной файл с описанием проекта, инструкциями по установке и использованию.
- стримлит файл

## Результаты:
- ссылка на деплой модели https://house-prices-kaggle.streamlit.app/
- результаты метрики: 0.12292, текущий результат в рейтинге: 427 

