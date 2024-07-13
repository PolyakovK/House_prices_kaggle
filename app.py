import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sklearn
sklearn.set_config(transform_output="pandas")

# Явное указание функции
def create_features(df):
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['OverallGrade'] = df['OverallQual'] * df['OverallCond']
    df['AgeAtSale'] = df['YrSold'] - df['YearBuilt']
    df['TotalBathrooms'] = df['FullBath'] + df['HalfBath']*0.5 + df['BsmtFullBath'] + df['BsmtHalfBath']*0.5
    
    drop_columns = ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'OverallQual', 'OverallCond', 
                    'YearBuilt', 'YrSold', 'FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath', 'Alley', 'MiscFeature']
    df = df.drop(columns=drop_columns)
    
    return df

# Загрузка модели
ml_pipeline_CB = joblib.load('model_pipeline.pkl')

# Streamlit приложение
st.title("House Price Prediction")
st.subheader("Загрузите тестовый датафрейм для создания submission файла с предсказанными значениями")

uploaded_file = st.file_uploader("Загрузите Test.csv", type="csv")
if uploaded_file is not None:
    test = pd.read_csv(uploaded_file, index_col='Id')
    y_pred = ml_pipeline_CB.predict(test)
    y_pred = np.expm1(y_pred)
    submission_cb = pd.DataFrame({'Id': test.index, 'SalePrice': y_pred})
    
    # Разделение на две колонки для отображения
    left_column, right_column = st.columns([2, 1])
    
    # Отображение таблицы с предсказаниями в левой колонке
    with left_column:
        st.write(submission_cb)
        # Скачивание файла с предсказаниями
        csv = submission_cb.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Скачать файл CSV",
            data=csv,
            file_name='submission.csv',
            mime='text/csv'
        )
    
    # Отображение результата на Kaggle и места в рейтинге в правой колонке
    with right_column:
        st.metric(label="Результат на Kaggle", value="0.12292")
        st.metric(label="Место в рейтинге", value="427")
