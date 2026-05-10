import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os


# функции предобработки данных для входящего csv

def clean_str(string, symbol):
    """забирает первое вхождение в строке"""
    return str(string).split()[0] if pd.notna(string) else string

def get_rpm(x):
    """извлекает обороты из torque"""
    text = str(x).replace(',', '').lower()
    numbers = re.findall(r'(\d+)\s*rpm', text)
    if numbers:
        return max(map(int, numbers))
    match = re.search(r'[@at/]\s*(\d+(?:[-\s]\d+)*)', text)
    if match:
        numbers = re.findall(r'\d+', match.group(1))
        if numbers:
            return max(map(int, numbers))
    return None

def get_torque(x):
    """извлекает значение крутящего момента и переводит kgm в Nm"""
    text = str(x).replace(',', '').lower()
    torque_match = re.search(r'(\d+\.?\d*)', text)
    torque = float(torque_match.group(1)) if torque_match else None
    if torque and 'kgm' in text:
        torque = torque * 9.81
    return torque

def preprocess_csv(df):
    """предобрабатывает загруженный CSV так же, как train"""
    df = df.copy()
    
    # 1. очистка mileage, engine, max_power
    if 'mileage' in df.columns:
        df['mileage'] = df['mileage'].apply(lambda x: clean_str(x, 'kmpl'))
        df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')
    
    if 'engine' in df.columns:
        df['engine'] = df['engine'].apply(lambda x: clean_str(x, 'CC'))
        df['engine'] = pd.to_numeric(df['engine'], errors='coerce')
    
    if 'max_power' in df.columns:
        df['max_power'] = df['max_power'].apply(lambda x: clean_str(x, 'bhp'))
        df['max_power'] = pd.to_numeric(df['max_power'], errors='coerce')
    
    # 2. разбор torque на torque и max_torque_rpm
    if 'torque' in df.columns:
        df['max_torque_rpm'] = df['torque'].apply(get_rpm)
        df['torque'] = df['torque'].apply(get_torque)
    
    # 3. приведение seats к числовому
    if 'seats' in df.columns:
        df['seats'] = pd.to_numeric(df['seats'], errors='coerce')
    
    # 4. заполнение пропусков медианой (из train, но здесь используем медиану по данным)
    continuous_cols = ['mileage', 'engine', 'max_power', 'seats', 'torque', 'max_torque_rpm']
    for col in continuous_cols:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    
    # 5. приведение типов
    if 'engine' in df.columns:
        df['engine'] = df['engine'].astype(int)
    if 'seats' in df.columns:
        df['seats'] = df['seats'].astype(int)
    
    return df

st.set_page_config(page_title='Предсказание цены авто', layout='wide')
st.title('Предсказание стоимости автомобиля')

# путь к файлам
current_dir = os.path.dirname(os.path.abspath(__file__))
pkl_path = os.path.join(current_dir, 'car_pipeline.pkl')
csv_path = os.path.join(current_dir, 'train_cleared.csv')
html_path = os.path.join(current_dir, 'eda_report_train.html')

# загрузка модели, скейлера и списка признаков из одного pickle
@st.cache_resource
def load_pipeline():
    with open(pkl_path, 'rb') as f:
        pipeline = pickle.load(f)
    return pipeline['model'], pipeline['scaler'], pipeline['feature_names']


# загрузка данных для визуализаций
@st.cache_data
def load_data():
    df = pd.read_csv(csv_path)
    return df


model, scaler, numeric_features = load_pipeline()
df_train = load_data()


# боковая панель - ввод данных
st.sidebar.header('Ввод данных')

input_method = st.sidebar.radio(
    'Выберите способ ввода:',
    ['Ручной ввод', 'Загрузить CSV файл']
)


# получение предсказания
def predict_price(features_df):
    features_scaled = scaler.transform(features_df)
    prediction = model.predict(features_scaled)
    return prediction[0]


# ручной ввод
if input_method == 'Ручной ввод':
    st.sidebar.subheader('Параметры автомобиля:')

    year = st.sidebar.number_input('Год выпуска', min_value=1983, max_value=2020, value=2015)
    km_driven = st.sidebar.number_input('Пробег', min_value=0, max_value=2500000, value=50000)
    mileage = st.sidebar.number_input('Расход топлива', min_value=0.0, max_value=42.0, value=19.0)
    engine = st.sidebar.number_input('Объём двигателя', min_value=500, max_value=3600, value=1248)
    max_power = st.sidebar.number_input('Мощность', min_value=30, max_value=400, value=82)
    torque = st.sidebar.number_input('Крутящий момент', min_value=47, max_value=400, value=160)
    seats = st.sidebar.number_input('Количество мест', min_value=2, max_value=14, value=5)
    max_torque_rpm = st.sidebar.number_input('Максимальные обороты крутящего момента',
                                             min_value=500, max_value=5000, value=3000)

    if st.sidebar.button('Предсказать цену'):
        input_data = pd.DataFrame([[
            year, km_driven, mileage, engine, max_power, torque, seats, max_torque_rpm
        ]], columns=numeric_features)

        price = predict_price(input_data)

        st.sidebar.success(f'Предсказанная цена: {price:.0f}')

# загрузка CSV
elif input_method == 'Загрузить CSV файл':
    uploaded_file = st.sidebar.file_uploader('Загрузите CSV файл', type=['csv'])

    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file)
        
        # предобработка загруженного CSV
        df_input = preprocess_csv(df_input)
        
        # проверка, что все нужные колонки есть
        missing_cols = set(numeric_features) - set(df_input.columns)
        if missing_cols:
            st.sidebar.error(f'Отсутствуют колонки: {missing_cols}')
            st.stop()
        
        df_input = df_input[numeric_features]

        if st.sidebar.button('Предсказать цены'):
            df_scaled = scaler.transform(df_input)
            predictions = model.predict(df_scaled)

            df_input['predicted_price'] = predictions

            st.subheader('Результаты предсказаний')
            df_input['predicted_price'] = df_input['predicted_price'].astype(int)
            st.dataframe(df_input)

            csv = df_input.to_csv(index=False)
            st.download_button(
                label='Скачать результаты (CSV)',
                data=csv,
                file_name='predictions.csv',
                mime='text/csv'
            )

# основная область - визуализации
st.header('Анализ данных и визуализации')

# вкладки
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    'Важность признаков',
    'Корреляции',
    'Распределения',
    'О модели',
    'Виджет ProfileReport'
])

# вкладка 1: важность признаков
with tab1:
    st.subheader('Коэффициенты модели')

    coef = model.coef_

    coef_df = pd.DataFrame({
        'Признак': numeric_features,
        'Коэффициент': coef,
        'Абс. значение': np.abs(coef)
    }).sort_values('Абс. значение', ascending=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ['green' if c > 0 else 'red' for c in coef_df['Коэффициент']]
    bars = ax.barh(coef_df['Признак'], coef_df['Коэффициент'], color=colors)
    ax.tick_params(axis='x', labelrotation=45)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Коэффициент')
    ax.set_title('Влияние признаков на цену')

    for bar, val in zip(bars, coef_df['Коэффициент']):
        ax.text(bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                f'{val:,.0f}', va='center', fontsize=4)

    st.pyplot(fig)

    st.markdown('''
    Зелёный цвет - признак увеличивает цену.
    Красный цвет - признак уменьшает цену.
    Самый важный признак: max_power (мощность)
    ''')

# вкладка 2: корреляции
with tab2:
    st.subheader('Матрица корреляции (по данным тренировочной выборки)')

    # числовые колонки
    numeric_cols_for_corr = ['year', 'selling_price', 'km_driven', 'mileage',
                             'engine', 'max_power', 'torque', 'seats']
    # считаем корреляцию
    correlation_matrix = df_train[numeric_cols_for_corr].corr()
    fig, ax = plt.subplots(figsize=(6, 4))

    sns.heatmap(correlation_matrix,
                annot=True,
                cmap='coolwarm'
                )
    plt.title('Тепловая карта корреляций Пирсона')

    st.pyplot(fig)

    st.markdown('''
    **Интерпретация:**
    - Красный цвет (близко к 1) - сильная положительная связь
    - Синий цвет (близко к -1) - сильная отрицательная связь
    - Белый цвет (близко к 0) - связи нет

    **Наблюдения:**
    - max_power сильно коррелирует с selling_price (0.69) и engine (0.68)
    - km_driven имеет отрицательную корреляцию с year (-0.37)
    - max_torque_rpm слабо связан с целевой selling_price
    ''')

# вкладка 3: распределения
with tab3:
    st.subheader('Распределения признаков')

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Цена
    sns.histplot(df_train['selling_price'].dropna(), bins=30, kde=True,
                 color='lightblue', ax=axes[0, 0], edgecolor=None, alpha=0.6)
    axes[0, 0].axvline(df_train['selling_price'].median(), color='red', linestyle='-', linewidth=1, label='Медиана')
    axes[0, 0].axvline(df_train['selling_price'].mean(), color='green', linestyle='-', linewidth=1, label='Среднее')
    axes[0, 0].set_title('Распределение цены', fontsize=10)
    axes[0, 0].legend()

    # Пробег
    sns.histplot(df_train['km_driven'].dropna(), bins=30, kde=True,
                 color='orange', ax=axes[0, 1], edgecolor=None, alpha=0.6)
    axes[0, 1].axvline(df_train['km_driven'].median(), color='red', linestyle='-', linewidth=1, label='Медиана')
    axes[0, 1].axvline(df_train['km_driven'].mean(), color='green', linestyle='-', linewidth=1, label='Среднее')
    axes[0, 1].set_title('Распределение пробега', fontsize=12)
    axes[0, 1].legend()

    # Год выпуска
    sns.histplot(df_train['year'].dropna(), bins=30, kde=True,
                 color='green', ax=axes[1, 0], edgecolor=None, alpha=0.6)
    axes[1, 0].axvline(df_train['year'].median(), color='red', linestyle='-', linewidth=1, label='Медиана')
    axes[1, 0].axvline(df_train['year'].mean(), color='green', linestyle='-', linewidth=1, label='Среднее')
    axes[1, 0].set_title('Распределение года выпуска', fontsize=12)
    axes[1, 0].legend()

    # Мощность
    sns.histplot(df_train['max_power'].dropna(), bins=30, kde=True,
                 color='purple', ax=axes[1, 1], edgecolor=None, alpha=0.6)
    axes[1, 1].axvline(df_train['max_power'].median(), color='red', linestyle='-', linewidth=1, label='Медиана')
    axes[1, 1].axvline(df_train['max_power'].mean(), color='green', linestyle='-', linewidth=1, label='Среднее')
    axes[1, 1].set_title('Распределение мощности', fontsize=12)
    axes[1, 1].legend()

    plt.tight_layout()
    st.pyplot(fig)
    st.markdown('''
    **Как читать графики:**
    - Голубая/оранжевая/зелёная/фиолетовая линия — оценка плотности (KDE), показывает форму распределения
    - Красная линия — медиана
    - Зелёная линия — среднее значение

    **Выводы:**
    - Цена и пробег имеют правый хвост (есть дорогие автомобили и машины с большим пробегом)
    - Год выпуска имеет левый хвост (редкие старые автомобили)
    - Распределение мощности близко к симметричному, но с правым хвостом
    ''')

# вкладка 4: о модели
with tab4:
    st.subheader('Информация о модели')

    st.markdown('''
    **Тип модели:** Линейная регрессия

    **Признаки:**
    - year - год выпуска
    - km_driven - пробег
    - mileage - расход топлива
    - engine - объём двигателя
    - max_power - мощность
    - torque - крутящий момент
    - seats - количество мест
    - max_torque_rpm - обороты макс. крутящего момента

    **Качество модели на тесте:**
    - R2 = 0.60
    - MSE = 2.3 * 10^11

    **Выводы:**
    - Самый важный признак - мощность
    - Пробег снижает цену
    - Качество можно улучшить, добавив категориальные признаки (тип топлива, коробка передач)
    ''')

# вкладка 5: ProfileReport
with tab5:
    st.subheader('Подробный EDA отчёт (ydata-profiling)')

    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    components.html(html_content, height=800, scrolling=True)

st.markdown('---')
st.caption('Модель обучена на данных об автомобилях')