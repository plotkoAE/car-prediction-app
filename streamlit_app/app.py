import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import auc

#настройка страницы
st.set_page_config(
    page_title='Кредитный скоринг',
    layout='wide'
)
st.title('🎯 Кредитный скоринг: прогнозирование дефолта')


#полный список признаков для one-hot кодирования
ALL_FEATURE_NAMES = [
    'pre_since_opened', 'pre_since_confirmed', 'pre_pterm', 'pre_fterm',
    'pre_till_pclose', 'pre_till_fclose',
    'pre_loans_credit_limit', 'pre_loans_next_pay_summ', 'pre_loans_outstanding',
    'pre_loans_total_overdue', 'pre_loans_max_overdue_sum', 'pre_loans_credit_cost_rate',
    'pre_loans5', 'pre_loans530', 'pre_loans3060', 'pre_loans6090', 'pre_loans90',
    'is_zero_loans5', 'is_zero_loans530', 'is_zero_loans3060', 'is_zero_loans6090', 'is_zero_loans90',
    'pre_util', 'pre_over2limit', 'pre_maxover2limit',
    'is_zero_util', 'is_zero_over2limit', 'is_zero_maxover2limit',
    'enc_paym_0', 'enc_paym_1', 'enc_paym_2', 'enc_paym_3', 'enc_paym_4',
    'enc_paym_5', 'enc_paym_6', 'enc_paym_7', 'enc_paym_8', 'enc_paym_9',
    'enc_paym_10', 'enc_paym_11', 'enc_paym_12', 'enc_paym_13', 'enc_paym_14',
    'enc_paym_15', 'enc_paym_16', 'enc_paym_17', 'enc_paym_18', 'enc_paym_19',
    'enc_paym_20', 'enc_paym_21', 'enc_paym_22', 'enc_paym_23', 'enc_paym_24',
    'enc_loans_account_holder_type', 'enc_loans_credit_status',
    'enc_loans_credit_type', 'enc_loans_account_cur',
    'pclose_flag', 'fclose_flag'
]


#загрузка модели и данных (кеширование)
@st.cache_resource
def load_model():
    with open('streamlit_app/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_data
def load_feature_columns():
    with open('streamlit_app/feature_columns.json', 'r') as f:
        return json.load(f)

@st.cache_data
def load_eda_data():
    return pd.read_csv('streamlit_app/df_eda_sample.csv')

@st.cache_data
def load_shap_data():
    shap_values = pd.read_csv('streamlit_app/shap_values_sample.csv')
    top_features = pd.read_csv('streamlit_app/shap_top_features.csv')
    return shap_values, top_features

@st.cache_data
def load_metrics():
    return pd.read_csv('streamlit_app/model_metrics.csv')

model = load_model()
feature_columns = load_feature_columns()
df_eda = load_eda_data()
shap_values_df, shap_top = load_shap_data()
metrics_df = load_metrics()


#функция предобработки для прогноза
def preprocess_for_prediction(df_raw, feature_columns):
    """
    Применяет one-hot кодирование и агрегацию по id.
    Если какой-то колонки из feature_columns нет - добавляет её с нулями.
    """
    #one-hot кодирование
    dummies = pd.get_dummies(df_raw[ALL_FEATURE_NAMES], columns=ALL_FEATURE_NAMES)

    #группировка по id (groupby сортирует id по возрастанию - это финальный порядок строк)
    dummies['id'] = df_raw['id']
    result = dummies.groupby('id').sum().reset_index()

    #добавляем недостающие колонки с нулями
    for col in feature_columns:
        if col not in result.columns:
            result[col] = 0

    #оставляем id + только те колонки, которые нужны модели
    result = result[['id'] + feature_columns]

    return result


#горизонтальные вкладки
tab1, tab2, tab3, tab4 = st.tabs([
    '📋 Описание проекта',
    '📊 EDA',
    '📈 Модели',
    '🔮 Прогноз'
])

#вкладка описание проекта
with tab1:
    st.header('О проекте')

    st.markdown('''
    ### Актуальность темы

    **Кредитный дефолт** - это ситуация, когда заемщик перестает выполнять обязательства по кредитному договору. Для банков и МФО это означает прямые финансовые потери, увеличение резервов и нагрузку на регулятивный капитал

   ### Актуальные данные по просроченной задолженности в России

    **Общая просрочка по кредитам физлиц**  
    Достигла **1,65 трлн рублей** к началу 2026 года, что на треть больше, чем годом ранее. Доля проблемных потребительских ссуд выросла до **4,6%**, что стало максимумом за последние пять лет.  
    *Источник: [Известия - Груз — тоска: просрочка россиян по кредитам достигла рекордных 1,6 трлн](https://iz.ru/2036522/evgenii-grachev/gruz-toska-prosrochka-rossiyan-po-kreditam-dostigla-rekordnyh-1-6-trln)*

    **Просрочка по кредитным картам**  
    Доля просроченных кредитов по картам на срок **30–90 дней** в третьем квартале 2025 года снизилась до **1,6%**. Однако эксперты связывают это с улучшением процедур взыскания, а не с повышением платежной дисциплины клиентов.  
    *Источник: [Frank Media - Доля просрочки по кредитным картам вернулась к уровню прошлого года](https://frankmedia.ru/229497)*

    **Глубокая просрочка (свыше 90 дней)**  
    По состоянию на май 2025 года, **8,3 млн кредитных карт** имели просрочку более 90 дней. Совокупная задолженность по ним достигла **575,9 млрд рублей**, а доля таких карт в общем портфеле — **12%**.  
    *Источник: [Forbes — У россиян растет просрочка по кредиткам](https://www.forbes.ru/finansy/540492-u-rossian-rastet-prosrocka-po-kreditkam)*
                
    **Ключевая проблема:** традиционные скоринговые модели, основанные на логистической регрессии, имеют ограниченную предсказательную силу в условиях роста нестандартных заемщиков и новых кредитных продуктов. 
    Они не способны учитывать сложные нелинейные зависимости между признаками, например, одновременное влияние нескольких просрочек на разных этапах кредита.

    **Машинное обучение** позволяет:
    - Находить скрытые паттерны в поведении заемщиков
    - Учитывать взаимодействия между десятками признаков
    - Строить модели, которые адаптируются к изменению поведения клиентов

    **Цель проекта** - построить модель машинного обучения для прогнозирования дефолта заемщика на основе исторических данных о кредитах и платежах. На основе лучшей модели мы создали **Streamlit-приложение**, которое позволяет:
    - Исследовать данные через EDA
    - Сравнивать модели и их метрики
    - Загружать новых клиентов и получать прогноз вероятности дефолта

    **Практическая ценность:** такая система может быть встроена в процесс принятия решений по выдаче кредитов, помогая снижать долю невозвратов.
    ''')


    st.divider()

    st.subheader('Использованные источники')
    st.markdown('''
    1. [Базовые решения и полезные функции](https://github.com/SmirnovValeriy/dl-fintech-bki)
    2. [Улучшение нейросетевого бэйзлайна](https://habr.com/ru/companies/alfa/articles/551130/)
    3. [Применение нейронных сетей на многомерных временных данных](https://ods.ai/tracks/dl_in_finance)
    4. [AUC ROC (площадь под кривой ошибок)](https://alexanderdyakonov.wordpress.com/2017/07/28/auc-roc-площадь-под-кривой-ошибок/)
    5. [ВКР: Выбор моделей машинного обучения для оценки кредитоспособности](https://elar.urfu.ru/bitstream/10995/140505/1/m_th_a.v.zaitsev_2024.pdf)
    6. [Опыт моделирования вероятности кредитного дефолта](https://ej.hse.ru/data/2019/12/27/1524756384/Поляков.pdf)
    7. [Прогнозирование кредитоспособности клиентов](https://cyberleninka.ru/article/n/prognozirovanie-kreditosposobnosti-klientov-na-osnove-metodov-mashinnogo-obucheniya)
    8. [Как нейросети выдают кредиты?](https://habr.com/ru/articles/836402/)
    9. [Перспективы использования методов машинного обучения для оценки кредитных рисков](https://esj.today/PDF/70FAVN225.pdf)
    10. [Краткий курс машинного обучения для скоринга](https://habr.com/ru/articles/340792/)
    ''')


#вкладка EDA
with tab2:
    st.header('Исследовательский анализ данных')
    with st.expander('Выводы по EDA'):
        st.markdown('''
        **Ключевые инсайты:**
        - Доля дефолтов в выборке составляет ~3.2% - сильный дисбаланс классов
        - Признаки `enc_paym_*` (история платежей по месяцам) сильно коррелируют между собой, что требует агрегации
        - Признак `pre_loans_total_overdue` обладает нулевой дисперсией и не несет новой информации - надо удалить
        - Флаги `is_zero_util`, `is_zero_over2limit`, `is_zero_maxover2limit` дублируют информацию из `pre_*` признаков и могут быть удалены
        - Наиболее информативные признаки: `pre_util`, `pre_loans530`, `pre_loans_credit_cost_rate`
        ''')

    col1, col2 = st.columns(2)

    with col1:
        default_rate = df_eda['flag'].mean()
        st.subheader('Распределение целевой переменной')
        st.metric('**Доля дефолтов в выборке**', f'{default_rate:.2%}')
        fig, ax = plt.subplots(figsize=(6, 4))
        df_eda['flag'].value_counts().plot(kind='bar', ax=ax, color=["#1d78b9", "#db7012"])
        ax.set_xlabel('Дефолт')
        ax.set_ylabel('Количество')
        ax.set_xticklabels(['Нет дефолта', 'Дефолт'], rotation=0)
        st.pyplot(fig, use_container_width=False)


    with col2:
        st.subheader('Анализ признаков')
        key_features = ['rn'] + ALL_FEATURE_NAMES
        available_features = [f for f in key_features if f in df_eda.columns]

        if available_features:
            feature = st.selectbox('Выберите признак для анализа', available_features)
            fig, ax = plt.subplots(figsize=(6, 4))
            for flag in [0, 1]:
                data = df_eda[df_eda['flag'] == flag][feature]
                ax.hist(data, bins=30, alpha=0.5, label=f'Дефолт={flag}')
            ax.set_xlabel(feature)
            ax.set_ylabel('Частота')
            ax.legend()
            st.pyplot(fig, use_container_width=False)

    with st.expander('Описание признаков'):

        st.markdown('''
         
* rn	Порядковый номер кредитного продукта в кредитной истории. Большему номеру соответствует продукт с более поздней датой открытия.  
* pre_since_opened	Дней с даты открытия кредита до даты сбора данных (бинаризовано*)  
* pre_since_confirmed	Дней с даты подтверждения информации по кредиту до даты сбора данных (бинаризовано*)  
* pre_pterm	Плановое количество дней с даты открытия кредита до даты закрытия (бинаризовано*)  
* pre_fterm	Фактическое количество дней с даты открытия кредита до даты закрытия (бинаризовано*)  
* pre_till_pclose	Плановое количество дней с даты сбора данных до даты закрытия кредита (бинаризовано*)  
* pre_till_fclose	Фактическое количество дней с даты сбора данных до даты закрытия кредита (бинаризовано*)  
* pre_loans_credit_limit	Кредитный лимит (бинаризовано*)  
* pre_loans_next_pay_summ	Сумма следующего платежа по кредиту (бинаризовано*)  
* pre_loans_outstanding	Оставшаяся невыплаченная сумма кредита (бинаризовано*)  
* pre_loans_total_overdue	Текущая просроченная задолженность (бинаризовано*)  
* pre_loans_max_overdue_sum	Максимальная просроченная задолженность (бинаризовано*)  
* pre_loans_credit_cost_rate	Полная стоимость кредита (бинаризовано*)  
* pre_loans5	Число просрочек до 5 дней (бинаризовано*)  
* pre_loans530	Число просрочек от 5 до 30 дней (бинаризовано*)  
* pre_loans3060	Число просрочек от 30 до 60 дней (бинаризовано*)  
* pre_loans6090	Число просрочек от 60 до 90 дней (бинаризовано*)  
* pre_loans90	Число просрочек более, чем на 90 дней (бинаризовано*)  
* is_zero_loans5	Флаг: нет просрочек до 5 дней  
* is_zero_loans530	Флаг: нет просрочек от 5 до 30 дней  
* is_zero_loans3060	Флаг: нет просрочек от 30 до 60 дней  
* is_zero_loans6090	Флаг: нет просрочек от 60 до 90 дней  
* is_zero_loans90	Флаг: нет просрочек более, чем на 90 дней  
* pre_util	Отношение оставшейся невыплаченной суммы кредита к кредитному лимиту (бинаризовано*)  
* pre_over2limit	Отношение текущей просроченной задолженности к кредитному лимиту (бинаризовано*)  
* pre_maxover2limit	Отношенение максимальной просроченной задолженности к кредитному лимиту (бинаризовано*)  
* is_zero_util	Флаг: отношение оставшейся невыплаченной суммы кредита к кредитному лимиту равняется 0  
* is_zero_over2limit	Флаг: отношение текущей просроченной задолженности к кредитному лимиту равняется 0  
* is_zero_maxover2limit	Флаг: отношение максимальной просроченной задолженности к кредитному лимиту равняется 0  
* enc_paym_{0..N}	Статусы ежемесячных платежей за последние N месяцев (закодировано**)  
* enc_loans_account_holder_type	Тип отношения к кредиту (закодировано**)  
* enc_loans_credit_status	Статус кредита (закодировано**)  
* enc_loans_account_cur	Валюта кредита (закодировано**)  
* enc_loans_credit_type	Тип кредита (закодировано**)  
* pclose_flag	Флаг: плановое количество дней с даты открытия кредита до даты закрытия не определено  
* fclose_flag	Флаг: фактическое количество дней с даты открытия кредита до даты закрытия не определено   

`*` область значений поля разбивается на N непересекающихся промежутков, каждому промежутку случайным образом ставится в соответствие уникальный номер от 0 до N-1, значение поля заменяется номером промежутка, которому оно принадлежит  
`**` каждому уникальному значению поля случайным образом ставится в соответствие уникальный номер от 0 до K, значение поля заменяется номером этого значения  	

        ''')


    st.subheader('Корреляционная матрица всех признаков')

    df_filtered = df_eda.select_dtypes(include=[np.number])
    corr_matrix = df_filtered.corr(method='pearson')

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        corr_matrix,
        vmin=-1,
        vmax=1,
        cmap='coolwarm',
        center=0,
        linewidths=0.4,
        linecolor='white',
        annot=False,
        square=False,
        cbar_kws={'shrink': 0.8},
        ax=ax
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)


#вкладка Модели
with tab3:

    st.header('Сравнение моделей')

    st.subheader('Метрики качества')
    st.dataframe(metrics_df.style.background_gradient(subset=['ROC-AUC'], cmap='Greens'))

    with st.expander('Выводы по моделям'):
        st.markdown('''
        **Почему LightGBM (настроенный) без Feature Engineering показал лучший результат?**
        - LightGBM - градиентный бустинг на деревьях. В отличие от логистической регрессии, он не требует линейной зависимости между признаками и таргетом: 
                    модель сама находит нелинейные зависимости и взаимодействия между признаками, не требуя их явного конструирования вручную.
        - Бустинг устойчив к мультиколлинеарности и слабым по отдельности признакам: на каждом шаге выбирается наиболее информативное на данный момент разбиение, а избыточные/дублирующие признаки просто не используются,
                    то есть модели не нужно так тщательно чистить входные данные, как линейной регрессии
        - Агрегация enc_paym_0..enc_paym_24 в max_streak/count-признаки убрала информацию о порядке и временной динамике платежей, которую бустинг умел использовать напрямую через последовательность сплитов, а отбор признаков не учитывал потенциальную пользу слабых флагов в нелинейных взаимодействиях с другими переменными. 
                    Поскольку LightGBM не нуждался в этой чистке так, как нуждалась логистическая регрессия, на необработанных, более детализированных признаках он показал более высокий ROC-AUC (0.7567 против 0.7486 с FE)
        - Подбор гиперпараметров позволил найти оптимальный баланс между качеством и переобучением:
                    невысокий learning_rate в сочетании с увеличенным n_estimators дает более плавное и устойчивое обучение, чем дефолтные параметры, 
                    а ограничение max_depth/num_leaves не дает деревьям переусложняться и подстраиваться под шум в обучающей выборке
        
        **Почему выбрана метрика ROC-AUC?**
        - Классы сильно несбалансированы (дефолтов ~3.2%)
        - Accuracy в таких условиях бессмысленна - модель может предсказывать 'все не дефолт' и получить 96.8% accuracy
        - ROC-AUC оценивает способность модели **ранжировать** клиентов по риску, что критически важно для кредитного скоринга
        - AUC = 0.76 означает, что в 76% случаев модель ставит дефолтному клиенту более высокий риск, чем недефолтному
        ''')

    st.subheader('ROC-кривые')
    with open('roc_data.json', 'r') as f:
        roc_data = json.load(f)

    model_names = {
        'lr': 'Логистическая регрессия',
        'rf': 'Random Forest',
        'lgb_base': 'LightGBM (базовый)',
        'lgb_tuned': 'LightGBM (настроенный)',
        'lgb_tuned_fe': 'LightGBM (настроенный, FE)'
    }

    fig, ax = plt.subplots(figsize=(6, 4))
    for item in roc_data:
        model_key = item['model']
        fpr = np.array(item['fpr'])
        tpr = np.array(item['tpr'])
        roc_auc = auc(fpr, tpr)
        label = model_names.get(model_key, model_key)
        ax.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.3f})', linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Случайная модель (AUC = 0.5)')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    st.pyplot(fig, use_container_width=False)

    st.subheader('Важность признаков')
    st.subheader('Топ-10 признаков по SHAP-важности')

    fig, ax = plt.subplots(figsize=(6, 4))
    top10 = shap_top.head(10)
    ax.barh(top10['feature'], top10['mean_abs_shap'])
    ax.set_xlabel('Средний |SHAP|')
    ax.invert_yaxis()
    st.pyplot(fig, use_container_width=False)


#вкладка Прогноз
with tab4:
    st.header('Прогнозирование дефолта')

    st.markdown('''
    **Загрузите CSV-файл с данными клиентов в сыром виде** (как в обучающей выборке).  
    Файл должен содержать колонки: `id`, `rn`, `pre_since_opened`, `enc_paym_*` и другие признаки.
    ''')

    show_error_details = st.sidebar.checkbox('Показывать технические детали ошибок', value=False)

    uploaded_file = st.file_uploader('Выберите CSV-файл', type=['csv'])

    if uploaded_file is not None:
        try:
            #загружаем сырые данные
            df_user_raw = pd.read_csv(uploaded_file)
            st.write(f'Загружено {len(df_user_raw)} записей, {len(df_user_raw.columns)} признаков')

            #применяем предобработку (id сохраняется внутри функции,
            #порядок строк результата гарантированно соответствует порядку id)
            with st.spinner('Выполняется предобработка данных...'):
                df_user_processed = preprocess_for_prediction(df_user_raw, feature_columns)
                st.write(
                    f'После предобработки: {len(df_user_processed)} строк, '
                    f'{len(df_user_processed.columns) - 1} признаков'
                )

            #сохраняем id и убираем его перед подачей в модель
            result_ids = df_user_processed['id'].values
            X_pred = df_user_processed.drop(columns=['id'])

            #предсказание
            with st.spinner('Выполняется предсказание...'):
                predictions = model.predict_proba(X_pred)[:, 1]
                predictions_class = (predictions >= 0.50).astype(int)

            st.subheader('Результаты прогнозирования')

            result_df = pd.DataFrame({
                'id': result_ids,
                'probability': predictions,
                'predicted_class': predictions_class
            })

            csv = result_df.to_csv(index=False)
            st.download_button(
                label='📥 Скачать результаты (CSV)',
                data=csv,
                file_name='predictions.csv',
                mime='text/csv'
            )

            col1, col2, col3 = st.columns(3)
            col1.metric('Всего записей', len(result_df))
            col2.metric('Прогнозируемых дефолтов', result_df["predicted_class"].sum())
            col3.metric('Средняя вероятность', f'{result_df["probability"].mean():.2%}')

            #распределение вероятностей
            fig, ax = plt.subplots(figsize=(6, 4))
            result_df['probability'].hist(bins=50, ax=ax)
            ax.set_xlabel('Вероятность дефолта')
            ax.set_ylabel('Количество')
            st.pyplot(fig, use_container_width=False)

            #детальные результаты
            st.subheader('Детальные результаты')
            st.dataframe(result_df.head(100))

        except Exception as e:
            st.error(f'Ошибка при обработке файла: {str(e)}')
            if show_error_details:
                st.exception(e)
