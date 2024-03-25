import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import lightgbm as lgbm
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score
import optuna
import matplotlib.pyplot as plt
import seaborn as sns


def get_session_state():
    return st.session_state

def empty_pandas_table_creation():
    data = {
        'ext_debt_stocks': 0,
        'inflation_rate': 0,
        'unemployment_rate': 0,
        'mortality_by_1000': 0,
        'resources_rent_procent_of_GDP': 0,
        'military_expenditure_procent_of_GDP': 0,
        'arms_imports': 0,
        'democracy_index': 0,
        'corruption_index': 0,
        'GPD_per_capita': 0,
        'fertility_rate': 0,
        'energy_access_procent_of_population': 0,
        'population': 0,
        'unseat_current_state_head_attempt': 0,
        'trade_openness': 0,
        'state_control_over_territory': 0,
        'political_violence': 0
    }

    table = pd.DataFrame(data, index=[0])

    return table

def load_model(file_path):
    with open(file_path, 'rb') as model_file:
        model = pickle.load(model_file)
    return model

def scale_features(table):

    table['ext_debt_stocks'] = np.log(table['ext_debt_stocks'])
    table['inflation_rate'] = np.log1p(table['inflation_rate'])
    table['unemployment_rate'] = np.log1p(table['unemployment_rate'])
    table['mortality_by_1000'] = np.log1p(table['mortality_by_1000'])
    table['resources_rent_procent_of_GDP'] = np.log1p(table['resources_rent_procent_of_GDP'])
    table['military_expenditure_procent_of_GDP'] = np.log1p(table['military_expenditure_procent_of_GDP'])
    table['arms_imports'] = np.log(table['arms_imports']) ** 2
    table['corruption_index'] = table['corruption_index'] ** 2
    table['GPD_per_capita'] = np.log(table['GPD_per_capita'])
    table['state_control_over_territory'] = np.log1p(table['state_control_over_territory'])
    table['political_violence'] = (table['political_violence'] ** 2) ** 0.5
    return table

def user_input_features(plain_table):
    ext_debt_stocks = st.sidebar.slider('Внешний валовый долг, долл.', 1.1,
                                        600000000000.0, step=1000000.0)


    inflation_rate = st.sidebar.slider('Инфляция, %', 1.1, 200.0, step=1.0)
    unemployment_rate = st.sidebar.slider('Уровень безработицы, %', 0, 100, step=1)
    mortality_by_1000 = st.sidebar.slider('Смертность, на 1000 чел.', 0.0, 110.0, step=0.1)
    resources_rent_procent_of_GDP = st.sidebar.slider('Природная рента, % от ВВП', 0.0, 100.0, step=0.1)
    military_expenditure_procent_of_GDP = st.sidebar.slider('Военные расходы, % от ВВП',
                                                            0.0, 100.0, step=0.1)
    arms_imports = st.sidebar.slider('Импорт ВВСТ, долл.', 0, 5500000000, step=1000000)
    if arms_imports == 0:
        arms_imports = 1

    democracy_index = st.sidebar.slider('Демократичность власти', 0.0, 1.0, step=0.01)
    corruption_index = st.sidebar.slider('Индекс коррупции', 0.0, 1.0, step=0.01)

    GPD_per_capita = st.sidebar.slider('ВВП на душу населения', 1.0, 1500000.0, step=1000.0)
    fertility_rate = st.sidebar.slider('Детская смертность', 0.0, 20.0, step=0.1)
    energy_access_procent_of_population = st.sidebar.slider('Доступ населения к энергоресурсам, % населения',
                                                            0.0, 100.0, step=0.01)

    circle_html_for_population = """
        <div style="display: flex; align-items: center;">
          <div style="flex: 1;"></div>
          <div style="position: relative; margin-right: 320px; margin-bottom: 0px;">
            <div style="position: absolute; width: 20px; height: 20px; border-radius: 50%; background-color: blue; color: white; text-align: center; line-height: 20px; cursor: pointer;" title="Для выставления значения показателя 'Население' разделите реальное значение на 10000">i</div>
          </div>
        </div>
    """
    # Вывод кружка и ползунка
    st.sidebar.write(circle_html_for_population, unsafe_allow_html=True)

    population = st.sidebar.slider('Население', 0, 152368, step=10)

    unseat_current_state_head_attempt = st.sidebar.selectbox('Попытка свержения действующей власти',
                                                             ('Происходила', 'Не происходила'))
    if unseat_current_state_head_attempt == 'Происходила':
        unseat_current_state_head_attempt = 1
    else:
        unseat_current_state_head_attempt = 0
    trade_openness = st.sidebar.slider('Открытость торговли', 0.0, 10.0, step=0.01)

    state_control_over_territory = st.sidebar.slider('Контроль власти над территорией страны',
                                                     0.0, 10.0, step=0.01)

    political_violence = st.sidebar.slider('Уровень политических репрессий', 0.0, 10.0, step=0.01)

    plain_table['ext_debt_stocks'] = ext_debt_stocks
    plain_table['inflation_rate'] = inflation_rate
    plain_table['unemployment_rate'] = unemployment_rate
    plain_table['mortality_by_1000'] = mortality_by_1000
    plain_table['resources_rent_procent_of_GDP'] = resources_rent_procent_of_GDP
    plain_table['military_expenditure_procent_of_GDP'] = military_expenditure_procent_of_GDP
    plain_table['arms_imports'] = arms_imports
    plain_table['democracy_index'] = democracy_index
    plain_table['corruption_index'] = corruption_index
    plain_table['GPD_per_capita'] = GPD_per_capita
    plain_table['fertility_rate'] = fertility_rate
    plain_table['energy_access_procent_of_population'] = energy_access_procent_of_population
    plain_table['population'] = population
    plain_table['unseat_current_state_head_attempt'] = unseat_current_state_head_attempt
    plain_table['trade_openness'] = trade_openness
    plain_table['state_control_over_territory'] = state_control_over_territory
    plain_table['political_violence'] = political_violence

    return plain_table


# Функция для оптимизации модели XGBoost с использованием Optuna
def optimize_xgb(progress_bar, X_train, y_train, X_test, y_test, n_trials):
    def objective(trial):
        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'booster': 'gbtree',
            'scale_pos_weight': len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
            'n_estimators': trial.suggest_int('n_estimators', 1000, 3000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'subsample': trial.suggest_float('subsample', 0.1, 1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1),
            'gamma': trial.suggest_float('gamma', 0, 1),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 0.5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 0.7),
            'seed': 42,
        }

        xgb_clf = xgb.XGBClassifier(**xgb_params)

        xgb_clf.fit(X_train, y_train, verbose=False)

        return roc_auc_score(y_test, xgb_clf.predict_proba(X_test)[:, 1])

    # Создаем новый прогресс-бар
    progress_bar.progress(0)

    study_xgb = optuna.create_study(study_name='xgb-seed42', direction='maximize')

    # Оптимизация с отображением прогресс-бара
    study_xgb.optimize(objective, n_trials=n_trials,
                       callbacks=[lambda study, trial: progress_bar.progress(trial.number / n_trials)])

    best_params = study_xgb.best_params

    return best_params

st.write("""
# Прогнозирование вероятности возникновения внутреннего вооруженного конфликта

Данное приложение прогнозирует вероятность возникновения внутреннего 
вооруженного конфликта на **основе экономических, военно-политических и демографических** показателей.
""")

st.write('**Ресурсы**, откуда Вы можете взять данные для прогнозирования вероятности возникновения внутреннего вооруженного конфликта:')

st.markdown("""
<ul>
    <li><a href="https://v-dem.net/data/the-v-dem-dataset/" target="_blank">Датасеты V-DEM</a></li>
    <li><a href="https://data.worldbank.org/" target="_blank">World Bank Open Data</a></li>
    <li><a href="https://ourworldindata.org/" target="_blank">Our world in data</a></li>
    <li><a href="https://www.sipri.org/databases" target="_blank">Датасеты SIPRI</a></li>
    <li><a href="https://www.politicalterrorscale.org/Data/" target="_blank">Политические репрессии</a></li>

</ul>
""", unsafe_allow_html=True)

st.sidebar.header('Настройки пользователя')

sidebar_option = st.sidebar.selectbox('Выберите функционал',
                                      ['Предсказание вероятности ВВК', 'Обучение модели'])

if sidebar_option == 'Предсказание вероятности ВВК':
    st.sidebar.subheader('Требуются данные для предсказания')

    data_input_option = st.sidebar.selectbox('Выбери вариант подачи данных',
                                             ['Загрузить csv-файл', 'Ручной ввод значений предикторов'])

    if data_input_option == 'Загрузить csv-файл':

        uploaded_file = st.sidebar.file_uploader('Загрузите Ваш файл в формате CSV', type=['csv'])

        if uploaded_file is not None:
            input_data = pd.read_csv(uploaded_file)
            st.write('Ваши данные для загруженного файла')
            st.write(input_data)

            if 'btn_csv_download_page' not in st.session_state:
                st.session_state.btn_csv_download_page = False

            if 'button_label_csv_download_page' not in st.session_state:
                st.session_state.button_label_csv_download_page = 'Получить предсказание'


            def click_button():
                st.session_state.btn_csv_download_page = not st.session_state.btn_csv_download_page

                if st.session_state.btn_csv_download_page:
                    st.session_state.button_label_csv_download_page = 'Назад'
                else:
                    st.session_state.button_label_csv_download_page = 'Подтвердить загруженные данные'


            st.button(st.session_state.button_label_csv_download_page, on_click=click_button)

            if st.session_state.btn_csv_download_page:
                model_option = st.selectbox('Выберите модель машинного обучения',
                                            ('XGBoost', 'LightGBM'))

                if model_option == 'XGBoost':
                    if st.button('Получить вероятность возникновения ВВК'):
                        xgb_model = load_model('social_tensions_xgb_clf.pkl')
                        predictions_xgb = xgb_model.predict_proba(input_data.iloc[[0]])[:, 1][0]

                        st.write(f'Вероятность возникновения ВВК при заданных параметрах: {predictions_xgb * 100}%')

                        xgb_feature_importance = open('xgb_importance.png', 'rb').read()
                        st.image(xgb_feature_importance, caption='Роль признаков в предсказании', use_column_width=True)

                        st.write('Таким образом, в предсказании ВВК наибольшую роль сыграли: \n'
                                 '1) **Уровень политических репрессий** \n'
                                 '2) **Население страны** \n'
                                 '3) **Государственный контроль над территорией страны**')

                        

                elif model_option == "LightGBM":
                    if st.button('Получить вероятность возникновения ВВК'):
                        lgbm_model = load_model('social_tensions_lgbm_clf.pkl')
                        predictions_lgbm = lgbm_model.predict_proba(input_data.iloc[[0]])[:, 1][0]

                        st.write(f'Вероятность возникновения ВВК при заданных параметрах: {predictions_lgbm * 100}%')

                        lgbm_feature_importance = open('lgbm_importance.png', 'rb').read()
                        st.image(lgbm_feature_importance, caption='Роль признаков в предсказании',
                                 use_column_width=True)

                        st.write('Таким образом, в предсказании ВВК наибольшую роль сыграли: \n'
                                 '1) **Инфляция, %** \n'
                                 '2) **Открытость торговли** \n'
                                 '3) **Военные расходы, % от ВВП**')




    elif data_input_option == 'Ручной ввод значений предикторов':

        st.write('Регулирируйте ползунки в **левой части экрана**, '
                 'выставляя нужные значения предикторов. '
                 'Если все значения выставлены, нажмите кнопку **Подтвердить**')

        plain_table = empty_pandas_table_creation()


        filled_table = user_input_features(plain_table)

        if 'btn' not in st.session_state:
            st.session_state.btn = False

        if 'button_label' not in st.session_state:
            st.session_state.button_label = 'Подтвердить'
        def click_button():
            st.session_state.btn = not st.session_state.btn

            if st.session_state.btn:
                st.session_state.button_label = 'Назад'
            else:
                st.session_state.button_label = 'Подтвердить'


        st.button(st.session_state.button_label, on_click=click_button)

        if st.session_state.btn:
            scaled_table = scale_features(filled_table)

            st.write('Таблица проскалирована')
            st.write(scaled_table)

            scaled_table = scaled_table.drop(columns=['GPD_per_capita', 'fertility_rate'])

            model_option = st.selectbox('Выберите модель машинного обучения',
                                        ('XGBoost', 'LightGBM'))

            if model_option == "XGBoost":
                if st.button('Получить вероятность возникновения ВВК'):
                    xgb_model = load_model('social_tensions_xgb_clf.pkl')
                    predictions_xgb = xgb_model.predict_proba(scaled_table.iloc[[0]])[:, 1][0]

                    st.write(f'Вероятность возникновения ВВК при заданных параметрах: {predictions_xgb * 100 }%')

                    xgb_feature_importance = open('xgb_importance.png', 'rb').read()
                    st.image(xgb_feature_importance, caption='Роль признаков в предсказании', use_column_width=True)

                    st.write('Таким образом, в предсказании ВВК наибольшую роль сыграли: \n'
                             '1) **Уровень политических репрессий** \n'
                             '2) **Население страны** \n'
                             '3) **Государственный контроль над территорией страны**')

            elif model_option == "LightGBM":
                if st.button('Получить вероятность возникновения ВВК'):
                    lgbm_model = load_model('social_tensions_lgbm_clf.pkl')
                    predictions_lgbm = lgbm_model.predict_proba(scaled_table.iloc[[0]])[:, 1][0]

                    st.write(f'Вероятность возникновения ВВК при заданных параметрах: {predictions_lgbm * 100}%')

                    lgbm_feature_importance = open('lgbm_importance.png', 'rb').read()
                    st.image(lgbm_feature_importance, caption='Роль признаков в предсказании', use_column_width=True)

                    st.write('Таким образом, в предсказании ВВК наибольшую роль сыграли: \n'
                             '1) **Инфляция, %** \n'
                             '2) **Открытость торговли** \n'
                             '3) **Военные расходы, % от ВВП**')

        else:
            st.write(plain_table)


elif sidebar_option == 'Обучение модели':
    uploaded_file_X = st.file_uploader('Загрузите данные для обучения модели в формате CSV', type=['CSV'])

    if uploaded_file_X is not None:
        data_X = pd.read_csv(uploaded_file_X)
        st.write(data_X)

        uploaded_file_y = st.file_uploader('Загрузите данные (**столбец**) с величинами для предсказания в формате CSV. \n'
                                           'Данная таблица должна иметь размерность **Nx1**, где N - количество событий', type=['CSV'])

        if uploaded_file_y is not None:
            data_y = pd.read_csv(uploaded_file_y)
            st.write(data_y)

            if data_y.shape[1] == 1:

                model_option = st.selectbox('Выберите модель для обучения',
                                            options=('XGBoost', 'LightGBM'))

                if model_option == 'XGBoost':
                    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.25, random_state=42)

                    st.write(f"75% данных будут использованы для обучения, 25% - для тестирования. \n"
                             f"Размерность **X_train** выборки: {X_train.shape}, **X_test** выборки: {X_test.shape}. \n"
                             f"Размерность **y_train** выборки: {y_train.shape}, **y_test** выборки: {y_test.shape}",
                             unsafe_allow_html=True)

                    # Получаем текущее состояние сессии
                    session_state = get_session_state()

                    session_state.pop('best_xgb_params', None)
                    session_state.pop('best_xgb_model', None)
                    session_state.pop('y_pred_xgb', None)

                    if 'progress_bar' not in session_state:
                        session_state.progress_bar = None

                    if 'best_xgb_params' not in session_state:
                        session_state.best_xgb_params = None

                    if 'best_xgb_model' not in session_state:
                        session_state.best_xgb_model = None

                    if 'y_pred_xgb' not in session_state:
                        session_state.y_pred_xgb = None



                    if st.button('Начать обучение'):
                        session_state.progress_bar = st.progress(0)
                        session_state.best_xgb_params = optimize_xgb(session_state.progress_bar, X_train, y_train,
                                                                     X_test, y_test, n_trials=20)

                        st.write(f'Параметры лучшей модели: {session_state.best_xgb_params}')

                        session_state.best_xgb_model = xgb.XGBClassifier(**session_state.best_xgb_params)
                        session_state.best_xgb_model.fit(X_train, y_train, verbose=False)

                        session_state.y_pred_xgb = session_state.best_xgb_model.predict_proba(X_test)[:, 1]

                    if 'best_xgb_params' is not None:
                        model_statistics_or_download_selectbox = st.selectbox(
                            'Теперь вы можете визуализировать результаты обучения или скачать обученную модель',
                            options=('Просмотр статистики', 'Скачать модель'))

                        if model_statistics_or_download_selectbox == 'Просмотр статистики':
                            st.write(f'ROC-AUC-score для лучшей модели: {roc_auc_score(y_test, session_state.y_pred_xgb)}')
                            st.write(f'Accuracy для лучшей модели: {accuracy_score(y_test, session_state.best_xgb_model.predict(X_test))}')
                            st.write(f'Precision для лучшей модели: {precision_score(y_test, session_state.best_xgb_model.predict(X_test))}')
                            st.write(f'Recall для лучшей модели: {recall_score(y_test, session_state.best_xgb_model.predict(X_test))}')

                            session_state.xgb_feature_importance = session_state.best_xgb_model.feature_importances_
                            session_state.xgb_feature_importance_df = pd.DataFrame(
                                {'feature': data_X.columns,
                                 'XGBoost': session_state.xgb_feature_importance}
                            ).sort_values(by='XGBoost', ascending=False)

                            plt.figure(figsize=(10, 8))
                            sns.barplot(data=session_state.xgb_feature_importance_df, x='XGBoost', y='feature', palette='viridis')
                            plt.title('Важность признаков')
                            plt.xlabel('Важность признаков')
                            plt.ylabel('Признак')

                            st.pyplot(plt)


                        elif model_statistics_or_download_selectbox == 'Скачать модель':
                            with open('trained_model.pkl', 'wb') as f:
                                pickle.dump(session_state.best_xgb_model, f)

                            st.write('Обученная модель сохранена в файл **trained_model.pkl**')
                            st.download_button(label='Скачать обученную модель', data='trained_model.pkl', file_name='trained_model.pkl', mime='application/octet-stream')




























