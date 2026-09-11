import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from names_dataset import NameDataset
nd = NameDataset()
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, mean_absolute_error, mean_absolute_percentage_error
from tabpfn_client import TabPFNRegressor, set_access_token
import os
import logging
import warnings
import joblib
from datetime import datetime

class BLPayoutModelsFit:

    def __init__(self, input_path, input_file, output_predictors_path, train_test):
        self.input_path = input_path
        self.input_file = input_file
        self.output_predictors_path = output_predictors_path
        self.train_test = train_test
        self.logger = self.setup_bl_logger(log_dir=self.output_predictors_path)
        self.capture_warnings()

    def setup_bl_logger(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"log_file_bl_train_{timestamp}.log"
        log_filepath = os.path.join(log_dir, log_filename)
        # get the root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        # clear existing handlers to prevent duplicates
        if root_logger.hasHandlers():
            root_logger.handlers.clear()

        file_handler = logging.FileHandler(log_filepath, mode='w')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        root_logger.addHandler(file_handler)
        return root_logger

    def capture_warnings(self):
        # Redirect warnings to the logger
        def custom_warn(message, category, filename, lineno, file=None, line=None):
            self.logger.warning(f'{category.__name__}: {message} (File: {filename}, line: {lineno})')
        warnings.showwarning = custom_warn

    def check_match(self, row):
        disp = str(row['disposition_source']).lower()
        cli = str(row['client_name']).lower()
        if pd.isna(disp) or pd.isna(cli):
            return np.nan
        if disp == cli:
            return row['client_name']
        src_words = [w for w in disp.split() if w != 'api']
        for word in src_words:
            if word in cli:
                return row['client_name']
        return np.nan

    def import_preprocess(self):
        bl_data = pd.read_csv(self.input_path + self.input_file, low_memory=False)
        # for train - historical data with labels: 'payout', 'disposition', 'disposition_source'
        needed_columns = ['session_id', 'session_dt', 'conversion_dt', 'register_date',
                          'campaign_id', 'page', 'auto_city', 'auto_country', 'auto_state', 'device_type', 'sub1',
                          'sub2', 'sub3', 'business_type', 'credit_score', 'industry', 'loan_amount', 'loan_reason',
                          'monthly_revenue', 'time_in_business',
                          'fname', 'lname', 'cellphone', 'client_name', 'payout', 'disposition', 'disposition_source']
        bl_data = bl_data[needed_columns]

        # target columns
        bl_data['payout'] = bl_data['payout'].fillna(0)
        bl_data['payout_adj'] = bl_data['payout']
        bl_data['n_session'] = bl_data.groupby('session_id')['session_dt'].transform('count')
        bl_data['client_buy'] = bl_data.apply(self.check_match, axis=1)
        bl_data['client_buy'] = np.where(bl_data['client_buy'].isna() & (bl_data['n_session'] == 1) &
                                         (~bl_data['client_name'].isna()) & (bl_data['disposition_source'] == 'Lead'),
                                         bl_data['client_name'], bl_data['client_buy'])
        bl_data['payout_adj'] = np.where(bl_data['client_buy'].isna(), 0, bl_data['payout_adj'])
        bl_data['disposition'] = np.where(bl_data['client_buy'].isna(), np.nan, bl_data['disposition'])
        bl_data['client_name'] = bl_data['client_name'].fillna('other')
        bl_data = bl_data.drop(columns=['n_session', 'client_buy'])
        bl_data['sold_to_client'] = (bl_data['disposition'] == 'Lead').astype(int)

        self.logger.info(f'raw data rows: {bl_data.shape[0]}')
        # filter out rows with null register_date since they cannot be leads
        bl_data = bl_data[bl_data['register_date'].notna()]
        bl_data = bl_data.rename(columns={'auto_city': 'city', 'auto_state': 'state', 'auto_country': 'country'})
        self.logger.info(f'register date exists - can be leads: {bl_data.shape[0]}')
        bl_data[['country', 'state', 'city', 'sub1', 'sub2', 'sub3']] = (
            bl_data[['country', 'state', 'city', 'sub1', 'sub2', 'sub3']].fillna('Other'))
        bl_data['country_state'] = np.where(bl_data['country'] == 'United States', bl_data['state'], bl_data['country'])
        bl_data['session_dt'] = pd.to_datetime(bl_data['session_dt'], errors="coerce")
        bl_data['register_date'] = pd.to_datetime(bl_data['register_date'], errors="coerce")
        bl_data['sub1'] = bl_data['sub1'].astype(str)
        bl_data['sub2'] = bl_data['sub2'].astype(str)
        bl_data['sub3'] = bl_data['sub3'].astype(str)
        bl_data['cellphone_prefix'] = bl_data['cellphone'].astype(int).astype(str).str[:3].astype(str)
        return bl_data

    def impute_survey_columns(self, bl_data):
        survey_columns = ["credit_score", "industry", "loan_amount", "loan_reason", "monthly_revenue",
                                 "time_in_business", "device_type", "business_type"]
        # only 1 or 2 survey questions may be missing; otherwise the row will be removed
        bl_data = bl_data.dropna(subset=survey_columns, thresh=5) # 7 main features - 2 missing answers = 5
        self.logger.info(f'register date exists - survey answers are sufficient to processing: {bl_data.shape[0]}')
        for col in survey_columns:
            bl_data[col] = bl_data[col].str.lower()
            bl_data[col] = bl_data[col].where(bl_data[col].str.len() > 1, np.nan)

        bl_data[survey_columns] = bl_data[survey_columns].fillna("other")
        return bl_data

    def credit_score_to_numeric(self, bl_data):
        bl_data['credit_score_num'] = -99
        bl_data.loc[bl_data['credit_score'].str.contains('550'), 'credit_score_num'] = 501
        bl_data.loc[bl_data['credit_score'].str.contains('550') &
                        bl_data['credit_score'].str.contains('599'), 'credit_score_num'] = 551
        bl_data.loc[bl_data['credit_score'].str.contains('600') &
                        bl_data['credit_score'].str.contains('649'), 'credit_score_num'] = 601
        bl_data.loc[bl_data['credit_score'].str.contains('650') &
                        bl_data['credit_score'].str.contains('719'), 'credit_score_num'] = 651
        bl_data.loc[bl_data['credit_score'].str.contains('720'), 'credit_score_num'] = 721
        bl_data = bl_data.drop(['credit_score'], axis=1)
        return bl_data

    def credit_loan_amount_to_numeric(self, bl_data):
        # impute unknown outlying value
        bl_data['loan_amount_num'] = -99
        # impute middle of the range
        bl_data.loc[bl_data['loan_amount'].str.contains('10,000') &
                    bl_data['loan_amount'].str.contains('24,999'), 'loan_amount_num'] = 17500
        bl_data.loc[bl_data['loan_amount'].str.contains('25,000') &
                    bl_data['loan_amount'].str.contains('49,999'), 'loan_amount_num'] = 47500
        bl_data.loc[bl_data['loan_amount'].str.contains('50,000') &
                    bl_data['loan_amount'].str.contains('74,999'), 'loan_amount_num'] = 57500
        bl_data.loc[bl_data['loan_amount'].str.contains('75,000') &
                    bl_data['loan_amount'].str.contains('99,999'), 'loan_amount_num'] = 75000
        bl_data.loc[bl_data['loan_amount'].str.contains('100,000'), 'loan_amount_num'] = 150000
        bl_data.loc[bl_data['loan_amount'].str.contains('200,000'), 'loan_amount_num'] = 250000
        bl_data = bl_data.drop(['loan_amount'], axis=1)
        return bl_data

    def monthly_revenue_to_numeric(self, bl_data):
        # impute unknown outlying value
        bl_data['monthly_revenue_num'] = -99
        # impute middle of the range
        bl_data.loc[bl_data['monthly_revenue'].str.contains('9,999'), 'monthly_revenue_num'] = 5000
        bl_data.loc[bl_data['monthly_revenue'].str.contains('10,000') &
                    bl_data['monthly_revenue'].str.contains('19,999'), 'monthly_revenue_num'] = 15000
        bl_data.loc[bl_data['monthly_revenue'].str.contains('20,000') &
                    bl_data['monthly_revenue'].str.contains('49,999'), 'monthly_revenue_num'] = 35000
        bl_data.loc[bl_data['monthly_revenue'].str.contains('50,000') &
                    bl_data['monthly_revenue'].str.contains('99,999'), 'monthly_revenue_num'] = 75000
        bl_data.loc[bl_data['monthly_revenue'].str.contains('100,000'), 'monthly_revenue_num'] = 150000
        bl_data.loc[bl_data['monthly_revenue'].str.contains('200,000'), 'monthly_revenue_num'] = 250000
        bl_data = bl_data.drop(['monthly_revenue'], axis=1)
        return bl_data

    def time_in_business_to_num(self, bl_data):
        # impute worse case - new business
        mapping = {
            '2+ years': 36,
            'less than 6 months': 3,
            '1-2 years': 18,
            '$75,000 - $99,999': 87500,
            '6-12 months': 8
        }
        bl_data['time_in_business_num'] = bl_data['time_in_business'].map(mapping).fillna(0)
        bl_data = bl_data.drop(['time_in_business'], axis=1)
        return bl_data

    def time_features(self, bl_data):
        # day, day of week and hour of day
        bl_data['session_day'] = bl_data['session_dt'].dt.day
        bl_data['session_day_of_week'] = bl_data['session_dt'].dt.day_name()
        bl_data['session_hour'] = bl_data['session_dt'].dt.hour
        bl_data['from_start_to_register'] = (bl_data['register_date'] - bl_data['session_dt']).dt.total_seconds()
        bl_data = bl_data.drop(['register_date', 'conversion_dt'], axis=1)
        return bl_data

    def detect_gender_with_confidence(self, fname):
        result = nd.search(fname)
        if result is None:
            return 'unknown', 0.0
        first = result.get('first_name')
        if first is None:
            return 'unknown', 0.0
        gender_data = first.get('gender')
        if gender_data is None:
            return 'unknown', 0.0
        m = gender_data.get('Male', 0)
        f = gender_data.get('Female', 0)
        total = m + f
        if total == 0:
            return 'unknown', 0.0
        if m >= f:
            return 'male', round(m / total, 3)
        else:
            return 'female', round(f / total, 3)

    def additional_features(self, bl_data):
        bl_data['ratio_loan_amount_revenue'] = bl_data['loan_amount_num'] / bl_data['monthly_revenue_num']
        # from name
        bl_data[['gender', 'gender_confidence']] = bl_data['fname'].apply(
            lambda x: pd.Series(self.detect_gender_with_confidence(str(x).strip().capitalize())))
        bl_data['fname_len'] = bl_data['fname'].str.len()
        bl_data['lname_len'] = bl_data['lname'].str.len()
        return bl_data

    def split_by_time(self, bl_data, days_for_test):
        bl_data = bl_data.sort_values(by='session_dt')
        # split and cut the first week with incomplete history
        max_date = bl_data['session_dt'].max()
        recent_start = max_date - pd.Timedelta(days=days_for_test - 1)
        # split indicator (0 - train, 1:days_for_test - daily tests)
        bl_data['split_day'] = (bl_data['session_dt'] - recent_start).dt.days + 1
        bl_data.loc[bl_data['split_day'] < 1, 'split_day'] = 0
        bl_train = bl_data[bl_data['split_day'] == 0].drop('split_day', axis=1)
        bl_test_all_days = bl_data[bl_data['split_day'] > 0]
        return bl_train, bl_test_all_days

    def bl_preprocessing(self):
        bl_data = self.import_preprocess()
        bl_data = self.impute_survey_columns(bl_data)
        bl_data = self.credit_score_to_numeric(bl_data)
        bl_data = self.credit_loan_amount_to_numeric(bl_data)
        bl_data = self.monthly_revenue_to_numeric(bl_data)
        bl_data = self.time_in_business_to_num(bl_data)
        bl_data = self.time_features(bl_data)
        bl_data = self.additional_features(bl_data)
        bl_data = bl_data.reset_index(drop=True)
        bl_train, bl_test_all_days = self.split_by_time(bl_data, days_for_test=7)
        train_columns = ['campaign_id', 'page', 'city', 'device_type', 'sub1', 'sub2', 'sub3',
        'business_type', 'industry', 'loan_reason', 'cellphone_prefix',
         'country_state', 'credit_score_num', 'loan_amount_num', 'monthly_revenue_num',
        'time_in_business_num', 'session_day', 'session_day_of_week',
        'session_hour', 'from_start_to_register', 'ratio_loan_amount_revenue',
        'gender', 'fname_len', 'lname_len', 'client_name']
        label_columns = ['sold_to_client', 'payout', 'payout_adj']
        x_train = bl_train[train_columns]
        y_train = bl_train[label_columns]
        if self.train_test == True:
            x_test_all_days = bl_test_all_days[train_columns + ['split_day']]
            y_test_all_days = bl_test_all_days[label_columns + ['split_day']]
        else:
            x_test_all_days = pd.DataFrame()
            y_test_all_days = pd.DataFrame()
            all_clients = bl_data[['client_name']].drop_duplicates()
            all_clients.to_csv(self.output_predictors_path + 'all_clients.csv', index=False)
        return x_train, y_train, x_test_all_days, y_test_all_days

    def catbosot_model_sold_to_client(self, x_train, y_train):
        cat_features = x_train.select_dtypes(include=['object', 'category']).columns.tolist()
        CB_bl_lead = CatBoostClassifier(random_seed=42, depth=8, n_estimators=800, task_type='CPU',
                                        eval_metric='F1', verbose=False, allow_writing_files=False)
        CB_bl_lead.fit(x_train, y_train['sold_to_client'], cat_features=cat_features)
        return CB_bl_lead

    def accuracy_classification_model(self, CB_model, y_test_all_days, x_test_all_days, target_col):
        self.logger.info(f'CB accuracy report - ' + target_col + ' for all test \n'
             f'{classification_report(y_test_all_days[target_col],
                    CB_model.predict(x_test_all_days.drop("split_day", axis=1)))}')
        for day in range(1, 8):
            y_test_day = y_test_all_days[y_test_all_days['split_day'] == day][target_col]
            x_test_day = x_test_all_days[x_test_all_days['split_day'] == day]
            x_test_day = x_test_day.drop('split_day', axis=1)
            self.logger.info(f'CB accuracy report ' + target_col + ' (BL) for day {day} \n'
                                                        f' :{classification_report(y_test_day, CB_model.predict(x_test_day))}')

    def prepare_for_cont_payout_prediction_train(self, x_train, y_train):
        # cont. payout prediction
        mask_payout_train = y_train['payout'] > 0
        y_train_payout = y_train[mask_payout_train]
        x_train_payout = x_train[mask_payout_train]
        return x_train_payout, y_train_payout

    def prepare_for_cont_payout_prediction_test(self, x_test_all_days, y_test_all_days):
        mask_payout_test = y_test_all_days['payout'] > 0
        y_test_payout = y_test_all_days[mask_payout_test].drop('split_day', axis=1)
        x_test_payout = x_test_all_days[mask_payout_test].drop('split_day', axis=1)
        return x_test_payout, y_test_payout

    def tabpfn_regression_payout(self, x_train_payout, y_train_payout):
        self.logger.info('start tabular transformer')
        set_access_token(os.environ["TABPFN_TOKEN"])
        CONTEXT_SIZE = 1000  # most recent context
        model_tfm = TabPFNRegressor(ignore_pretraining_limits=True)
        tfm_context = {
            'x': x_train_payout.iloc[-CONTEXT_SIZE:],
            'y': y_train_payout['payout'].iloc[-CONTEXT_SIZE:],
            'columns': list(x_train_payout.columns),
        }
        model_tfm.fit(tfm_context['x'], tfm_context['y'])
        self.logger.info('finish tabular transformer')
        return model_tfm, tfm_context

    def accuracy_cont_payout_prediction(self, model_tfm, x_test_payout, y_test_payout):
        # tabpfn is fitted on raw payout - no inverse transform
        y_test_payout['pred_payout'] = model_tfm.predict(x_test_payout)
        self.logger.info(f'Overall: \n Payout mean_absolute_percentage_error:'
              f'{mean_absolute_percentage_error(y_test_payout["payout"], y_test_payout["pred_payout"])}')
        self.logger.info(f'Payout mean_absolute_error:{mean_absolute_error(y_test_payout["payout"],
                                                                y_test_payout["pred_payout"])}')
        # per day
        for thr in [5, 10, 15, 20]:
            self.logger.info(f'thr: {thr}')
            y_test_payout_restricted = y_test_payout[y_test_payout['pred_payout'] >= thr]
            self.logger.info(f'Payout mean_absolute_percentage_error:{mean_absolute_percentage_error(y_test_payout_restricted["payout"],
                                                                                          y_test_payout_restricted["pred_payout"])}')
            self.logger.info(f'Payout mean_absolute_error:{mean_absolute_error(y_test_payout_restricted["payout"],
                                                                    y_test_payout_restricted["pred_payout"])}')

    def save_models(self, tfm_context, CB_bl_lead):
        joblib.dump(tfm_context, self.output_predictors_path + 'payout_tfm_context.joblib')
        CB_bl_lead.save_model(self.output_predictors_path + 'CB_bl_lead.cbm', format='cbm')

    def fit_(self):
        x_train, y_train, x_test_all_days, y_test_all_days = self.bl_preprocessing()
        CB_bl_lead = self.catbosot_model_sold_to_client(x_train, y_train)
        self.logger.info("Catboost classifier for lead finished")
        x_train_payout, y_train_payout = self.prepare_for_cont_payout_prediction_train(x_train, y_train)
        model_tfm, tfm_context = self.tabpfn_regression_payout(x_train_payout, y_train_payout)
        self.logger.info("TabPFN regressor for $ payout finished")

        if self.train_test == True:
            x_test_payout, y_test_payout = self.prepare_for_cont_payout_prediction_test(x_test_all_days, y_test_all_days)
            self.accuracy_classification_model(CB_bl_lead, y_test_all_days, x_test_all_days, target_col='sold_to_client')
            self.accuracy_cont_payout_prediction(model_tfm, x_test_payout, y_test_payout)
            self.logger.info("Train-test assessment finished")
        else:
            self.save_models(tfm_context, CB_bl_lead)
            self.logger.info("Train on all data for production finished")


if __name__ == "__main__":
    input_path = '/your_path/data/'
    input_file = 'bl_full_data.csv'
    output_predictors_path = '/your_path/predictors/'
    os.makedirs(input_path, exist_ok=True)
    os.makedirs(output_predictors_path, exist_ok=True)
    model = BLPayoutModelsFit(input_path, input_file, output_predictors_path, train_test=True)
    model.fit_()
