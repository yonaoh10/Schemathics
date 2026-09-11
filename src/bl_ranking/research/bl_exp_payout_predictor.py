import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from names_dataset import NameDataset
nd = NameDataset()
from catboost import CatBoostClassifier
import os
import logging
import warnings
import joblib
from datetime import datetime
from tabpfn_client import TabPFNRegressor, set_access_token

class BLPayoutModelsPredict:

    def __init__(self, predictors_path, user_data, log_path):
        self.predictors_path = predictors_path
        self.user_data = user_data
        self.user_data_file = user_data
        self.logger = self.setup_bl_logger(log_dir=log_path)
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

    def load_models(self):
        tfm_context = joblib.load(self.predictors_path + 'payout_tfm_context.joblib')
        set_access_token(os.environ["TABPFN_TOKEN"])
        model_tfm = TabPFNRegressor(ignore_pretraining_limits=True)
        model_tfm.fit(tfm_context['x'], tfm_context['y'])
        CB_bl_lead = CatBoostClassifier(allow_writing_files=False).load_model(
            self.predictors_path + 'CB_bl_lead.cbm', format='cbm')
        return model_tfm, tfm_context['columns'], CB_bl_lead

    def import_preprocess(self):
        bl_data = pd.DataFrame([user_data])
        all_clients = pd.read_csv(self.predictors_path + 'all_clients.csv')
        needed_columns = ['session_dt', 'conversion_dt', 'register_date',
                          'campaign_id', 'page', 'auto_city', 'auto_country', 'auto_state', 'device_type', 'sub1',
                          'sub2', 'sub3',
                          'business_type', 'credit_score', 'industry', 'loan_amount', 'loan_reason', 'monthly_revenue',
                          'time_in_business', 'fname', 'lname', 'cellphone']
        bl_data = bl_data[needed_columns]
        if bl_data['register_date'].isna().any():
            raise Exception("user cannot be a lead - register_date is absent")

        bl_data = bl_data.merge(all_clients, how='cross')
        bl_data = bl_data[bl_data['client_name'] != 'other']
        self.logger.info(f'raw data rows: {bl_data.shape[0]}')
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
        bl_data = bl_data.dropna(subset=survey_columns, thresh=5)  # 7 main features - 2 missing answers = 5
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

    def bl_preprocessing(self):
        bl_data = self.import_preprocess()
        if bl_data.shape[0] == 0:
            return pd.DataFrame()
        bl_data = self.impute_survey_columns(bl_data)
        bl_data = self.credit_score_to_numeric(bl_data)
        bl_data = self.credit_loan_amount_to_numeric(bl_data)
        bl_data = self.monthly_revenue_to_numeric(bl_data)
        bl_data = self.time_in_business_to_num(bl_data)
        bl_data = self.time_features(bl_data)
        bl_data = self.additional_features(bl_data)
        bl_data = bl_data.reset_index(drop=True)
        pred_columns = ['campaign_id', 'page', 'city', 'device_type', 'sub1', 'sub2', 'sub3',
                         'business_type', 'industry', 'loan_reason', 'cellphone_prefix',
                         'country_state', 'credit_score_num', 'loan_amount_num', 'monthly_revenue_num',
                         'time_in_business_num', 'session_day', 'session_day_of_week',
                         'session_hour', 'from_start_to_register', 'ratio_loan_amount_revenue',
                         'gender', 'fname_len', 'lname_len', 'client_name']
        bl_data = bl_data[pred_columns]
        return bl_data

    def prediction_expected_payout(self, bl_data, model_tfm, tfm_columns, CB_bl_lead):
        bl_data = bl_data[tfm_columns]
        bl_data['predicted_prob_lead'] = CB_bl_lead.predict_proba(bl_data[tfm_columns])[:, 1]
        self.logger.info('start tabular transformer payout prediction')
        # tabpfn is fitted on raw payout - no inverse transform
        bl_data['payout'] = model_tfm.predict(bl_data[tfm_columns])
        self.logger.info('finish tabular transformer payout prediction')
        bl_data['expected_payout'] = bl_data['payout'] * bl_data['predicted_prob_lead']
        bl_data = bl_data[bl_data['expected_payout'] > 0.01]
        if bl_data.shape[0] == 0:
            return {}
        bl_data = bl_data.sort_values(by=['expected_payout'], ascending=False)
        bl_data['id'] = 'new'
        bl_data['rank_exp_payout'] = bl_data.groupby('id')['expected_payout'].rank(
            ascending=False, method='first')
        df_rank = bl_data[['rank_exp_payout', 'expected_payout', 'client_name']]
        result_clients_ranks = (df_rank.set_index('client_name').rename(columns={'rank_exp_payout': 'rank'}).
                                to_dict(orient='index'))
        return result_clients_ranks

    def predict_(self):
        bl_data = self.bl_preprocessing()
        if bl_data.shape[0] == 0:
            return {"expected_payout": 0, "prob_lead": 0}
        model_tfm, tfm_columns, CB_bl_lead = self.load_models()
        self.logger.info('models were loaded')
        result_clients_ranks = self.prediction_expected_payout(bl_data, model_tfm, tfm_columns, CB_bl_lead)
        self.logger.info(f"expected_payout {result_clients_ranks}")
        return result_clients_ranks


if __name__ == "__main__":
    predictors_path = '/Users/yurygubman/Results/BL/predictors/'
    user_data = {
        "session_dt": "2026-01-06 19:24:22",
        "conversion_dt": "2026-01-06 19:26:10",
        "register_date": "2026-01-06 19:26:07",
        "campaign_id": 120227360861540306,
        "page": "top10us.com/app/business-loans-v2",
        "auto_city": "Fort Lauderdale",
        "auto_country": "United States",
        "auto_state": "Florida",
        "device_type": "mobile",
        "sub1": 1121993,
        "sub2": "01121993 Ad set",
        "sub3": 1513124082,
        "business_type": "C Corporation",
        "credit_score": "Very Poor - Under 550",
        "industry": "construction",
        "loan_amount": "$25,000 - $49,999",
        "loan_reason": "Equipment purchase",
        "monthly_revenue": "$20,000 - $49,999",
        "time_in_business": "2+ years",
        "fname": "Rigoberto",
        "lname": "Rodriguez",
        "cellphone": 7869914030,
    }
    log_path = '/Users/yurygubman/Results/BL/logs/'
    os.makedirs(log_path, exist_ok=True)
    result_clients_ranks = BLPayoutModelsPredict(predictors_path, user_data, log_path).predict_()
