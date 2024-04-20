import sys

from sklearn.pipeline import FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn import set_config
set_config(transform_output = "pandas")

PROJECT_ROOT = '../../'
sys.path.append(PROJECT_ROOT)
from scripts.data_utils import extract_day_of_year


def SVM_preprocess_steps():
    ## Define
    pass_features = [ 'leaf_area_index_high_vegetation']
    categorical_features = []
    numeric_features = ['fire_count_past_3Days', 'global_noon_LST_2m_temperature', 'FFMC', 'DMC', 'FWI', 'BUI', 'global_noon_LST_relative_humidity', '24hr_max_temperature', 'day_of_the_year']
    ####

    # Define custom preprocessing functions. Put any custom functions in SVM_functions.py also so they are accessible by the ensemble

    # Define numeric and categorical transformer below
    date_transformer = ColumnTransformer([('date', FunctionTransformer(extract_day_of_year, validate=False), ['date'])], verbose_feature_names_out=False, remainder='passthrough')
    scale=ColumnTransformer([('scale_transformer',StandardScaler(),numeric_features)],verbose_feature_names_out=False).set_output(transform='pandas')
    cate=ColumnTransformer([('categorical_transformer',OneHotEncoder(sparse_output=False),categorical_features)],verbose_feature_names_out=False).set_output(transform='pandas')
    pss=ColumnTransformer([('Pass_transformer','passthrough',pass_features)],verbose_feature_names_out=False).set_output(transform='pandas')

    feature_union = FeatureUnion([
        ('numeric', scale),
        ('categorical', cate),
        ('pass', pss),
    ])

    return date_transformer, feature_union


def SVM_predict(model,pipeline,X):
    X_transformed = pipeline.transform(X)
    return model.predict(X_transformed)
