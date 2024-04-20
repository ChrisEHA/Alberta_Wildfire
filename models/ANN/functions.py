import sys

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
import tensorflow as tf

PROJECT_ROOT = '../../'
sys.path.append(PROJECT_ROOT)
from scripts.data_utils import extract_day_of_year

def get_preprocess_pipeline():
    # Define features to include #
    pass_features = ['leaf_area_index_high_vegetation', 'slope_of_sub_gridscale_orography']
    categorical_features = ['type_of_high_vegetation']
    numeric_features = ['fire_count_past_3Days','fire_count_past_30Days','DMC','global_noon_LST_2m_temperature','BUI',
                    'FWI','latitude','FFMC','global_noon_LST_relative_humidity','24hr_max_temperature',
                    'global_noon_LST_2m_temperature_1dayLag','global_noon_LST_2m_temperature_2dayLag',
                    'high_vegetation_cover','24hr_max_temperature_1dayLag','low_vegetation_cover',
                    '24hr_accumulated_precipitation', 'day_of_the_year']
    ####

    # Define numeric and categorical transformer below
    date_transformer = ColumnTransformer([('date', FunctionTransformer(extract_day_of_year, validate=False), ['date'])], verbose_feature_names_out=False, remainder='passthrough')
    scale=ColumnTransformer([('scale_transformer',StandardScaler(),numeric_features)],verbose_feature_names_out=False).set_output(transform='pandas')
    cate=ColumnTransformer([('categorical_transformer',OneHotEncoder(sparse_output=False),categorical_features)],verbose_feature_names_out=False).set_output(transform='pandas')
    pss=ColumnTransformer([('Pass_transformer','passthrough',pass_features)],verbose_feature_names_out=False).set_output(transform='pandas')

    feature_union = FeatureUnion([
        ('numeric', scale),
        ('categorical', cate),
        ('pass', pss)
    ])


    # Final Pipeline
    final_pipeline = Pipeline([
        ('date_of_the_year', date_transformer),
        ('feature_union', feature_union)
    ])

    return final_pipeline

def load_h5(model_path):
    return tf.keras.models.load_model(model_path)

def extract_day_of_year(X):
    day_of_year = X['date'].dt.dayofyear.to_frame(name='day_of_the_year')
    return day_of_year

def ANN_predict_proba(model,pipeline,X):
    X_transformed = pipeline.transform(X)
    return model.predict(X_transformed)