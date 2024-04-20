import sys

from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn import set_config
set_config(transform_output = "pandas")

PROJECT_ROOT = '../../'
sys.path.append(PROJECT_ROOT)
from scripts.data_utils import extract_day_of_year


## Begin functions
def get_predictions_pipeline(ann_predictor,rf_predictor):
    """
    Create pipeline to add predictions from participating models to feature space
    """
    ann = FunctionTransformer(ann_predictor.predict, validate=False)
    rf = FunctionTransformer(rf_predictor.predict, validate=False)
    passthrough = FunctionTransformer(lambda x: x, validate=False)

    Participant_preditcions = FeatureUnion([
        ('original', passthrough),
        ('ann', ann),
        ('rf', rf),
    ])

    return Participant_preditcions

def create_classifier_pipeline():
    """
    Creates the classification pipeline. Includes prediction and most preprocessing steps
    Add participating model predictions to feature space before passing to this pipeline.
    """
    # Define features to include #
    pass_features = ['leaf_area_index_high_vegetation', 'ANN_Prediction', 'RF_Prediction', 'slope_of_sub_gridscale_orography']
    categorical_features = ['type_of_high_vegetation']
    numeric_features = ['fire_count_past_10Days','DMC','global_noon_LST_2m_temperature','BUI',
                    'FWI','latitude','FFMC','global_noon_LST_relative_humidity','24hr_max_temperature',
                    'global_noon_LST_2m_temperature_1dayLag','global_noon_LST_2m_temperature_2dayLag',
                    'high_vegetation_cover','24hr_max_temperature_1dayLag','low_vegetation_cover',
                    '24hr_accumulated_precipitation']
    ####

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

    params = {'bootstrap': True,
    'ccp_alpha': 0.0,
    'class_weight': None,
    'criterion': 'gini',
    'max_depth': 55,
    'max_features': 'sqrt',
    'max_leaf_nodes': None,
    'max_samples': None,
    'min_impurity_decrease': 0.0,
    'min_samples_leaf': 1,
    'min_samples_split': 2,
    'min_weight_fraction_leaf': 0.0,
    'monotonic_cst': None,
    'n_estimators': 1500,
    'oob_score': False,
    'random_state': 42,
    'classifier__warm_start': False}

    ensemble_classifier = RandomForestClassifier(**params)

    # Final Pipeline (not including participant predictions)
    ensemble_pipeline = Pipeline([
        ('day_of_year', date_transformer),
        ('feature_union', feature_union),
        ('classifier', ensemble_classifier)
    ])

    return ensemble_pipeline