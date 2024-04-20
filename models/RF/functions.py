from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion

def get_preprocess_pipeline(X_train):
    pass_features = [ 'leaf_area_index_high_vegetation', 'leaf_area_index_low_vegetation', 
                 'slope_of_sub_gridscale_orography',]
    categorical_features = ['type_of_high_vegetation', 'type_of_low_vegetation']
    numeric_features = X_train.drop(columns=pass_features).drop(columns=categorical_features).keys().drop(['date','fire_count_past_3Days', 'fire_count_past_7Days',
        'fire_count_past_10Days', 'fire_count_past_30Days'])
    feature_names =['pass__slope_of_sub_gridscale_orography',
    'numeric__DMC',
    'numeric__global_noon_LST_2m_temperature',
    'numeric__BUI',
    'numeric__FWI',
    'numeric__latitude',
    'numeric__longitude',
    'numeric__FFMC',
    'numeric__global_noon_LST_relative_humidity',
    'numeric__24hr_max_temperature',
    'numeric__global_noon_LST_2m_temperature_1dayLag',
    'pass__leaf_area_index_high_vegetation']

    # Define numeric and categorical transformer below
    scale=ColumnTransformer([('scale_transformer',StandardScaler(),numeric_features)],verbose_feature_names_out=False).set_output(transform='pandas')

    cate=ColumnTransformer([('categorical_transformer',OneHotEncoder(sparse_output=False),categorical_features)],verbose_feature_names_out=False).set_output(transform='pandas')

    pss=ColumnTransformer([('Pass_transformer','passthrough',pass_features)],verbose_feature_names_out=False).set_output(transform='pandas')
    Drop_transformer=ColumnTransformer([('Drop_transformer','passthrough',feature_names)],verbose_feature_names_out=False).set_output(transform='pandas')

    Data_pipeline = Pipeline(steps=[
        ('Feature Union',FeatureUnion([('numeric', scale),('categorical',cate),('pass',pss)])),
        ('pass',Drop_transformer)]
        )
    
    return Data_pipeline

def RF_predict_proba(model,pipeline,X):
    X_transformed = pipeline.transform(X)
    return model.predict_proba(X_transformed)[:,1]