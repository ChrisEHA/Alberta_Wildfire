## Functions for SVM Deployment ##
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder

def SVM_test_train_validation_split(validation_df, test_train_df, target_variable='fire', test_proportion=0.33):
    """
    Validation data is obtained by taking all data after a certain time. This is similar to model deployment.
    Train and test data are obtained using a stratified split
    """
    X_validation = validation_df.drop(columns={target_variable})
    y_validation = validation_df[target_variable]

    X_train, X_test, y_train, y_test = train_test_split(test_train_df.drop(columns={target_variable}), 
                                                        test_train_df[target_variable], 
                                                        test_size=test_proportion,
                                                        stratify=test_train_df[target_variable], 
                                                        random_state=42)
    
    return X_train, X_test, X_validation, y_train, y_test, y_validation

def SVM_pipeline():
    ## Define model features here ##
    pass_features = [ 'leaf_area_index_high_vegetation']
    categorical_features = []
    numeric_features = ['fire_count_past_3Days', 'global_noon_LST_2m_temperature', 'FFMC', 'DMC', 'FWI', 'BUI', 'global_noon_LST_relative_humidity', '24hr_max_temperature']
    ####

    def extract_day_of_year(X):
        day_of_year = X['date'].dt.dayofyear.to_frame(name='day_of_the_year')
        return day_of_year

    # Define numeric and categorical transformer below
    date_transformer = FunctionTransformer(extract_day_of_year)
    scale = ColumnTransformer([('scale_transformer', StandardScaler(), numeric_features)], verbose_feature_names_out=False).set_output(transform='pandas')
    cate = ColumnTransformer([('categorical_transformer', OneHotEncoder(sparse_output=False), categorical_features)], verbose_feature_names_out=False).set_output(transform='pandas')
    pss = ColumnTransformer([('Pass_transformer', 'passthrough', pass_features)], verbose_feature_names_out=False).set_output(transform='pandas')

    # Create the pipeline
    Data_pipeline = Pipeline(steps=[
        ('Feature Union', FeatureUnion([
            ('numeric', scale),
            ('categorical', cate),
            ('pass', pss),
            ('date', date_transformer)
        ]))
    ])

    return Data_pipeline

