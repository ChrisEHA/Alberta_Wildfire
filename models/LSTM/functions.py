import sys
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import BinaryAccuracy, AUC
from keras_tuner import Hyperband
from keras_tuner import HyperParameters as hp

PROJECT_ROOT = '../../'
sys.path.append(PROJECT_ROOT)
from scripts.data_utils import extract_day_of_year


## Begin Functions
def LSTM_test_train_validation_split(validation_df, test_train_df, target_variable='fire', test_train_split_date=pd.Timestamp('2013-01-01')):
    """
    Validation data is obtained by taking all data after a certain time.
    Train and test data are obtained using temporal splits
    """ 
    # Work on local version of dataframe
    validation_df = validation_df.copy()
    test_train_df = test_train_df.copy()

    # Step 1: Sort data by month, then by year within each month
    validation_df['month'] = validation_df['date'].dt.month
    validation_df['year'] = validation_df['date'].dt.year
    test_train_df['month'] = test_train_df['date'].dt.month
    test_train_df['year'] = test_train_df['date'].dt.year

    # Sorting by month and year
    validation_df.sort_values(by=['month', 'year'], inplace=True)
    test_train_df.sort_values(by=['month', 'year'], inplace=True)
    
    # Temporal data split based on a specified date
    train_df = test_train_df[test_train_df['date'] < test_train_split_date]
    test_df = test_train_df[test_train_df['date'] >= test_train_split_date]

    # Extracting the target variable and dropping it from feature dataframes
    y_train = train_df[target_variable]
    y_test = test_df[target_variable]
    y_validation = validation_df[target_variable]

    X_train = train_df.drop(columns=target_variable)
    X_test = test_df.drop(columns=target_variable)
    X_validation = validation_df.drop(columns=target_variable)
    
    return X_train, X_test, X_validation, y_train, y_test, y_validation

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
    date_transformer = ColumnTransformer([('date', FunctionTransformer(extract_day_of_year, validate=False), ['date'])], verbose_feature_names_out=False, remainder='passthrough').set_output(transform='pandas')
    scale=ColumnTransformer([('scale_transformer',StandardScaler(),numeric_features)],verbose_feature_names_out=False).set_output(transform='pandas')
    cate=ColumnTransformer([('categorical_transformer',OneHotEncoder(sparse_output=False),categorical_features)],verbose_feature_names_out=False).set_output(transform='pandas')
    pss=ColumnTransformer([('Pass_transformer','passthrough',pass_features)],verbose_feature_names_out=False).set_output(transform='pandas')

    feature_union = FeatureUnion([
        ('numeric', scale),
        ('categorical', cate),
        ('pass', pss)
    ])

    # Final preprocessing pipeline
    Data_pipeline = Pipeline([
        ('date_of_the_year', date_transformer),
        ('feature_union', feature_union)
    ])

    return Data_pipeline

## Set up LSTM that is compatible with hyperparemeter tuning
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def LSTM_classifier_wrapper(input_shape):
    def LSTM_classifier(hp):
        model = Sequential()

        # Define the input shape based on the outer function's argument
        model.add(Input(shape=(1, input_shape)))

        # Tuning the number of LSTM layers
        for i in range(1, 3):  # Adjust the range as needed
            model.add(LSTM(units=hp.Int(f'units_{i}', min_value=input_shape, max_value=input_shape*5, step=2),
                           activation=hp.Choice(f'activation_{i}', ['relu', 'tanh', 'sigmoid', 'swish', 'linear']),
                           recurrent_activation='sigmoid',
                           return_sequences=(i < 2)))

        # Tune whether to use dropout
        if hp.Boolean("dropout"):
            model.add(Dropout(rate=0.20))

        # Add the output layer
        model.add(Dense(1, activation='sigmoid'))

        # Define the optimizer learning rate as a hyperparameter
        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")

        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='binary_crossentropy',
                      metrics=[BinaryAccuracy(), AUC()])

        return model
    return LSTM_classifier
####

def load_h5(model_path):
    return tf.keras.models.load_model(model_path)

def LSTM_predict_proba(model,pipeline,X):
    X_transformed = pipeline.transform(X)
    return model.predict_proba(X_transformed)[:,1]