import os
import pandas as pd
from sklearn.model_selection import train_test_split

## Data Loading and Split ##
def load_downsampled_df(PROJECT_ROOT):
    """
    loads project_root/data/processed/downsampled_df.csv
    Converts date column to datetime
    """
    path = os.path.join(PROJECT_ROOT,'data','processed','downsampled_df.csv')
    main_df = pd.read_csv(path,index_col=0)
    main_df['date'] = pd.to_datetime(main_df['date'])
    return main_df

def load_full_df(PROJECT_ROOT):
    """
    loads project_root/data/processed/processed_wildfire_ERA5_FWI.csv
    Performs basic preprocessing on this data to prepare for prediction
    """
    path = os.path.join(PROJECT_ROOT,'data','processed','processed_wildfire_ERA5_FWI.csv')
    main_df = pd.read_csv(path,index_col=0)
    main_df['date'] = pd.to_datetime(main_df['date'])
    main_df.rename(columns={'latitude_ERA5': 'latitude', 'longitude_ERA5': 'longitude'},inplace=True)
    unnamed_cols = [col for col in main_df.columns if col.startswith('Unnamed:')]
    main_df.drop(columns=unnamed_cols, inplace=True)
    main_df['type_of_high_vegetation'] = main_df['type_of_high_vegetation'].astype(int)
    main_df['type_of_low_vegetation'] = main_df['type_of_low_vegetation'].astype(int)
    return main_df

def get_train_validation_df(df,date_split=pd.Timestamp('2019-01-01')):
    """
    Splits the dataframe at a specific point in time to create training and validation sets.
    Input df needs to have 'date' column of type datetime
    """
    validation_df = df[df['date'] > date_split]
    test_train_df = df[df['date'] < date_split]
    return validation_df, test_train_df

def test_train_validation_split(validation_df,test_train_df,target_variable='fire',test_proportion=0.33):
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
####

## Data Processing and Feature Engineering ##
def extract_day_of_year(X):
    day_of_year = X['date'].dt.dayofyear.to_frame(name='day_of_the_year')
    return day_of_year
####

