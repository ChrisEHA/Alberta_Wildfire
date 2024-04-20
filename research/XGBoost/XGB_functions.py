## Functions for SVM Deployment ##
from sklearn.model_selection import train_test_split
import xgboost as xgb

def ANN_test_train_validation_split(validation_df,test_train_df,target_variable='fire',test_proportion=0.33):
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

def extract_day_of_year(X):
    day_of_year = X['date'].dt.dayofyear.to_frame(name='day_of_the_year')
    return day_of_year

def XGB_predict_proba(model,pipeline,X):
    X_transformed = pipeline.transform(X)
    dX = xgb.DMatrix(X_transformed)
    return model.predict(dX)