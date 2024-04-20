import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
import xgboost as xgb
import pickle
import joblib

def En_test_train_validation_split(validation_df, test_train_df, target_variable='fire', test_proportion=0.33):
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

def load_constituent_models(*model_paths):
    """
    Load any number of models from their model_paths.
    """
    # Load functions for respective libraries
    def load_h5(model_path):
        return tf.keras.models.load_model(model_path)
    def load_pickle(model_path):
        with open(model_path, 'rb') as file:
            return pickle.load(file)
    def load_joblib(model_path):
        return joblib.load(model_path)
    def load_json(model_path):
        model = xgb.Booster()
        model.load_model(model_path)
        return model
    
    # Map extensions to their corresponding load functions
    load_functions = {
        '.h5': load_h5,
        '.pkl': load_pickle,
        '.joblib': load_joblib,
        '.json': load_json
    }

    # Initialize dictionary to hold loaded models
    models = {}

    # Iterate over each model path
    for model_path in model_paths:
        # Get the file extension
        _, ext = os.path.splitext(model_path)

        # Get the correct load function based on the file extension
        load_function = load_functions.get(ext)
        if load_function:
            # Load the model and store it in the dictionary
            models[model_path] = load_function(model_path)
        else:
            raise ValueError(f"No load function available for files with extension: {ext}")

    return models

def load_constituent_pipelines(*pipeline_paths):
    """
    Loads any number of pipelines from their paths. Pipelines for constituent models should be fitted to appropriate training data before saving
    """
    # Functions to load based on file type
    def load_pickle(model_path):
        with open(model_path, 'rb') as file:
            return pickle.load(file)
    def load_joblib(model_path):
        return joblib.load(model_path)

    load_functions = {
        '.pkl': load_pickle,
        '.joblib': load_joblib
    }

    # Initialize dictionary to hold loaded pipelines
    pipelines = {}

    # Iterate over each pipeline path
    for pipeline_path in pipeline_paths:
        # Get the file extension
        _, ext = os.path.splitext(pipeline_path)

        # Get the correct load function based on the file extension
        load_function = load_functions.get(ext)
        if load_function:
            # Load the model and store it in the dictionary
            pipelines[pipeline_path] = load_function(pipeline_path)
        else:
            raise ValueError(f"No load function available for files with extension: {ext}")

    return pipelines

class ModelPredictor:
    """
    Bundles model pipeline and predict functions and transforms prediction output into pandas dataframe
    """
    def __init__(self, model, pipeline, predict_function, name):
        self.model = model
        self.pipeline = pipeline
        self.predict_function = predict_function
        self.name = name

    def predict(self, X):
        predictions = self.predict_function(self.model, self.pipeline, X)
        return pd.DataFrame(predictions, columns=[f'{self.name}_Prediction']).astype('float64')

def plot_roc_curve(y_true, probabilities, point_thresholds=None, point_tprs=None):
    """
    Plots the ROC curve for given probabilities and marks points specified by thresholds or TPRs,
    with legends indicating threshold values and corresponding TPR.
    
    Args:
        y_true (array): True binary labels.
        probabilities (array): Probabilities of the positive class.
        point_thresholds (list): List of thresholds for which points to mark on the ROC curve.
        point_tprs (list): List of TPRs for which points to mark on the ROC curve.
    """
    # Calculate ROC curve points
    fpr, tpr, thresholds = roc_curve(y_true, probabilities)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')

    # Collect legend information
    legend_handles = [plt.Line2D([0], [0], color='darkorange', lw=2)]
    legend_labels = [f'ROC curve (area = {roc_auc:.2f})']

    # Handle point_thresholds
    if point_thresholds is not None:
        for thresh in point_thresholds:
            idx = np.argmin(np.abs(thresholds - thresh))
            plt.plot(fpr[idx], tpr[idx], 'ro')
            plt.vlines(x=fpr[idx], ymin=0, ymax=tpr[idx], color='grey', linestyle='--')
            plt.hlines(y=tpr[idx], xmin=0, xmax=fpr[idx], color='grey', linestyle='--')
            print(f"Threshold: {thresh:.8f}, TPR: {tpr[idx]:.4f}, FPR: {fpr[idx]:.4f}")
            legend_handles.append(plt.Line2D([0], [0], color='red', marker='o', linestyle=''))
            legend_labels.append(f'Point (Thresh={thresh:.3f}, TPR={tpr[idx]:.3f})')

    # Handle point_tprs
    if point_tprs is not None:
        for t in point_tprs:
            idx = np.argmin(np.abs(tpr - t))
            plt.plot(fpr[idx], tpr[idx], 'bo')
            plt.vlines(x=fpr[idx], ymin=0, ymax=tpr[idx], color='grey', linestyle='--')
            plt.hlines(y=tpr[idx], xmin=0, xmax=fpr[idx], color='grey', linestyle='--')
            print(f"Threshold: {thresholds[idx]:.8f}, TPR: {t:.4f}, FPR: {fpr[idx]:.4f}")
            legend_handles.append(plt.Line2D([0], [0], color='blue', marker='o', linestyle=''))
            legend_labels.append(f'Point (Thresh={thresholds[idx]:.3f}, TPR={t:.3f})')

    # Add legend to plot
    plt.legend(legend_handles, legend_labels, loc="lower right")
    plt.show()

