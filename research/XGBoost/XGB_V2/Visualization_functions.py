import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from sklearn.metrics import roc_curve, auc, confusion_matrix

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

def plot_multiple_ROC(y_true, y_prediction_proba, names):
    """
    Plot ROC curves for multiple predictions on the same plot.

    Parameters:
    - y_true: array-like of shape (n_samples,) - True binary labels.
    - y_prediction_proba: list of array-like predictions from different models, 
                          each of shape (n_samples,).
    - names: list of strings, names corresponding to each prediction in y_prediction_proba.
    
    Each element in y_prediction_proba corresponds to a prediction array from a model,
    and each element in names is the name of the model used in the legend.
    """
    plt.figure(figsize=(10, 8))
    for y_pred, name in zip(y_prediction_proba, names):
        # Compute ROC curve and ROC area for each class
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        # Plot the ROC curve
        plt.plot(fpr, tpr, label=f'{name}: AUC = {roc_auc:.2f}')
    
    plt.plot([0, 1], [0, 1], 'k--')  # Plot the diagonal 45 degree line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

def get_cm(y_true, y_predict, save_path):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_predict)
    
    # Calculate the percentage of each value
    cm_sum = np.sum(cm)
    cm_percentage = cm / cm_sum * 100

    # Create annotation labels
    # Combine count and percentage in the annotation, formatted for display
    annot_labels = (np.asarray(["{}\n({:.2f}%)".format(count, percentage)
                                for count, percentage in zip(cm.flatten(), cm_percentage.flatten())])
                    .reshape(cm.shape))
    
    # Plotting the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=annot_labels, fmt='', cmap='Blues', cbar=True)  # Using '' for fmt as we're passing strings in annot_labels
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    
    # Save the figure
    plt.savefig(save_path)
    plt.close()  # Close the plot to free memory if generating multiple plots

## Functions for plotting predictions ##
def add_back_original_features(X_processed,df_original):
    """
    Adds removed features from the original dataframe back to a processed dataframe
    """
    non_overlapping_columns = [col for col in df_original.columns if col not in X_processed.columns]
    df_right_selected = df_original[non_overlapping_columns]
    X_combined = pd.merge(X_processed, df_right_selected, left_index=True, right_index=True, how='left')
    return X_combined

def prepare_df_for_plotting(X,y_pred,df):
    """
    X is the test inputs, y_pred is the resulting model predictions
    df is the original dataframe to add back date, latitude, and longitude columns
    """
    df = add_back_original_features(X,df)
    df['prediction'] = y_pred
    # Classify each prediction
    df['result'] = 'TN'
    df.loc[(df['fire'] == 1) & (df['prediction'] == 1), 'result'] = 'TP'
    df.loc[(df['fire'] == 0) & (df['prediction'] == 1), 'result'] = 'FP'
    df.loc[(df['fire'] == 1) & (df['prediction'] == 0), 'result'] = 'FN'

    return df

def plot_daily_predictions(df,save_dir):
    """
    Plots and saves each plot as a png into save_dir.
    Use prepare_df_for_plotting to get prediction column (true negative, true positive, etc.)
    Parameters:
        df = dataframe with a column that classifies prediction type (TN, TP, FP, FN)
        save_dir = path to directory to save daily images
    """
    os.makedirs(save_dir,exist_ok=True)
    df['date'] = pd.to_datetime(df['date']).dt.date
    # Define the geographic bounds of Alberta
    extent = [-120, -110, 49, 60]  # [west, east, south, north]
    provinces = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces_lines',
            scale='50m',
            facecolor='none',
            edgecolor='black'
        )

    for date in df['date'].unique():
        df_date = df[df['date'] == date]

        fig = plt.figure(figsize=(10, 10))
        # Define the projection and extent
        ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-115, central_latitude=55))
        ax.set_extent(extent)
        
        # Add geographic features
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.LAKES, alpha=0.5)
        ax.add_feature(provinces, linestyle='-', edgecolor='black')

        # Plot points for the given date
        categories = ['TP', 'TN', 'FP', 'FN']
        colors = ['green', 'blue', 'red', 'yellow']
        for category, color in zip(categories, colors):
            df_cat = df_date[df_date['result'] == category]
            plt.scatter(df_cat['longitude'], df_cat['latitude'], c=color, label=category, alpha=0.6, transform=ccrs.PlateCarree())
        
        plt.title(f'Fire Prediction Results for {date}')
        plt.legend(loc='lower left')
        
        # Save the plot
        plt.savefig(os.path.join(save_dir, f'{date}_predictions.png'))
        plt.close(fig)

def plot_incorrect_predictions(df, save_path):
    """
    Plots a heatmap of incorrect predictions over a dataset.
    Use prepare_df_for_plotting to get prediction column (true negative, true positive, etc.)
    Parameters:
        df = dataframe with prediction column that classifies prediction type (TP, TN, FP, FN)
        save_path = path to save the heatmap image
    """
    # Create a DataFrame of all unique latitude and longitude pairs with zero counts
    all_coords = df[['latitude', 'longitude']].drop_duplicates()
    all_coords['counts'] = 0

    # Filter to include only incorrect predictions
    incorrect_df = df[(df['result'] == 'FP') | (df['result'] == 'FN')]
    incorrect_counts = incorrect_df.groupby(['latitude', 'longitude']).size().reset_index(name='counts_fn')

    # Merge the counts back to all coordinates
    agg_df = all_coords.merge(incorrect_counts, on=['latitude', 'longitude'], how='left')
    agg_df['counts'] = agg_df['counts_fn'].fillna(0) + agg_df['counts']
    agg_df.drop(columns=['counts_fn'], inplace=True)

    # Plotting
    fig = plt.figure(figsize=(40, 40))
    ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-115, central_latitude=55))
    ax.set_extent([-120, -110, 49, 60])

    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines', scale='50m', facecolor='none', edgecolor='black'), linestyle='-')

    norm = plt.Normalize(vmin=agg_df['counts'].min(), vmax=agg_df['counts'].max())
    scatter = ax.scatter(agg_df['longitude'], agg_df['latitude'], c=agg_df['counts'], cmap='Reds', norm=norm, edgecolor='k', linewidth=0.5, alpha=0.7, s=600, transform=ccrs.PlateCarree())
    cbar = plt.colorbar(scatter, shrink=0.5, aspect=5)
    cbar.set_label('Number of Incorrect Predictions', fontsize=40)
    cbar.ax.tick_params(labelsize=30)

    plt.title('Incorrect Predictions Across Alberta', fontsize=40)
    plt.savefig(save_path)
    plt.close(fig)

def plot_false_positives(df, save_path):
    """
    Same as plot_incorrect_predictions except plots false positives rather than incorrect predictions
    Plots a heatmap of incorrect predictions over a dataset.
    Use prepare_df_for_plotting to get prediction column (true negative, true positive, etc.)
    Parameters:
        df = dataframe with prediction column that classifies prediction type (TP, TN, FP, FN)
        save_path = path to save the heatmap image
    """
    # Create a DataFrame of all unique latitude and longitude pairs with zero counts
    all_coords = df[['latitude', 'longitude']].drop_duplicates()
    all_coords['counts'] = 0

    # Filter to include only false positive predictions
    fp_df = df[df['result'] == 'FP']
    fp_counts = fp_df.groupby(['latitude', 'longitude']).size().reset_index(name='counts_fn')

    # Merge the counts back to all coordinates
    agg_df = all_coords.merge(fp_counts, on=['latitude', 'longitude'], how='left')
    agg_df['counts'] = agg_df['counts_fn'].fillna(0) + agg_df['counts']
    agg_df.drop(columns=['counts_fn'], inplace=True)

    # Plotting
    fig = plt.figure(figsize=(40, 40))
    ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-115, central_latitude=55))
    ax.set_extent([-120, -110, 49, 60])  # Geographic bounds for Alberta

    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines', scale='50m', facecolor='none', edgecolor='black'), linestyle='-')

    norm = plt.Normalize(vmin=agg_df['counts'].min(), vmax=agg_df['counts'].max())
    scatter = ax.scatter(agg_df['longitude'], agg_df['latitude'], c=agg_df['counts'], cmap='Reds', norm=norm, edgecolor='k', linewidth=0.5, alpha=0.7, s=600, transform=ccrs.PlateCarree())
    cbar = plt.colorbar(scatter, shrink=0.5, aspect=5)
    cbar.set_label('Number of False Positives', fontsize=40)
    cbar.ax.tick_params(labelsize=30)

    plt.title('False Positive Predictions Across Alberta', fontsize=40)
    plt.savefig(save_path)
    plt.close(fig)

def plot_false_negatives(df, save_path):
    """
    Variant of plot_incorrect_predictions for looking at false negatives specifically.
    """
    # Create a DataFrame of all unique latitude and longitude pairs
    all_coords = df[['latitude', 'longitude']].drop_duplicates()
    all_coords['counts'] = 0  # Initialize counts to zero

    # Filter to include only false negative predictions
    fn_df = df[df['result'] == 'FN']

    # Group by latitude and longitude, count occurrences
    fn_counts = fn_df.groupby(['latitude', 'longitude']).size().reset_index(name='counts')

    # Merge the counts back to all coordinates
    agg_df = all_coords.merge(fn_counts, on=['latitude', 'longitude'], how='left', suffixes=('', '_fn'))
    agg_df['counts'] = agg_df['counts_fn'].fillna(0) + agg_df['counts']
    agg_df.drop(columns=['counts_fn'], inplace=True)

    # Plotting
    fig = plt.figure(figsize=(40, 40))
    ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-115, central_latitude=55))
    ax.set_extent([-120, -110, 49, 60])  # Geographic bounds for Alberta

    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines', scale='50m', facecolor='none', edgecolor='black'), linestyle='-')

    norm = plt.Normalize(vmin=agg_df['counts'].min(), vmax=agg_df['counts'].max())
    scatter = ax.scatter(agg_df['longitude'], agg_df['latitude'], c=agg_df['counts'], cmap='Blues', norm=norm, edgecolor='k', linewidth=0.5, alpha=0.7, s=600, transform=ccrs.PlateCarree())
    cbar = plt.colorbar(scatter, shrink=0.5, aspect=5)
    cbar.set_label('Number of False Negatives', fontsize=40)
    cbar.ax.tick_params(labelsize=30)

    plt.title('False Negative Predictions Across Alberta', fontsize=40)
    plt.savefig(save_path)
    plt.close(fig)

## High Level Function for getting all visualizations ##
def generate_visualizations(X_processed,y_predict,y_true,df_original,save_path):
    """
    Generates visualizations for model predictions
    """
    os.makedirs(save_path,exist_ok=True)
    # Add features back to prediction dataset
    df = prepare_df_for_plotting(X_processed,y_predict,df_original)
    
    # Prepare and save plots
    get_cm(y_true,y_predict,os.path.join(save_path,'cm.png'))
    plot_incorrect_predictions(df, os.path.join(save_path,'incorrect_predictions.png'))
    plot_false_positives(df, os.path.join(save_path,'false_positives.png'))
    plot_false_negatives(df, os.path.join(save_path,'false_negatives.png'))
    #plot_daily_predictions(df,os.path.join(save_path,'validation_predictions'))
