## Libraries General ##
import os
import glob
import io
import random

import numpy as np
import pandas as pd
import xarray as xr
####

## Libraries for google drive ##
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
####

## Common functions ##
def haversine_distances(X, Y):
    """
    Vectorized approach for haversine distance between multiple points
    Parameters:
        X and Y are numpy arrays of latitude longitude
    Returns:
        distances is a matrix where each row corresponds to an entry in X and each column corresponds to its distance from each point in Y.
        output is in km
    """
    R = 6371.0  # Radius of the Earth in kilometers
    
    # Prepare data for calculation
    lat_X, lon_X = np.radians(X[:, 0]), np.radians(X[:, 1])
    lat_Y, lon_Y = np.radians(Y[:, 0]), np.radians(Y[:, 1])

    dlat = lat_X[:, np.newaxis] - lat_Y[np.newaxis, :]
    dlon = lon_X[:, np.newaxis] - lon_Y[np.newaxis, :]
    
    # calculate haversine
    a = np.sin(dlat / 2)**2 + np.cos(lat_X)[:, np.newaxis] * np.cos(lat_Y) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    distances = R * c
    return distances
####

## ERA5 Download Functions ##
def ERA5_download_save_yearly(output_dir,excludeVariables,timeStart,timeEnd,zarr_path='gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3/',chunking={'time': 5}):
    """
    Downloads select data from zarr_path after applying aggregation. Creates yearly netCDF files from the zarr data.
    Parameters:
        output_dir = directory to save the data for each year
        excludeVariables = list of variables to exclude that are in the original zarr
        timeStart = string of the starting time in the format yyyy-mm-dd. i.e., '2000-01-01'
        timeEnd = end time in the format yyyy-mm-dd
        zarr_path = path to the zarr file. Needs to be zarr data. The default it ERA5 data. 
        chunking = chunking strategy for working on zarr data. The default was obtained from testing for a specific machine
    Outputs:
        The function returns nothing. It saves relevant data from the zarr to the output_dir
    
    March 2024
    """    

    os.makedirs(output_dir, exist_ok=True)
    
    ## Check if the output dir is empty. If not, return with a warning and stop execution. ##
    if len(os.listdir(output_dir)) != 0:
        print(f'{output_dir} is not empty. Empty the directory to redownload ERA5 data.')
        return
    ####
    # If the output dir it empty or doesn't exist, proceed to download the data

    # Open the Zarr dataset without loading it into memory
    ERA5_ds = xr.open_zarr(
        zarr_path,
        drop_variables=excludeVariables,
        chunks=chunking
    )

    # Convert time to integers to use in a range
    start_year, end_year = int(timeStart[:4]), int(timeEnd[:4])

    for year in range(start_year, end_year + 1):
        # Crop dataset for the specific year and roughly in space (Alberta area)
        ERA5_year_ds = ERA5_ds.sel(time=str(year)).sel(latitude=slice(60, 49), longitude=slice(240, 250))
        
        # ERA5 contains multiple levels for various altitudes. Aggregate to the max of these.
        ERA5_year_ds = ERA5_year_ds.max(dim='level').resample(time='D').max()
        
        # Load the processed yearly dataset into RAM
        ERA5_year_ds = ERA5_year_ds.load()
        
        # Save the yearly dataset to a NetCDF file
        output_file_path = os.path.join(output_dir, f'ERA5_Alberta_{year}.nc')
        ERA5_year_ds.to_netcdf(output_file_path)
        print(f'Saved {output_file_path}')

def ERA5_save_full(yearly_netCDF_dir,output_path):
    """
    Saves yearly netCDF files in path into a singular netCDF. All yearly netCDF files should have the same variables and latitude/longitude for xr.concat()
    Parameters:
        yearly_netCDF_dir = path to directory that contains yearly netCDF files
        output_path = path to singular output netCDF file
    Outputs:
        The function returns nothing. A singular netCDF file is created and saved.
    
    March 2024
    """
    
    files = glob.glob(f"{yearly_netCDF_dir}/*.nc")

    # Load and process each file
    datasets = [xr.open_dataset(file) for file in files]

    # Concatenate all processed datasets into a single dataset
    combined_ds = xr.concat(datasets, dim='time')

    combined_ds.to_netcdf(output_path)
####

## FWI Download Functions ##
def download_file_from_google_drive(file_id, local_filename,SCOPES = ['https://www.googleapis.com/auth/drive.readonly']):
    """
    Download a file from Google Drive by file ID. If the file already exists, it will be overwritten. Used in FWI_process_and_save_year function
    """
    if os.path.exists(local_filename):
        os.remove(local_filename)
    
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    
    service = build('drive', 'v3', credentials=creds)
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    
    with open(local_filename, 'wb') as f:
        fh.seek(0)
        f.write(fh.read())
    return local_filename

def FWI_process_and_save_year(file_id, variable, year, temp_dir, processed_dir):
    """
    Downloads and processed the file in file_id from google drive and saves based off variable and year.
    """
    local_filename = os.path.join(temp_dir, f"{variable}_{year}.nc")
    download_file_from_google_drive(file_id, local_filename)
    
    ds = xr.open_dataset(local_filename, engine='h5netcdf', chunks={'time': 20})
    
    # FWI inputs and outputs have different capitalization for latitude and longitude.
    try:
        sliced_ds = ds.sel(Latitude=slice(60, 49), Longitude=slice(240, 250))
    except:
        sliced_ds = ds.sel(latitude=slice(60, 49), longitude=slice(240, 250))

    processed_filename = os.path.join(processed_dir, f"processed_{variable}_{year}.nc")
    sliced_ds.to_netcdf(processed_filename, engine='h5netcdf')

def link_google_drive(creds_file="mycreds.txt"):
    """
    Sets up connection to google drive. Requires a credentials file and a token (will prompt to set up token the first time)
    Parameters:
        creds_file = path to credentials file from google API
    Outputs:
        drive = Google Drive connection
    
    March 2024
    """
    gauth = GoogleAuth()
    gauth.LoadCredentialsFile(creds_file)
    if gauth.credentials is None:
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        gauth.Refresh()
    else:
        gauth.Authorize()
    gauth.SaveCredentialsFile(creds_file)

    drive = GoogleDrive(gauth)
    return drive

def FWI_download_process_save(file_metadata,variables,years,temp_dir="temp_downloads",processed_dir="processed"):
    """
    Downloads all the files matching the input variables and year range from the linked google drive. Does not check for previously downloaded
    files and overwrites any previous files of the same variable and year. Downloads full netCDF and saves to temp_dir. Then, performs slicing and saves
    the processed netCDF to processed_dir.
    Parameters:
        file_metadata = file metadata from the google drive.
        variables = list of variables to download. Should be as they appear in file names in the google drive
        years = range of years to download. i.e., range(2000,2021)
        temp_dir = path to dir where temporary downloads are performed. Can delete these files after processing is complete.
        processed_dir = path to store processed yearly variable netCDFs
    Outputs:
        Function returns nothing. Saves netCDF files cropped to Alberta for passed variables and years to processed_dir.
    
    March 2024
    """
    # Make download dirs if needed
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # Loop through each file and download then process
    for metadata in file_metadata:
        for variable in variables:
            for year in years:
                if f"{variable}_{year}.nc" in metadata['title']:
                    print(f"Processing: {variable}_{year}.nc with file_id: {metadata['id']}")
                    FWI_process_and_save_year(metadata['id'], variable, year, temp_dir, processed_dir)

def load_and_process_file_FWIoutputs(file_path):
    """
    Converts a variable_year.nc file downloaded from FWI to an xarray dataset.
    Renames coordinates to be consistent with analysis datasets.
    """
    # Extract the year from the filename assuming format: '..._year.nc'
    year = int(file_path.split('_')[-1].split('.')[0])
    
    # Open the dataset
    ds = xr.open_dataset(file_path)
    
    # Convert Time to datetime and rename coordinates
    dates = pd.to_datetime(ds['Time'].values - 1, unit='D', origin=pd.Timestamp(year=year, month=1, day=1))
    ds = ds.assign_coords(time=dates).drop_vars('Time').rename({'Time': 'time', 'Latitude': 'latitude', 'Longitude': 'longitude'})
    ds = ds.drop_vars('crs')

    return ds

def load_and_process_file_FWIinputs(file_path):
    """
    Converts a variable_year.nc file downloaded from FWI to an xarray dataset.
    Renames coordinates to be consistent with analysis datasets.
    """
    # Extract the year from the filename assuming format: '..._year.nc'
    year = int(file_path.split('_')[-1].split('.')[0])
    
    # Extract the variable name from the filename
    # This assumes the format 'processed_variable_name_year.nc'
    variable_name = "_".join(file_path.split('\\')[-1].split('_')[1:-1])
    
    # Open the dataset
    ds = xr.open_dataset(file_path)
    
    # Convert 'z' coordinate (day of the year) to datetime objects and rename to 'time'
    dates = pd.to_datetime(ds['z'].values - 1, unit='D', origin=pd.Timestamp(year=year, month=1, day=1))
    ds = ds.assign_coords(time=dates).drop_vars('z').rename({'z': 'time'})

    # Rename the variable to that in the filename. Assuming variable is the name of the data variable
    ds = ds.rename({'variable': variable_name})
    ds = ds.drop_vars('crs')

    return ds

def FWI_save_combined(processing_function,inputs_dir,output_path,years=range(2000,2020)):
    """
    Combines all netCDF files in the inputs_dir into one netCDF to output_path. All netCDF should have the same coordinate range.
    First, creates yearly merged datasets for all the variables. Then, concatenates the yearly datasets to create one netCDF.
    Parameters:
        processing_function = the function used to process the netCDF. Different for FWI intputs and outputs
        inputs_dir = directory containing yearly processed netCDF for each variable
        output_path = path to save final combined netCDF
        years = range that should encompass all years for combining
    Outputs:
        The function returns nothing. A singular netCDF is saved to output_path.
    
    March 2024.
    """
    # Recurring code for this function. Defined nested function for readability
    def merge_yearly_datasets(year, file_paths):
        datasets = [processing_function(fp) for fp in file_paths if str(year) in fp]
        if datasets:
            merged_yearly_ds = xr.merge(datasets)
            return merged_yearly_ds
        else:
            return None

    files = glob.glob(f"{inputs_dir}/*.nc")

    # Create yearly datasets with all variables merged
    yearly_datasets = []
    for year in years:
        yearly_ds = merge_yearly_datasets(year, files)
        if yearly_ds is not None:
            yearly_datasets.append(yearly_ds)

    # Concatenate all yearly merged datasets
    combined_ds = xr.concat(yearly_datasets, dim='time')

    combined_ds.to_netcdf(output_path)
####

## Data merging and point selection functions ##
def map_to_ds(df, ds):
    """
    Adds the closest ERA5 latitude and longitude to each row in the dataframe based on haversine distances
    and returns a list of unique (latitude_ERA5, longitude_ERA5) pairs.

    Parameters:
    - df: A pandas DataFrame with columns 'latitude' and 'longitude'.
    - ds: An xarray Dataset with latitude and longitude coordinates. Longitudes should be in 0-360 range

    Returns:
    - Tuple: Modified DataFrame and list of unique (latitude_ERA5, longitude_ERA5) pairs.
    """

    # Ensure dataframe longitudes are in the 0 to 360 range
    df['longitude'] = df['longitude'] % 360

    # Create list of latitude/longitude pairs to calculate haversine distance
    X = df[['latitude', 'longitude']].to_numpy()
    Y = np.array([(lat, lon) for lat in ds.latitude.values for lon in ds.longitude.values])
    
    distances = haversine_distances(X, Y) # Matrix of distances
    
    # Find smallest distance and get indeces for xarray ds grid format
    min_indices = np.argmin(distances, axis=1)
    lat_shape, lon_shape = ds.latitude.shape[0], ds.longitude.shape[0]
    min_lat_indices = min_indices // lon_shape
    min_lon_indices = min_indices % lon_shape
    
    # Create new columns in df for the ERA grid point
    df['latitude_ERA5'] = ds.latitude.values[min_lat_indices]
    df['longitude_ERA5'] = ds.longitude.values[min_lon_indices]

    # Create a list of unique (latitude_ERA5, longitude_ERA5) pairs
    lat_lon_pairs = list(set(zip(df['latitude_ERA5'], df['longitude_ERA5'])))

    return df, lat_lon_pairs

def combine_ds(*datasets):
    """
    Combine multiple xarray Datasets into one Dataset.
    
    Parameters:
    *datasets : an arbitrary number of xarray.Dataset objects. Dimensions latitude and longitude should match in all datasets
    
    Returns:
    combined_ds : xarray.Dataset
        The combined dataset containing variables from all input datasets.
    """
    combined_ds = xr.merge(datasets)
    
    return combined_ds

def create_master_df(df,ds):
    """
    Takes the raw wildfire df and combines it with the processed combined ERA5, FWI dataset. Creates one master df with all the data.
    Parameters:
        df = dataframe. Maps latitude, longitude colums to ds spatial points. Needs to have a date column.
        ds = dataset combine with df
    Outputs:
        merged_df = a singular dataframe of the merged data from df and ds
    
    Apr. 2024
    """
    # Map each fire occurrence to the closest ds point and create new columns in the df for this point. lat_lon_pairs is a list of ds lat/lon points that have a fire associated with them.
    df, lat_lon_pairs = map_to_ds(df,ds)

    # Convert the df to a dataset and set coordinates
    df_xarray = ds.to_dataframe().reset_index()
    df_xarray.rename(columns={'latitude': 'latitude_ERA5','longitude':'longitude_ERA5','time':'date'},inplace=True)

    # Remove latitude/longitudes not in lat_lon_pairs
    mask = df_xarray.apply(lambda row: (row['latitude_ERA5'], row['longitude_ERA5']) in lat_lon_pairs, axis=1)
    df_xarray = df_xarray[mask]

    merged_df = pd.merge(df, df_xarray, on=['longitude_ERA5', 'latitude_ERA5', 'date'], how='outer')

    return merged_df

def merge_ERA5_FWI_wildfire(ERA5_path,FWI_input_path,FWI_output_path,wildfire_csv_path,output_path,wildfire_variables,timeStart,timeEnd):
    """
    Combines the ERA5, FWI, and raw wildfire data into a singular csv.
    Parameters:
        ERA5_path = path to combined processed ERA5 netCDF
        FWI_input_path = path to combined processed FWI inputs netCDF
        FSI_output_path = path to combined processed FWI outputs netCDF
        wildfire_csv_path = path to wildfire occurrence csv
        output_path = path to save final combined csv
        wildfire_variables = variables from the wildfire csv to carry to the combined csv
        timeStart = start time str 'yyyy-mm-dd' for temporal cropping of the wildfire data
        timeEnd = end time str 'yyyy-mm-dd' for temporal cropping of the wildfire data
    Outputs:
        df = merged df that contains wildfire, ERA5, FWI data.
        Saves the combined df to output_path as csv
    
    Apr. 2024
    """
    # Load wildfire df and perform basic preprocessing
    wildfire_df = pd.read_csv('processed_wildfire_df.csv')
    wildfire_df = wildfire_df.loc[:,wildfire_variables]
    wildfire_df['date'] = pd.to_datetime(wildfire_df['date'])
    wildfire_df['date'] = pd.to_datetime(wildfire_df['date'].dt.date)
    wildfire_df = wildfire_df[(wildfire_df['date'] >= pd.to_datetime(timeStart)) & (wildfire_df['date'] <= pd.to_datetime(timeEnd))]
    
    ERA5_ds = xr.load_dataset(ERA5_path)
    FWIin_ds = xr.load_dataset(FWI_input_path)
    FWIout_ds = xr.load_dataset(FWI_output_path)

    ds_allVariables = combine_ds(ERA5_ds, FWIin_ds, FWIout_ds)
    df = create_master_df(wildfire_df,ds_allVariables)
    df.to_csv(output_path)

    return df
####

## Data cleaning and feature lagging functions ##
def create_lag_features(df,target_variable,predict_shifts,variable_lags,variables_to_lag,historical_fire_count_days):
    """
    Performs feature lagging and nan removal on df.
    Parameters:
        df = combined dataframe that contains all variables of interest. 'fire_type' is a necessary column and will be used to determine fire occurrences
        target_variable = what to name the target variable column
        predict_shifts = Shift the fire occurrence date for target prediction. Can be a list, creates a new column for each
        variable_lags = Number of days to lag target variables. Creates new column for each. Can be a list of multiple lags
        variables_to_lag = List of variable names to lag using variable_lags
        historical_fire_count_days = Creates a column for each. Number of fire that occurred in the past x days
    Outputs:
        result_df_list = a list of pointwise (lat/lon pairs) dataframes that contain lagged variables.
    """
    # Get list of points that contained a fire
    temp_df = df[['latitude_ERA5', 'longitude_ERA5']].drop_duplicates().copy()
    lat_lon_pairs = list(zip(temp_df['latitude_ERA5'], temp_df['longitude_ERA5']))
    del temp_df

    # Create a new column that records whether or not a fire occurred
    df[target_variable] = np.where(df['fire_type'].isna(), 0, 1)

    result_df_list = []
    # Loop through each of the spatial latitude/longitude pairs
    for point in lat_lon_pairs:
        latitude, longitude = point

        # Filter the DataFrame for the current point
        point_df = df[(df['latitude_ERA5'] == latitude) & (df['longitude_ERA5'] == longitude)].copy()

        # Sum fire occurrences for the
        for days_historic in historical_fire_count_days:
            feature_name = 'fire_count_past_' + str(days_historic) + 'Days'
            point_df[feature_name] = point_df[target_variable].rolling(days_historic).sum().shift(1).fillna(0)

        # Perform lagging on the filtered DataFrame
        for feature in variables_to_lag:
            for lag in variable_lags:
                lagged_name = feature + "_" + str(lag) + "dayLag"
                point_df[lagged_name] = point_df[feature].shift(lag)

        # Shift target variable
        for Tshift in predict_shifts:
            target_shift_name = target_variable + "_" + str(Tshift) + "dayShift"
            point_df[target_shift_name] = point_df[target_variable].shift(-Tshift)

        # Combine the results in a list
        result_df_list.append(point_df)

    return result_df_list

def save_df_list(result_df_list,output_path):
    """
    Takes a list and turns it into a dataframe. Removes rows that contain nan and saves to output_path.
    Tailored for lagged feature list created by create_lag_features on the raw wildfire data.
    """
    # Concatenate all DataFrame slices at once
    result_df = pd.concat(result_df_list)
    del result_df_list

    # Remove nan and irrelevant wildfire_df columns
    result_df.sort_index(inplace=True)
    result_df.drop(columns={'general_cause', 'fire_type', 'latitude', 'longitude'},inplace=True)
    result_df.dropna(inplace=True)

    result_df.to_csv(output_path)
####

## Point selection functions ##
def df_extractDate(df,datetime_column,extract_date):
    """
    Return a dataframe containing only entries with dates matching the extraction date
    """
    return df[df[datetime_column] == extract_date]

def select_points(df,target_variable,lat_lon_pairs,nPoints_noFireDays,dist_from_fire,date,nonFire_factor):
    """
    Select daily sample points
    All fire entries in df for this date are considered
    df needs to already be filtered by date. Use df_extractDate() to get the filtered df
    Inputs:
        df = date filtered dataframe
        lat_lon_pairs = tuple list of ERA5 grid points for the analysis
        nPoints = min number of points to include for each day
        dist_from_fire = minimum distance (km) that a point needs to be from a fire point to be considered for random selection
        date = date for point selection. Used for random seed generation
    Output:
        List of (latitude, lontigude) tuples of interest for a date
    """
    # Collect fire occurrence points
    fires_df = df[df[target_variable] == 1]
    date_points = list(zip(fires_df['latitude'], fires_df['longitude']))

    ## Add non-fire points to the daily sample ##
    # Set a random seed based on fire occurrence locations (date_points) and date of interest (date)
    seed_value = int(sum(sum(tup) for tup in date_points) + int(date.strftime('%Y%m%d')))
    random.seed(seed_value)

    nFires = len(date_points)
    print(f'\tNumber of fires {nFires}')
    # If there are fires, filter out points that are within dist_from_fire then select non-fire points based on factor (factor non-fire/fire points)
    if nFires > 0:
        distances = haversine_distances(np.array(lat_lon_pairs), np.array(date_points))
        filterI = np.where((distances < dist_from_fire).any(axis=1))[0]
        
        # Use list comprehension to exclude the filtered indices
        lat_lon_pairs = [pair for i, pair in enumerate(lat_lon_pairs) if i not in filterI]

        # Select non-fire points
        nAdd = min(len(lat_lon_pairs),int(nFires*nonFire_factor))
        if len(lat_lon_pairs) > 0:  # Ensure lat_lon_pairs is not empty
            new_Points = random.sample(lat_lon_pairs, nAdd)
        else:
            new_Points = []

    else: # If there are no fires, choose nPoints_noFireDays
        nAdd = min(len(lat_lon_pairs),nPoints_noFireDays)
        if len(lat_lon_pairs) > 0:  # Ensure lat_lon_pairs is not empty
            new_Points = random.sample(lat_lon_pairs, nAdd)
        else:
            new_Points = []

    date_points.extend(new_Points)
    
    return date_points

def daily_point_selection(df,target_variable,nPoints_noFireDays,dist_from_fire,nonFire_factor):
    """
    df should have all columns create and shifted. There will be no temporal continuity for each point after this function
    df date column should be pandas datetime date. i.e., df['date'] = pd.to_datetime(df['date']).dt.date
    Spatial coordinates should be in latitude and longitude columns
    """
    days = df['date'].unique()

    # Loop though each date in the dataframe and perform point selection
    filtered_df_list = []
    for day in days:
        print(f'working on {day}')
        df_daily = df_extractDate(df,'date',day)

        # Collect unique spatial points for this day
        temp_df = df_daily[['latitude', 'longitude']].drop_duplicates().copy()
        lat_lon_pairs = list(zip(temp_df['latitude'], temp_df['longitude']))

        # Get the points for the current day
        date_points = select_points(df_daily, target_variable, lat_lon_pairs, nPoints_noFireDays, dist_from_fire, day, nonFire_factor)
        print(f'\tNumber of points {len(date_points)}')
        # For each point selected for the current day, filter df for those coordinates and the current day,
        # then add each matching row as a tuple to filtered_df_list
        for point in date_points:
            latitude, longitude = point
            
            # Filter for rows that match the current point and day
            matching_rows = df_daily[(df_daily['latitude'] == latitude) & (df_daily['longitude'] == longitude)]
            
            # Add each matching row as a tuple to the list
            for _, row in matching_rows.iterrows():
                filtered_df_list.append(tuple(row))

    filtered_df = pd.DataFrame(filtered_df_list, columns=df.columns)

    return filtered_df
####