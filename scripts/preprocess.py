"""
Prétraitement des données pour le projet de prédiction du S&P 500
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import pywt 
import random 
import matplotlib.pyplot as plt
from scripts.config import (
    SEQUENCE_LENGTH, FEATURES, TARGET_COL,
    TEST_SIZE, VAL_SIZE, RANDOM_SEED, BATCH_SIZE, PREDICTION_HORIZON
)

def wavelet_denoise(data, wavelet='sym8', level=1, denoise_enabled=True): 
    if not denoise_enabled:
        if isinstance(data, pd.Series):
            return data
        else:
            return pd.Series(data)

    if not isinstance(data, pd.Series):
        original_index = getattr(data, 'index', None)
        data_series = pd.Series(data)
        if original_index is not None and len(original_index) == len(data_series):
             data_series.index = original_index
    else:
        data_series = data.copy() # Work on a copy

    if data_series.empty:
        return data_series

    coeff = pywt.wavedec(data_series.values, wavelet, mode="per")
    
    detail_coeffs = coeff[-level]
    if len(detail_coeffs) == 0:
        return data_series

    sigma = (1/0.6745) * np.median(np.abs(detail_coeffs - np.median(detail_coeffs)))
    if sigma == 0: 
        uthresh = 0 
    else:
        uthresh = sigma * np.sqrt(2 * np.log(len(data_series) + 1e-9))
    
    thresholded_coeffs = [coeff[0]]
    for i in range(1, len(coeff)):
        thresholded_coeffs.append(pywt.threshold(coeff[i], value=uthresh, mode='soft'))
        
    denoised_values = pywt.waverec(thresholded_coeffs, wavelet, mode='per')
    
    if len(denoised_values) > len(data_series.values):
        denoised_values = denoised_values[:len(data_series.values)]
    elif len(denoised_values) < len(data_series.values):
        padding = np.full(len(data_series.values) - len(denoised_values), denoised_values[-1] if len(denoised_values) > 0 else 0)
        denoised_values = np.concatenate([denoised_values, padding])

    return pd.Series(denoised_values, index=data_series.index)


def prepare_data(df, denoise_features=True, visualize_denoising_col=None):

    df_original_for_target = df.copy() 

    df_for_processing = df.copy()

    if denoise_features:
        print("Applying wavelet denoising...")
        for col in FEATURES + [TARGET_COL]:
            if col in df_for_processing.columns:
                original_col_data_for_plot = df_for_processing[col].dropna().copy()
                
                denoised_col = wavelet_denoise(df_for_processing[col].dropna(), wavelet='sym8', level=1, denoise_enabled=True)
                df_for_processing[col] = denoised_col.reindex(df_for_processing.index).fillna(method='bfill').fillna(method='ffill')

                if visualize_denoising_col and col == visualize_denoising_col:
                    plt.figure(figsize=(15, 6))
                    plt.plot(original_col_data_for_plot.index, original_col_data_for_plot, label=f'Original {col}', alpha=0.7)
                    plot_denoised_data = df_for_processing[col].loc[original_col_data_for_plot.index.intersection(df_for_processing[col].index)]
                    plt.plot(plot_denoised_data.index, plot_denoised_data, label=f'Denoised {col} (sym8, level 1)', color='black') # Changed from red to black
                    plt.title(f'Wavelet Denoising Effect on {col}')
                    plt.xlabel('Date')
                    plt.ylabel('Value')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.show()
    
    # Create future target on the (potentially) DENOISED data
    df_for_processing[f'{TARGET_COL}_Future_Denoised'] = df_for_processing[TARGET_COL].shift(-PREDICTION_HORIZON)
    
    # Create future target on the ORIGINAL data
    df_original_for_target[f'{TARGET_COL}_Future_Original'] = df_original_for_target[TARGET_COL].shift(-PREDICTION_HORIZON)

    df_processed_denoised_target = df_for_processing.dropna(subset=[f'{TARGET_COL}_Future_Denoised'] + FEATURES)
    

    df_original_future_target_aligned = df_original_for_target[[f'{TARGET_COL}_Future_Original']].reindex(df_processed_denoised_target.index)
    df_original_future_target_aligned = df_original_future_target_aligned.dropna()
    
    df_processed_denoised_target = df_processed_denoised_target.reindex(df_original_future_target_aligned.index)

    feature_scalers = {}
    df_scaled = pd.DataFrame(index=df_processed_denoised_target.index)
    
    for feature in FEATURES:
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_scaled[feature] = scaler.fit_transform(df_processed_denoised_target[[feature]])
        feature_scalers[feature] = scaler
    
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled[f'{TARGET_COL}_Future_Denoised'] = target_scaler.fit_transform(df_processed_denoised_target[[f'{TARGET_COL}_Future_Denoised']])
    feature_scalers[TARGET_COL] = target_scaler # This scaler is for the denoised target

    X, y_scaled_denoised, y_original_unscaled = create_sequences(
        df_scaled, 
        df_original_future_target_aligned,
        FEATURES, 
        f'{TARGET_COL}_Future_Denoised', 
        f'{TARGET_COL}_Future_Original'
    )
    
    test_size_idx = int(len(X) * (1 - TEST_SIZE))
    val_size_idx = int(test_size_idx * (1 - VAL_SIZE))
    
    X_train, y_train = X[:val_size_idx], y_scaled_denoised[:val_size_idx]
    X_val, y_val = X[val_size_idx:test_size_idx], y_scaled_denoised[val_size_idx:test_size_idx]
    X_test, y_test = X[test_size_idx:], y_scaled_denoised[test_size_idx:]
    
    y_train_orig = y_original_unscaled[:val_size_idx]
    y_val_orig = y_original_unscaled[val_size_idx:test_size_idx]
    y_test_orig = y_original_unscaled[test_size_idx:]
    
    print(f"Taille des ensembles - Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, y_train_orig, y_val_orig, y_test_orig, feature_scalers


def create_sequences(df_scaled_data, df_original_future_target_data, features_list, scaled_future_target_col_name, original_future_target_col_name):

    X_list = []
    y_scaled_denoised_list = []
    y_original_unscaled_list = []
    
    num_possible_sequences = len(df_scaled_data) - SEQUENCE_LENGTH
    
    for i in range(num_possible_sequences):
        seq_X = df_scaled_data[features_list].iloc[i:i+SEQUENCE_LENGTH].values
        
        target_scaled = df_scaled_data[scaled_future_target_col_name].iloc[i+SEQUENCE_LENGTH]
        
        target_original = df_original_future_target_data[original_future_target_col_name].iloc[i+SEQUENCE_LENGTH]
        
        X_list.append(seq_X)
        y_scaled_denoised_list.append(target_scaled)
        y_original_unscaled_list.append(target_original)
    
    return np.array(X_list), np.array(y_scaled_denoised_list), np.array(y_original_unscaled_list)

def create_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=BATCH_SIZE, seed=RANDOM_SEED):

    generator = torch.Generator()
    generator.manual_seed(seed)
    
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    )
    
    val_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    )
    
    test_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        generator=generator
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, val_loader, test_loader

