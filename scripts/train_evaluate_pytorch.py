"""
Entraînement et évaluation des modèles PyTorch pour le projet de prédiction du S&P 500
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

from scripts.config import (
    MODELS_DIR, REPORTS_DIR, EPOCHS, EARLY_STOPPING_PATIENCE, TARGET_COL, RANDOM_SEED
)

def set_seed(seed=RANDOM_SEED):
    """
    Set seeds for reproducibility across all random number generators.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seeds set to {seed} for reproducibility")

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=EPOCHS, patience=EARLY_STOPPING_PATIENCE, seed=RANDOM_SEED):
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation de l'appareil: {device}")
    
    model = model.to(device)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mse': [],
        'val_mse': []
    }
    
    best_val_loss = float('inf')
    counter = 0
    best_model_state = None
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Mode entraînement
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward et optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Stockage des prédictions et des cibles (sans seuil)
            train_preds.extend(outputs.cpu().detach().numpy().flatten())
            train_targets.extend(targets.cpu().numpy().flatten())
        
        train_loss /= len(train_loader)
        train_mse = mean_squared_error(train_targets, train_preds)
        
        # Mode évaluation
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                
                val_preds.extend(outputs.cpu().numpy().flatten())
                val_targets.extend(targets.cpu().numpy().flatten())
        
        val_loss /= len(val_loader)
        val_mse = mean_squared_error(val_targets, val_preds)
        
        # Mise à jour de l'historique
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mse'].append(train_mse)
        history['val_mse'].append(val_mse)
        
        # Affichage des métriques
        print(f"Époque {epoch+1}/{epochs} - "
              f"Loss: {train_loss:.4f} - MSE: {train_mse:.4f} - "
              f"Val Loss: {val_loss:.4f} - Val MSE: {val_mse:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            #best_model_state = model.state_dict().copy()
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping à l'époque {epoch+1}")
                break
    
    # Restauration du meilleur modèle
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    training_time = time.time() - start_time
    print(f"Temps d'entraînement total: {training_time:.2f} secondes")
    
    return model, history



def evaluate_model(model, test_loader, criterion, scalers=None, y_test_original_unscaled=None, device=None, plot_results=False, model_name="Model"): 

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()
    
    test_loss_on_scaled_denoised = 0 
    all_preds_scaled = []
    all_targets_scaled_denoised = [] 
    
    with torch.no_grad():
        for inputs, targets_from_loader in test_loader:
            inputs, targets_from_loader = inputs.to(device), targets_from_loader.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets_from_loader) 
            
            test_loss_on_scaled_denoised += loss.item()
            
            all_preds_scaled.extend(outputs.cpu().numpy().flatten())
            all_targets_scaled_denoised.extend(targets_from_loader.cpu().numpy().flatten())
    
    test_loss_on_scaled_denoised /= len(test_loader)
    
    all_preds_scaled = np.array(all_preds_scaled).reshape(-1, 1)
    all_targets_scaled_denoised = np.array(all_targets_scaled_denoised).reshape(-1, 1)
    
    predictions_inversed_scale = all_preds_scaled.flatten() 
    if scalers and TARGET_COL in scalers: 
        predictions_inversed_scale = scalers[TARGET_COL].inverse_transform(all_preds_scaled).flatten()

    targets_denoised_inversed_scale = all_targets_scaled_denoised.flatten()
    if scalers and TARGET_COL in scalers:
        targets_denoised_inversed_scale = scalers[TARGET_COL].inverse_transform(all_targets_scaled_denoised).flatten()

    mse_vs_denoised = mean_squared_error(targets_denoised_inversed_scale, predictions_inversed_scale)
    rmse_vs_denoised = np.sqrt(mse_vs_denoised)
    mae_vs_denoised = mean_absolute_error(targets_denoised_inversed_scale, predictions_inversed_scale)
    r2_vs_denoised = r2_score(targets_denoised_inversed_scale, predictions_inversed_scale)
    
    print("\nRésultats de l'évaluation (vs Denoised Target):")
    print(f"Loss (on scaled, denoised): {test_loss_on_scaled_denoised:.4f}") 
    print(f"MSE: {mse_vs_denoised:.4f}")
    print(f"RMSE: {rmse_vs_denoised:.4f}")
    print(f"MAE: {mae_vs_denoised:.4f}")
    print(f"R²: {r2_vs_denoised:.4f}")

    results = {
        'loss_on_scaled_denoised': test_loss_on_scaled_denoised,
        'metrics_vs_denoised_target': {
            'mse': mse_vs_denoised,
            'rmse': rmse_vs_denoised,
            'mae': mae_vs_denoised,
            'r2': r2_vs_denoised,
        },
        'predictions_inversed_scale': predictions_inversed_scale.tolist(),
        'targets_denoised_inversed_scale': targets_denoised_inversed_scale.tolist()
    }

    y_test_original_unscaled_flat = None
    if y_test_original_unscaled is not None:
        y_test_original_unscaled_flat = y_test_original_unscaled.flatten()
        if len(y_test_original_unscaled_flat) == len(predictions_inversed_scale):
            mse_vs_original = mean_squared_error(y_test_original_unscaled_flat, predictions_inversed_scale)
            rmse_vs_original = np.sqrt(mse_vs_original)
            mae_vs_original = mean_absolute_error(y_test_original_unscaled_flat, predictions_inversed_scale)
            r2_vs_original = r2_score(y_test_original_unscaled_flat, predictions_inversed_scale)

            print("\nRésultats de l'évaluation (vs Original Target):")
            print(f"MSE: {mse_vs_original:.4f}")
            print(f"RMSE: {rmse_vs_original:.4f}")
            print(f"MAE: {mae_vs_original:.4f}")
            print(f"R²: {r2_vs_original:.4f}")

            results['metrics_vs_original_target'] = {
                'mse': mse_vs_original,
                'rmse': rmse_vs_original,
                'mae': mae_vs_original,
                'r2': r2_vs_original,
            }
            results['targets_original_unscaled'] = y_test_original_unscaled_flat.tolist()
        else:
            print(f"\nWarning: Length mismatch between predictions ({len(predictions_inversed_scale)}) and original targets ({len(y_test_original_unscaled_flat)}). Skipping metrics vs original target.")
            results['metrics_vs_original_target'] = None
    else:
        results['metrics_vs_original_target'] = None

    if plot_results:
        plt.figure(figsize=(18, 12))
        
        plot_time_index = np.arange(len(predictions_inversed_scale))

        # Plot 1: Predictions vs Denoised (Smoothed) Actual
        plt.subplot(2, 1, 1)
        plt.plot(plot_time_index, targets_denoised_inversed_scale, label='Actual Denoised (Inverse Scaled)', color='blue', alpha=0.7)
        plt.plot(plot_time_index, predictions_inversed_scale, label='Predicted (Inverse Scaled)', color='red', linestyle='--', alpha=0.9)
        plt.title(f'{model_name}: Predictions vs. Denoised Actual Targets')
        plt.xlabel('Test Sample Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 2: Predictions vs Raw Actual (if available)
        plt.subplot(2, 1, 2)
        if y_test_original_unscaled_flat is not None and results.get('metrics_vs_original_target') is not None:
            plt.plot(plot_time_index, y_test_original_unscaled_flat, label='Actual Raw (Original)', color='green', alpha=0.7)
            plt.plot(plot_time_index, predictions_inversed_scale, label='Predicted (Inverse Scaled)', color='red', linestyle='--', alpha=0.9)
            plt.title(f'{model_name}: Predictions vs. Raw Actual Targets')
        else:
            plt.text(0.5, 0.5, 'Raw actual target data not available or length mismatch for plotting.', 
                     horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
            plt.title(f'{model_name}: Predictions vs. Raw Actual Targets (Data N/A)')
        plt.xlabel('Test Sample Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout(pad=3.0)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not os.path.exists(REPORTS_DIR):
            os.makedirs(REPORTS_DIR)
        plot_save_path = os.path.join(REPORTS_DIR, f"{model_name}_evaluation_plot_{timestamp}.png")
        plt.savefig(plot_save_path)
        print(f"Evaluation plot saved to {plot_save_path}")
        plt.close()
        
    return results


def save_evaluation_report(model_name, metrics_results, params=None): # Renamed metrics to metrics_results

    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(REPORTS_DIR, f"{model_name}_evaluation_{timestamp}.txt")
    
    with open(report_path, 'w') as f:
        f.write(f"Rapport d'évaluation pour {model_name}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if params:
            f.write("Paramètres du modèle:\n")
            for key, value in params.items():
                f.write(f"- {key}: {value}\n")
            f.write("\n")
        
        f.write(f"Loss (on scaled, denoised targets): {metrics_results.get('loss_on_scaled_denoised', float('nan')):.4f}\n\n")

        if 'metrics_vs_denoised_target' in metrics_results and metrics_results['metrics_vs_denoised_target']:
            f.write("Métriques d'évaluation (vs Denoised Target):\n")
            m_denoised = metrics_results['metrics_vs_denoised_target']
            f.write(f"- MSE: {m_denoised.get('mse', float('nan')):.4f}\n")
            f.write(f"- RMSE: {m_denoised.get('rmse', float('nan')):.4f}\n")
            f.write(f"- MAE: {m_denoised.get('mae', float('nan')):.4f}\n")
            f.write(f"- R²: {m_denoised.get('r2', float('nan')):.4f}\n\n")

        if 'metrics_vs_original_target' in metrics_results and metrics_results['metrics_vs_original_target']:
            f.write("Métriques d'évaluation (vs Original Target):\n")
            m_original = metrics_results['metrics_vs_original_target']
            f.write(f"- MSE: {m_original.get('mse', float('nan')):.4f}\n")
            f.write(f"- RMSE: {m_original.get('rmse', float('nan')):.4f}\n")
            f.write(f"- MAE: {m_original.get('mae', float('nan')):.4f}\n")
            f.write(f"- R²: {m_original.get('r2', float('nan')):.4f}\n\n")
        
    print(f"Rapport d'évaluation sauvegardé à {report_path}")

def save_model(model, model_name):

    save_path = os.path.join(MODELS_DIR, model_name)
    torch.save(model.state_dict(), save_path)
    print(f"Modèle sauvegardé à {save_path}")

def load_model(model, model_name):

    load_path = os.path.join(MODELS_DIR, model_name)
    model.load_state_dict(torch.load(load_path))
    print(f"Modèle chargé depuis {load_path}")
    return model

def plot_training_history(history, model_name):

    plt.figure(figsize=(12, 5))
    
    # Courbe de perte
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title(f"Perte d'entraînement - {model_name}")
    plt.xlabel('Époque')
    plt.ylabel('Perte')
    plt.legend()
    
    # Courbe de MSE
    plt.subplot(1, 2, 2)
    plt.plot(history['train_mse'], label='Train')
    plt.plot(history['val_mse'], label='Validation')
    plt.title(f"MSE d'entraînement - {model_name}")
    plt.xlabel('Époque')
    plt.ylabel('MSE')
    plt.legend()
    
    plt.tight_layout()
    
    # Sauvegarde de la figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(REPORTS_DIR, f"{model_name}_training_curves_{timestamp}.png")
    plt.savefig(save_path)
    print(f"Courbes d'apprentissage sauvegardées à {save_path}")
    
    plt.close()