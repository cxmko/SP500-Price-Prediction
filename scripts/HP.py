
"""
Hyperparameter tuning script for S&P 500 prediction models.
"""
import os
import sys
import argparse
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np 


_current_script_path = os.path.abspath(__file__)
_scripts_dir = os.path.dirname(_current_script_path)
_project_root = os.path.dirname(_scripts_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.data_loader import load_data
from scripts.preprocess import prepare_data, create_dataloaders
from scripts.models_pytorch import LSTMModel, CNNModel 
from scripts.train_evaluate_pytorch import train_model, evaluate_model, set_seed
from scripts import config as base_config 

def define_search_space(model_type):
    if model_type == 'lstm':
        search_space = {
            'hidden_size': [32, base_config.LSTM_HIDDEN_SIZE, 96], 
            'num_layers': [1, base_config.LSTM_NUM_LAYERS, 3],      # e.g., [1, 2, 3]
            'dropout': [0.1, base_config.LSTM_DROPOUT, 0.3],        # e.g., [0.1, 0.2, 0.3]
            'learning_rate': [base_config.LEARNING_RATE / 2, base_config.LEARNING_RATE, base_config.LEARNING_RATE * 2] # e.g., [5e-5, 1e-4, 2e-4]
        }
    else:
        raise ValueError(f"Unsupported model type for tuning: {model_type}. Only 'lstm' is supported.")
    return search_space

def run_trial(model_type, trial_params, train_loader, val_loader, test_loader,
              scalers, y_test_orig, device, criterion, trial_epochs):
    print(f"\nRunning trial with params: {trial_params}")
    set_seed(base_config.RANDOM_SEED) 

    try:
        if model_type == 'lstm':
            model = LSTMModel(
                input_size=len(base_config.FEATURES),
                hidden_size=trial_params['hidden_size'],
                num_layers=trial_params['num_layers'],
                dropout=trial_params['dropout']
            ).to(device)
            optimizer = optim.Adam(model.parameters(), lr=trial_params['learning_rate'])
        else:
            # This case should ideally not be reached if model_type is validated upstream
            print(f"Unsupported model type in run_trial: {model_type}")
            return np.inf 

        # Train the model
        model, _ = train_model(
            model, train_loader, val_loader, criterion, optimizer,
            epochs=trial_epochs,
        )

        # Evaluate the model
        metrics = evaluate_model(
            model, test_loader, criterion, scalers,
            y_test_original_unscaled=y_test_orig,
            device=device,
            plot_results=False 
        )
        

        mae = metrics.get('metrics_vs_denoised_target', {}).get('mae', np.inf)
        
        print(f"Trial MAE: {mae:.4f}")
        return mae
    except Exception as e:
        print(f"Error during trial with params {trial_params}: {e}")
        return np.inf 

def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Tuning for S&P 500 Models')
    parser.add_argument('--model_type', type=str, required=True, choices=['lstm'],
                        help='Model type to tune (only lstm is supported)')
    parser.add_argument('--epochs_per_trial', type=int, default=10,
                        help='Number of epochs to train each hyperparameter configuration')
    parser.add_argument('--max_trials', type=int, default=None,
                        help='Maximum number of hyperparameter combinations to try (optional)')
    parser.add_argument('--print_every_n_trials', type=int, default=5,
                        help='How often to print the top configurations')
    parser.add_argument('--num_top_configs', type=int, default=3,
                        help='Number of top configurations to print')
    args = parser.parse_args()

    
    set_seed(base_config.RANDOM_SEED)

    print("Loading and preprocessing data...")
    df = load_data()
    if df is None:
        print("Failed to load data. Exiting.")
        return

    X_train, X_val, X_test, y_train, y_val, y_test, \
    _, _, y_test_orig, scalers = prepare_data(
        df,
        denoise_features=False, 
        visualize_denoising_col=None
    )
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, X_val, X_test, y_train, y_val, y_test, batch_size=base_config.BATCH_SIZE
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    print(f"Using device: {device}")

    search_space = define_search_space(args.model_type)
    param_names = list(search_space.keys())
    param_values = list(search_space.values())
    
    all_combinations = list(itertools.product(*param_values))
    
    if args.max_trials is not None and args.max_trials < len(all_combinations):
        print(f"Limiting to {args.max_trials} random trials from {len(all_combinations)} total combinations.")
        import random
        random.shuffle(all_combinations) # Shuffle to get random subset
        all_combinations = all_combinations[:args.max_trials]


    print(f"Starting hyperparameter tuning for {args.model_type} model...")
    print(f"Total combinations to test: {len(all_combinations)}")

    results = []

    for i, combo in enumerate(all_combinations):
        current_trial_params = dict(zip(param_names, combo))
        
        mae = run_trial(
            args.model_type, current_trial_params,
            train_loader, val_loader, test_loader,
            scalers, y_test_orig, device, criterion,
            args.epochs_per_trial
        )
        
        if mae != np.inf: # Check if trial was successful
            results.append({'params': current_trial_params, 'mae': mae})
            # Sort results by MAE (lower is better)
            results.sort(key=lambda x: x['mae'])
        
        print(f"--- Trial {i+1}/{len(all_combinations)} completed ---")

        if (i + 1) % args.print_every_n_trials == 0 or (i + 1) == len(all_combinations):
            print(f"\nTop {min(args.num_top_configs, len(results))} configurations after {i+1} trials (lower MAE is better):")
            # Ensure we only try to print up to the number of available results
            for r_idx, res in enumerate(results[:min(args.num_top_configs, len(results))]):
                print(f"  {r_idx+1}. MAE: {res['mae']:.4f} | Params: {res['params']}")
            print("-" * 30)

    print("\nHyperparameter tuning finished.")
    if results:
        print(f"Overall top {min(args.num_top_configs, len(results))} best configurations (lower MAE is better):")
        # Ensure we only try to print up to the number of available results
        for r_idx, res in enumerate(results[:min(args.num_top_configs, len(results))]):
            print(f"  {r_idx+1}. MAE: {res['mae']:.4f} | Params: {res['params']}")
    else:
        print("No successful trials completed.")

if __name__ == '__main__':
    main()