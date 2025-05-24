"""
Script principal pour le projet de prédiction du S&P 500 avec PyTorch
"""
import os
import sys

_current_script_path = os.path.abspath(__file__)
_scripts_dir = os.path.dirname(_current_script_path)
_project_root = os.path.dirname(_scripts_dir) # This should be 'predictsp500 - Copy'
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from scripts.data_loader import load_data
from scripts.preprocess import prepare_data, create_dataloaders
from scripts.models_pytorch import LSTMModel, CNNModel
from scripts.train_evaluate_pytorch import (
    train_model, evaluate_model, save_model, 
    plot_training_history, save_evaluation_report, set_seed
)
from scripts.config import (
    LSTM_MODEL_NAME, CNN_MODEL_NAME, LEARNING_RATE, 
    LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, LSTM_DROPOUT,
    CNN_FILTERS, CNN_KERNEL_SIZES, CNN_POOL_SIZES, CNN_DROPOUT,
    FEATURES, PREDICTION_HORIZON, EPOCHS, SEQUENCE_LENGTH, BATCH_SIZE,
    RANDOM_SEED, CNN_LEARNING_RATE,  CNN_EPOCHS,
    TARGET_COL, MODELS_DIR 
)


def main():
    set_seed(RANDOM_SEED)
    
    parser = argparse.ArgumentParser(description='Prédiction des prix du S&P 500 avec des modèles de Deep Learning')
    parser.add_argument('--model', type=str, choices=['lstm', 'cnn', 'both'], default='lstm',
                        help='Modèle à entraîner (lstm, cnn ou both)')
    parser.add_argument('--visualize_denoising', action='store_true', 
                        help='Désactiver la visualisation de l\'effet du débruitage (activée par défaut).') # Help text was for store_false, but action is store_true. Correcting help text if default is False.
                                                                                                           # If default is True (visualize_denoising=True by default), then action should be 'store_false'
                                                                                                           # Assuming current setup means --visualize_denoising flag ENABLES it.
    parser.add_argument('--load_weights', action='store_false', # This makes load_weights True by default
                        help='Charger les poids pré-entraînés du modèle avant l\'entraînement (True par défaut, use flag to disable).') # Corrected help text
    parser.add_argument('--plot_evaluation', action='store_false',
                        help='Générer et sauvegarder les graphiques d\'évaluation des prédictions (False par défaut).')
    args = parser.parse_args()
    
    # Chargement et prétraitement des données
    print("Chargement des données...")
    df = load_data()
    
    if df is None:
        print("Erreur lors du chargement des données. Arrêt du programme.")
        return
    
    print("\nPrétraitement des données...")
    visualize_col = TARGET_COL if args.visualize_denoising else None # visualize_denoising is True by default now
    X_train, X_val, X_test, y_train, y_val, y_test, \
    y_train_orig, y_val_orig, y_test_orig, scalers = prepare_data( # Unpack new y_orig arrays
        df,
        denoise_features=True, 
        visualize_denoising_col=visualize_col
    )
    
    print("\nCréation des DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test)
    
    # Définition du critère de perte et du device
    criterion = nn.MSELoss()  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Entraînement et évaluation des modèles
    if args.model in ['lstm', 'both']:
        print("\n" + "="*50)
        print("Modèle LSTM")
        print("="*50)
        
        # Initialisation du modèle LSTM
        lstm_model = LSTMModel(input_size=len(FEATURES)).to(device) # Move to device
        
        if args.load_weights:
            model_path = os.path.join(MODELS_DIR, LSTM_MODEL_NAME)
            if os.path.exists(model_path):
                try:
                    lstm_model.load_state_dict(torch.load(model_path, map_location=device))
                    print(f"Poids pré-entraînés chargés pour LSTM depuis {model_path}")
                except Exception as e:
                    print(f"Erreur lors du chargement des poids LSTM: {e}. Entraînement à partir de zéro.")
            else:
                print(f"Aucun poids pré-entraîné trouvé pour LSTM à {model_path}. Entraînement à partir de zéro.")
        else:
            print("Entraînement LSTM à partir de zéro (aucun poids pré-entraîné à charger).")

        lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE)
        
        # Paramètres du modèle pour le rapport
        lstm_params = {
            'model_type': 'LSTM',
            'input_size': len(FEATURES),
            'hidden_size': LSTM_HIDDEN_SIZE,
            'num_layers': LSTM_NUM_LAYERS,
            'dropout': LSTM_DROPOUT,
            'optimizer': 'Adam',
            'learning_rate': LEARNING_RATE,
            'criterion': 'MSELoss',
            'epochs': EPOCHS, # Assuming LSTM uses the general EPOCHS
            'batch_size': BATCH_SIZE, # Assuming LSTM uses the general BATCH_SIZE
            'sequence_length': SEQUENCE_LENGTH,
            'prediction_horizon': PREDICTION_HORIZON,
            'random_seed': RANDOM_SEED,
            'target_col': TARGET_COL,
            'features_list': FEATURES,
            'target_type_for_training': 'denoised_scaled', # Clarify
            'loaded_weights': args.load_weights # Add info about loaded weights
        }
        
        # Entraînement du modèle LSTM
        print("\nEntraînement du modèle LSTM...")
        lstm_model, lstm_history = train_model(
            lstm_model, train_loader, val_loader, criterion, lstm_optimizer
        )
        
        # Sauvegarde du modèle LSTM
        save_model(lstm_model, LSTM_MODEL_NAME)
        
        # Tracé des courbes d'apprentissage
        plot_training_history(lstm_history, 'LSTM')
        
        # Évaluation du modèle LSTM
        print("\nÉvaluation du modèle LSTM...")
        lstm_metrics = evaluate_model(
            lstm_model, 
            test_loader, 
            criterion, 
            scalers, 
            y_test_original_unscaled=y_test_orig,
            device=device,
            plot_results=args.plot_evaluation, 
            model_name="LSTM"                   
            )
        
        # Sauvegarde du rapport d'évaluation
        save_evaluation_report('LSTM', lstm_metrics, lstm_params)
    
    if args.model in ['cnn', 'both']:
        print("\n" + "="*50)
        print("Modèle CNN avec entraînement stabilisé")
        print("="*50)
        
        # Set global seeds
        set_seed(RANDOM_SEED)
        
        
        cnn_model = CNNModel(input_channels=len(FEATURES), sequence_length=SEQUENCE_LENGTH).to(device) # Move to device
        
        if args.load_weights:
            model_path = os.path.join(MODELS_DIR, CNN_MODEL_NAME)
            if os.path.exists(model_path):
                try:
                    cnn_model.load_state_dict(torch.load(model_path, map_location=device))
                    print(f"Poids pré-entraînés chargés pour CNN depuis {model_path}")
                except Exception as e:
                    print(f"Erreur lors du chargement des poids CNN: {e}. Entraînement à partir de zéro.")
            else:
                print(f"Aucun poids pré-entraîné trouvé pour CNN à {model_path}. Entraînement à partir de zéro.")
        else:
            print("Entraînement CNN à partir de zéro (aucun poids pré-entraîné à charger).")

        
        cnn_optimizer = optim.AdamW(
            cnn_model.parameters(), 
            lr=CNN_LEARNING_RATE if 'CNN_LEARNING_RATE' in globals() else LEARNING_RATE,
            betas=(0.9, 0.999)
        )
        
        
        # Enhanced parameters for reporting
        cnn_params = {
            'model_type': 'CNN (Stabilized)', # Corrected key from 'model' to 'model_type' for consistency
            'input_channels': len(FEATURES),
            'sequence_length': SEQUENCE_LENGTH,
            'cnn_filters': CNN_FILTERS,
            'cnn_kernel_sizes': CNN_KERNEL_SIZES,
            'cnn_pool_sizes': CNN_POOL_SIZES,
            'cnn_dropout': CNN_DROPOUT,
            'optimizer': 'AdamW',
            'learning_rate_max (OneCycleLR)': CNN_LEARNING_RATE,
            'optimizer_betas': (0.9, 0.999),
            'criterion': 'MSELoss',
            'epochs': CNN_EPOCHS,
            'batch_size': BATCH_SIZE, 
            'prediction_horizon': PREDICTION_HORIZON,
            'random_seed': RANDOM_SEED,
            'target_col': TARGET_COL,
            'features_list': FEATURES,
            'target_type_for_training': 'denoised_scaled', # Clarify
            'loaded_weights': args.load_weights # Add info about loaded weights
        }
        
        # Train with stability features
        print("\nEntraînement du modèle CNN stabilisé...")
        cnn_model, cnn_history = train_model(
            cnn_model, train_loader, val_loader, criterion, cnn_optimizer,epochs=CNN_EPOCHS, seed=RANDOM_SEED
        )
        
        # Sauvegarde du modèle CNN
        save_model(cnn_model, CNN_MODEL_NAME)
        
        # Tracé des courbes d'apprentissage
        plot_training_history(cnn_history, 'CNN')
        
        # Évaluation du modèle CNN
        print("\nÉvaluation du modèle CNN...")
        # Pass y_test_orig to evaluate_model
        cnn_metrics = evaluate_model(
            cnn_model, 
            test_loader, 
            criterion, 
            scalers, 
            y_test_original_unscaled=y_test_orig,
            device=device,
            plot_results=args.plot_evaluation, 
            model_name="CNN"                  
            )
        
        
        if 'lstm_params' in locals(): 
            lstm_params['loaded_weights'] = args.load_weights

        # Sauvegarde du rapport d'évaluation
        save_evaluation_report('CNN', cnn_metrics, cnn_params)
    
    print("\nTerminé!")

if __name__ == "__main__":
    main()