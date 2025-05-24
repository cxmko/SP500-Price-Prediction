"""
Configuration pour le projet de prédiction du S&P 500
"""

import os
from pathlib import Path

# Chemins des répertoires
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models_saved"
REPORTS_DIR = ROOT_DIR / "reports"

# Paramètres de données
DATA_FILE = DATA_DIR / "GSPC.csv"
RANDOM_SEED = 42  
TEST_SIZE = 0.07
VAL_SIZE = 0.07

# Paramètres de prétraitement
SEQUENCE_LENGTH = 60  # Fenêtre glissante de 60 jours
TARGET_COL = 'Close'  
FEATURES = ['Open', 'High', 'Low', 'Volume']  
PREDICTION_HORIZON = 1  # Nombre de jours à prédire dans le futur

# Paramètres d'entraînement
BATCH_SIZE = 64
EPOCHS = 150


LEARNING_RATE = 0.0001
EARLY_STOPPING_PATIENCE = 100

# Paramètres des modèles
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.2


CNN_FILTERS = [32, 64, 128]
CNN_KERNEL_SIZES = [3, 3, 3]
CNN_POOL_SIZES = [2, 2, 2]
CNN_DROPOUT = 0.4


CNN_LEARNING_RATE = 0.0001  
CNN_BATCH_SIZE = 64         
CNN_EPOCHS = 50            






# Noms des modèles sauvegardés
LSTM_MODEL_NAME = "lstm_model.pt"
CNN_MODEL_NAME = "cnn_model.pt"

# Créer les répertoires s'ils n'existent pas
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)