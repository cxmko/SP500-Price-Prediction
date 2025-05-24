"""
Modèles PyTorch pour le projet de prédiction du S&P 500
"""

import torch
import torch.nn as nn
from scripts.config import (
    SEQUENCE_LENGTH, FEATURES, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, 
    LSTM_DROPOUT, CNN_FILTERS, CNN_KERNEL_SIZES, CNN_POOL_SIZES, CNN_DROPOUT
)

class LSTMModel(nn.Module):
    """
    Modèle LSTM pour la prédiction de séries temporelles.
    """
    def __init__(self, input_size=len(FEATURES), hidden_size=LSTM_HIDDEN_SIZE, 
                 num_layers=LSTM_NUM_LAYERS, dropout=LSTM_DROPOUT):
        super(LSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        
        # Nous utilisons uniquement la sortie du dernier pas de temps
        lstm_out = lstm_out[:, -1, :]
        
        out = self.dropout(lstm_out)
        out = self.fc(out)
        
        return out


class CNNModel(nn.Module):
    """
    Modèle CNN 1D pour la prédiction de séries temporelles.
    """
    def __init__(self, input_channels=len(FEATURES), sequence_length=SEQUENCE_LENGTH):
        super(CNNModel, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # Première couche Conv1D
        self.layers.append(nn.Conv1d(input_channels, CNN_FILTERS[0], kernel_size=CNN_KERNEL_SIZES[0]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool1d(CNN_POOL_SIZES[0]))
        
        # Calcul de la taille après la première couche
        current_size = (sequence_length - CNN_KERNEL_SIZES[0] + 1) // CNN_POOL_SIZES[0]
        
        # Couches Conv1D supplémentaires
        for i in range(1, len(CNN_FILTERS)):
            self.layers.append(nn.Conv1d(CNN_FILTERS[i-1], CNN_FILTERS[i], kernel_size=CNN_KERNEL_SIZES[i]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.MaxPool1d(CNN_POOL_SIZES[i]))
            
            # Mise à jour de la taille de sortie
            current_size = (current_size - CNN_KERNEL_SIZES[i] + 1) // CNN_POOL_SIZES[i]
        
        # Aplatissement et couche Dense finale
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(CNN_DROPOUT)
        
        # Calcul de la taille d'entrée pour la couche FC
        self.fc_input_size = current_size * CNN_FILTERS[-1] if current_size > 0 else CNN_FILTERS[-1]
        
        self.fc = nn.Linear(self.fc_input_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, features)
        # Nous devons permuter pour obtenir (batch_size, features, sequence_length)
        x = x.permute(0, 2, 1)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        
        return x
    





