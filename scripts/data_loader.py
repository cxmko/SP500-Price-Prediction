"""
Chargeur de données pour le projet de prédiction du S&P 500
"""

import pandas as pd
import numpy as np
from scripts.config import DATA_FILE

def load_data():
    try:
        # Chargement des données
        df = pd.read_csv(
                DATA_FILE,
                skiprows=3,
                header=None,
                names=['Date','Close','High','Low','Open','Volume']
            )
        
        # Vérification de la structure des données
        print(f"Données chargées avec succès: {df.shape[0]} lignes et {df.shape[1]} colonnes")
        print("Aperçu des données:")
        print(df.head())
        
        # Conversion de la colonne Date en datetime et définition comme index
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
            df.set_index('Date', inplace=True)

        
        
        # Vérification des valeurs manquantes
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print("Valeurs manquantes détectées:")
            print(missing_values[missing_values > 0])
        else:
            print("Aucune valeur manquante détectée.")
        
        return df
    
    except Exception as e:
        print(f"Erreur lors du chargement des données: {e}")
        return None