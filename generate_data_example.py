# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Générer des données de séries temporelles exemple
np.random.seed(42)

# Dates
start_date = datetime(2020, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(1000)]

# Variable 1: Tendance + Saisonnalité + Bruit
trend = np.linspace(100, 200, 1000)
seasonal = 20 * np.sin(2 * np.pi * np.arange(1000) / 365.25)
noise = np.random.normal(0, 5, 1000)
var1 = trend + seasonal + noise

# Variable 2: Corrélée avec Variable 1 + sa propre saisonnalité
var2 = 0.8 * var1 + 15 * np.cos(2 * np.pi * np.arange(1000) / 30) + np.random.normal(0, 3, 1000)

# Variable 3: Indépendante avec tendance différente
var3 = 50 + 0.05 * np.arange(1000) + 10 * np.sin(2 * np.pi * np.arange(1000) / 90) + np.random.normal(0, 4, 1000)

# Créer DataFrame
df = pd.DataFrame({
    'Date': dates,
    'Variable_1': var1,
    'Variable_2': var2,
    'Variable_3': var3
})

# Sauvegarder
df.to_csv('exemple_series_temporelles.csv', index=False)
print("? Fichier exemple_series_temporelles.csv créé avec succès!")
print(f"?? {len(df)} observations, {len(df.columns)-1} variables")
print("\nAperçu des données:")
print(df.head())