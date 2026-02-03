import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from data_loader import get_stock_data
from engineering import add_features

# 1. Préparation des données
df = get_stock_data("AAPL", "2020-01-01", "2025-01-01")
df = add_features(df)

# 2. Choix des variables (X) et de la cible (y)
X = df[['SMA_10', 'SMA_50']]
y = df['Target']

# 3. Split Temporel (80% entraînement, 20% test)
split = int(0.8 * len(df))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 4. Entraînement du modèle
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Prédictions
predictions = model.predict(X_test)

# 6. Affichage des résultats
plt.figure(figsize=(12,6))
plt.plot(y_test.index, y_test.values, label="Réel (AAPL)", color="blue")
plt.plot(y_test.index, predictions, label="Prédiction", color="red", linestyle="--")
plt.title("Prédiction du prix AAPL avec Régression Linéaire")
plt.legend()
plt.show()

print(f"Le modèle est entraîné. Score (R²): {model.score(X_test, y_test):.4f}")