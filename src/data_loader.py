import yfinance as yf
import pandas as pd

def get_stock_data(ticker, start_date, end_date):
    """
    Télécharge les données de Yahoo Finance et simplifie le format.
    """
    print(f"Extraction des données pour {ticker}...")
    df = yf.download(ticker, start=start_date, end=end_date)
    
    # Nettoyage des noms de colonnes (pour éviter les MultiIndex complexes)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    return df

if __name__ == "__main__":
    # Test rapide
    data = get_stock_data("AAPL", "2020-01-01", "2025-01-01")
    if not data.empty:
        print("\n--- Aperçu des données ---")
        print(data.head())
        print("\nColonnes disponibles :", data.columns.tolist())