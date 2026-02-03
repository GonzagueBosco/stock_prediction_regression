import pandas as pd

def calculate_rsi(series, period=14):
    """
    Calcule l'indicateur RSI (Relative Strength Index).
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def add_features(df):
    """
    Ajoute des indicateurs techniques (SMA et RSI) au DataFrame.
    """
    df = df.copy()
    
    # Moyennes Mobiles
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Nouvel indicateur : RSI
    df['RSI'] = calculate_rsi(df['Close'])
    
    # Cible : Prix du lendemain
    df['Target'] = df['Close'].shift(-1)
    
    # On supprime les NaN créés par les calculs (le RSI a besoin de 14 jours)
    df.dropna(inplace=True)
    
    return df

if __name__ == "__main__":
    from data_loader import get_stock_data
    raw_data = get_stock_data("AAPL", "2020-01-01", "2025-01-01")
    enriched_data = add_features(raw_data)
    
    print("\n--- Données avec RSI ---")
    print(enriched_data[['Close', 'RSI', 'Target']].head())