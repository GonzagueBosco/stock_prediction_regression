import pandas as pd

def add_features(df):
    """
    Ajoute des indicateurs techniques (Moyennes Mobiles) au DataFrame.
    """
    # On crée une copie pour ne pas modifier l'original par erreur
    df = df.copy()
    
    # Moyenne mobile sur 10 jours et 50 jours
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # On crée la 'Cible' (Target) : le prix de clôture du LENDEMAIN
    # On décale les prix de -1 pour que la ligne d'aujourd'hui contienne le prix de demain
    df['Target'] = df['Close'].shift(-1)
    
    # On supprime les lignes vides (les 50 premières n'ont pas de SMA_50, 
    # et la dernière n'a pas de Target)
    df.dropna(inplace=True)
    
    return df

if __name__ == "__main__":
    # Test  
    from data_loader import get_stock_data
    
    raw_data = get_stock_data("AAPL", "2020-01-01", "2025-01-01")
    enriched_data = add_features(raw_data)
    
    print("\n--- Données avec Features ---")
    print(enriched_data[['Close', 'SMA_10', 'SMA_50', 'Target']].head())