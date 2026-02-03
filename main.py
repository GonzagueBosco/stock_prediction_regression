from src.data_loader import get_stock_data
from src.engineering import add_features
from sklearn.linear_model import LinearRegression

def run_pipeline(ticker):
    # 1. Get data
    df = get_stock_data(ticker, "2023-01-01", "2026-03-02")
    df = add_features(df)
    
    # 2. Train model (on everything to predict tomorrow)
    X = df[['SMA_10', 'SMA_50']]
    y = df['Target']
    
    model = LinearRegression()
    model.fit(X, y)
    
    # 3. Predict Tomorrow!
    # On prend la dernière ligne disponible pour prédire le futur
    last_features = X.tail(1)
    prediction = model.predict(last_features)
    
    print(f"\n🚀 Prédiction pour la prochaine clôture de {ticker} : {prediction[0]:.2f} $")

if __name__ == "__main__":
    symbol = input("Entrez le ticker d'une action (ex: AAPL, TSLA, MSFT) : ")
    run_pipeline(symbol)