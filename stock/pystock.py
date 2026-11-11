from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import json

app = Flask(__name__)
df = pd.read_csv("StockPriceDataset.csv")

@app.route("/")
def home():
    return render_template("share.html")

@app.route("/predict", methods=["POST"])
def predict():
    company_name = request.form["company"].upper()
    data = df[df['Ticker'] == company_name].copy()

    if data.empty:
        return render_template("share.html", prediction_text="❌ Company not found", error=True)

    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date')
    data = data.dropna()

   
    data['MA5'] = data['Close'].rolling(5).mean()
    data['MA10'] = data['Close'].rolling(10).mean()
    data['Daily_Return'] = data['Close'].pct_change()
    data['Volume_Change'] = data['Volume'].pct_change()

    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    data = data.dropna()
    data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
    data = data.dropna()

    features = ['Open','High','Low','Close','Volume','MA5','MA10','RSI','Daily_Return','Volume_Change']
    X = data[features]
    y = data['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

  
    y_pred = rf.predict(X_test)
    accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)

   
    latest_data = X.iloc[-1:]
    prediction = rf.predict(latest_data)
    direction = "UP" if prediction[0] == 1 else "DOWN"
    icon = "📈" if direction == "UP" else "📉"
    result = f"{company_name} Stock will go {direction} {icon}"


    last60 = data.tail(60)
    close_dates = last60['Date'].dt.strftime('%b %d').tolist()
    close_prices = last60['Close'].round(2).tolist()
    ma5_vals = last60['MA5'].round(2).tolist()
    ma10_vals = last60['MA10'].round(2).tolist()

    
    rsi_dates = last60.tail(30)['Date'].dt.strftime('%b %d').tolist()
    rsi_vals = last60.tail(30)['RSI'].round(2).tolist()

  
    vol_dates = last60.tail(30)['Date'].dt.strftime('%b %d').tolist()
    vol_vals = last60.tail(30)['Volume'].tolist()

  
    importances = rf.feature_importances_
    feat_imp = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
    feat_labels = [f[0] for f in feat_imp]
    feat_values = [round(f[1]*100, 2) for f in feat_imp]

   
    current_price = round(float(data['Close'].iloc[-1]), 2)
    price_change = round(float(data['Daily_Return'].iloc[-1]) * 100, 2)
    current_rsi = round(float(data['RSI'].iloc[-1]), 2)
    avg_volume = int(data['Volume'].tail(10).mean())

    chart_data = {
        "close_dates": close_dates,
        "close_prices": close_prices,
        "ma5_vals": ma5_vals,
        "ma10_vals": ma10_vals,
        "rsi_dates": rsi_dates,
        "rsi_vals": rsi_vals,
        "vol_dates": vol_dates,
        "vol_vals": vol_vals,
        "feat_labels": feat_labels,
        "feat_values": feat_values,
    }

    return render_template(
        "share.html",
        prediction_text=result,
        direction=direction,
        accuracy=accuracy,
        company=company_name,
        current_price=current_price,
        price_change=price_change,
        current_rsi=current_rsi,
        avg_volume=avg_volume,
        chart_data=json.dumps(chart_data),
        show_charts=True
    )

if __name__ == "__main__":
    app.run(debug=True)