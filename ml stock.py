from flask import Flask, render_template, request
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

df = pd.read_csv("StockPriceDataset.csv")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    company_name = request.form["company"].upper()

    data = df[df['Ticker'] == company_name].copy()

    if data.empty:
        return render_template("index.html", prediction_text="Company not found")

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

    features = [
        'Open','High','Low','Close','Volume',
        'MA5','MA10','RSI','Daily_Return','Volume_Change'
    ]

    X = data[features]
    y = data['Target']

    X_train, X_test, y_train, y_test = train_test_split(
        X,y,test_size=0.2,shuffle=False
    )

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    latest_data = X.iloc[-1:]

    prediction = rf.predict(latest_data)

    if prediction[0] == 1:
        result = f"{company_name} Stock will go UP 📈"
    else:
        result = f"{company_name} Stock will go DOWN 📉"

    return render_template("index.html", prediction_text=result)


if __name__ == "__main__":
    app.run(debug=True)