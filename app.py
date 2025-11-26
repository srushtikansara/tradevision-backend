from flask import Flask, request, jsonify
from flask_jwt_extended import *
import pandas as pd
import joblib
import datetime

from flask_cors import CORS

app = Flask(__name__)
CORS(app)

app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = "your_jwt_secret"
jwt = JWTManager(app)

users_db = {}  # Simple in-memory store for demo

def engineer_features(df):
    df = df.sort_values(['Stock', 'timestamp'])
    df['next_close'] = df.groupby('Stock')['close'].shift(-1)
    df['movement'] = (df['next_close'] > df['close']).astype(int)
    df['ma5'] = df.groupby('Stock')['close'].transform(lambda x: x.rolling(5).mean())
    df['std5'] = df.groupby('Stock')['close'].transform(lambda x: x.rolling(5).std())
    df['vol_ma5'] = df.groupby('Stock')['volume'].transform(lambda x: x.rolling(5).mean())
    df = df.dropna()
    return df

def predict_best_stock(today, df, model):
    today_df = df[df['timestamp'] == today]
    if today_df.empty:
        return None, None

    features = ['open', 'high', 'low', 'close', 'volume', 'ma5', 'std5', 'vol_ma5']
    X_today = today_df[features]
    pred = model.predict(X_today)
    today_df = today_df.assign(pred_movement=pred)

    up_stocks = today_df[today_df['pred_movement'] == 1]
    if not up_stocks.empty:
        best = up_stocks.sort_values('ma5', ascending=False).iloc[0]
    else:
        best = today_df.iloc[0]

    out = {
        "symbol": best['Stock'],
        "open": best['open'],
        "close": best['close'],
        "pred_movement": 'up' if best['pred_movement'] == 1 else 'down'
    }
    return out, today_df

# Load data/model on start
df = pd.read_csv("combined_stocks.csv", parse_dates=['timestamp'])
df = engineer_features(df)
model = joblib.load("rf_stock_model.pkl")

@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    if data['email'] in users_db:
        return jsonify({'msg': 'User exists'}), 409
    users_db[data['email']] = data['password']
    return jsonify({'msg': 'Signed up'}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    if users_db.get(data['email']) == data['password']:
        access = create_access_token(identity=data['email'])
        return jsonify({'access_token': access}), 200
    return jsonify({'msg': 'Bad credentials'}), 401

@app.route('/predict', methods=['GET'])
@jwt_required()
def predict():
    today = df['timestamp'].max()
    out, _ = predict_best_stock(today, df, model)
    if out is None:
        return jsonify({'msg': 'No data for today'}), 404
    return jsonify(out), 200

@app.route('/stocks', methods=['GET'])
@jwt_required()
def get_stocks():
    return jsonify(df.to_dict(orient='records'))

@app.route('/chart/<symbol>', methods=['GET'])
@jwt_required()
def chart(symbol):
    d = df[df['Stock'] == symbol].sort_values('timestamp').tail(15)
    return jsonify(d.to_dict(orient='records'))

from flask import Flask, request, jsonify
from flask_jwt_extended import *
import pandas as pd
import joblib
import datetime

app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = "your_jwt_secret"
jwt = JWTManager(app)

users_db = {}  # Simple in-memory store for demo

def engineer_features(df):
    df = df.sort_values(['Stock', 'timestamp'])
    df['next_close'] = df.groupby('Stock')['close'].shift(-1)
    df['movement'] = (df['next_close'] > df['close']).astype(int)
    df['ma5'] = df.groupby('Stock')['close'].transform(lambda x: x.rolling(5).mean())
    df['std5'] = df.groupby('Stock')['close'].transform(lambda x: x.rolling(5).std())
    df['vol_ma5'] = df.groupby('Stock')['volume'].transform(lambda x: x.rolling(5).mean())
    df = df.dropna()
    return df

def predict_best_stock(today, df, model):
    today_df = df[df['timestamp'] == today]
    if today_df.empty:
        return None, None

    features = ['open', 'high', 'low', 'close', 'volume', 'ma5', 'std5', 'vol_ma5']
    X_today = today_df[features]
    pred = model.predict(X_today)
    today_df = today_df.assign(pred_movement=pred)

    up_stocks = today_df[today_df['pred_movement'] == 1]
    if not up_stocks.empty:
        best = up_stocks.sort_values('ma5', ascending=False).iloc[0]
    else:
        best = today_df.iloc[0]

    out = {
        "symbol": best['Stock'],
        "open": best['open'],
        "close": best['close'],
        "pred_movement": 'up' if best['pred_movement'] == 1 else 'down'
    }
    return out, today_df

# Load data/model on start
df = pd.read_csv("combined_stocks.csv", parse_dates=['timestamp'])
df = engineer_features(df)
model = joblib.load("rf_stock_model.pkl")

@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    if data['email'] in users_db:
        return jsonify({'msg': 'User exists'}), 409
    users_db[data['email']] = data['password']
    return jsonify({'msg': 'Signed up'}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    if users_db.get(data['email']) == data['password']:
        access = create_access_token(identity=data['email'])
        return jsonify({'access_token': access}), 200
    return jsonify({'msg': 'Bad credentials'}), 401

@app.route('/predict', methods=['GET'])
@jwt_required()
def predict():
    today = df['timestamp'].max()
    out, _ = predict_best_stock(today, df, model)
    if out is None:
        return jsonify({'msg': 'No data for today'}), 404
    return jsonify(out), 200

@app.route('/stocks', methods=['GET'])
@jwt_required()
def get_stocks():
    return jsonify(df.to_dict(orient='records'))

@app.route('/chart/<symbol>', methods=['GET'])
@jwt_required()
def chart(symbol):
    d = df[df['Stock'] == symbol].sort_values('timestamp').tail(15)
    return jsonify(d.to_dict(orient='records'))

if __name__ == "__main__":
    # For local testing
    app.run(host="0.0.0.0", port=5000, debug=True)

