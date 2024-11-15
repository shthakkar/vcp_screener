from flask import Flask, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import yfinance as yf
from models import db, Stock

app = Flask(__name__)
CORS(app)

# Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize the database
db.init_app(app)


def fetch_and_store_stock_data():
    """Fetch AAPL stock data from Yahoo Finance and store it in SQLite DB."""
    stock_data = yf.Ticker("AAPL").history(period="1d")
    if not stock_data.empty:
        latest = stock_data.iloc[0]
        stock = Stock(
            name="AAPL",
            open_price=latest['Open'],
            close_price=latest['Close'],
            high_price=latest['High'],
            low_price=latest['Low']
        )
        db.session.add(stock)
        db.session.commit()


@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the VCP Screener API!"}), 200


@app.route('/api/initialize', methods=['GET'])
def initialize_db():
    """Endpoint to initialize the database and fetch stock data."""
    db.create_all()
    fetch_and_store_stock_data()
    return jsonify({"message": "Database initialized and stock data fetched"}), 200


@app.route('/api/stocks', methods=['GET'])
def get_stocks():
    """Return stock data from the database."""
    stocks = Stock.query.all()
    results = [
        {
            "name": stock.name,
            "open_price": stock.open_price,
            "close_price": stock.close_price,
            "high_price": stock.high_price,
            "low_price": stock.low_price
        } for stock in stocks
    ]
    return jsonify(results)


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
