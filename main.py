from flask import Flask, render_template, request
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import io
import base64
from datetime import datetime, timedelta
import numpy as np
import matplotlib


matplotlib.use('Agg')

app = Flask(__name__)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

def preprocess_text(text):
    return tokenizer(text, padding='max_length', max_length=128, truncation=True, return_tensors="pt")

def get_sentiment(text):
    inputs = preprocess_text(text)
    outputs = model(**inputs)
    logits = outputs.logits
    sentiment = torch.argmax(logits, dim=1).item()
    return sentiment  # 0: negative, 1: neutral, 2: positive

def sentiment_to_label(sentiment):
    labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    return labels[sentiment]

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    news_text = request.form['news_text']
    article_date = request.form['article_date']
    ticker_symbol = request.form['ticker_symbol'].upper()
    sentiment_value = get_sentiment(news_text)
    sentiment_label = sentiment_to_label(sentiment_value)
    
    sentiment_plot_value = sentiment_value - 1 
    news_data = pd.DataFrame({
        'Date': [article_date],
        'News_Headline': [news_text],
        'Sentiment': [sentiment_label]
    })

    article_date = pd.to_datetime(article_date)
    start_date = article_date - timedelta(days=4)  # Expanding the date range
    end_date = article_date + timedelta(days=4)

    try:
        stock_data = yf.download(ticker_symbol, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        stock_data.reset_index(inplace=True)
        print(f"Stock data retrieved:\n{stock_data}")  # Debug output
    except Exception as e:
        return render_template('index.html', sentiment=sentiment_label, article_date=article_date.strftime('%Y-%m-%d'), error=f"Failed to retrieve stock data: {str(e)}")
    
    if stock_data.empty:
        return render_template('index.html', sentiment=sentiment_label, article_date=article_date.strftime('%Y-%m-%d'), error="No stock data available for the given date range.")

    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    nearest_date = stock_data.loc[(stock_data['Date'] - article_date).abs().idxmin()]['Date']
    print(f"Nearest available date: {nearest_date}")  

    valid_date_range = (nearest_date - timedelta(days=2), nearest_date + timedelta(days=2))
    stock_data = stock_data[(stock_data['Date'] >= valid_date_range[0]) & (stock_data['Date'] <= valid_date_range[1])]
    print(f"Filtered stock data for valid date range:\n{stock_data}")  

    scaler = MinMaxScaler(feature_range=(-1, 1))
    stock_data['Scaled_Close'] = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))

    fig, ax = plt.subplots(figsize=(14, 7))
    
    sns.lineplot(data=stock_data, x='Date', y='Scaled_Close', label='Scaled Close Price', marker='o', ax=ax)
    
    ax.scatter([nearest_date], [sentiment_plot_value], color='orange', label='Sentiment Score', marker='x', s=100)
    
    ax.set_ylim(-1.2, 1.2)
    ax.set_title(f'Sentiment vs. Stock Price for {ticker_symbol}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Scaled Values')
    ax.legend()

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    plt.close(fig)

    return render_template('index.html', sentiment=sentiment_label, article_date=nearest_date.strftime('%Y-%m-%d'), image_base64=image_base64, ticker_symbol=ticker_symbol)

if __name__ == '__main__':
    app.run(debug=True)
