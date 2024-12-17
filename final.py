import streamlit as st
import base64
import yfinance as yf
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model # type: ignore
import plotly.graph_objects as go
 
# Set page configuration Logo and web application title ke liye hai
st.set_page_config(
    page_title="Stock Price Analysis and Prediction",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)
 
# base64 encoding background image keliye hai
def get_base64(file):
    with open(file, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()
 
# For Background image on dashboard
image_file = "bgi.jpg"  # Use the path of your image here
background_image = get_base64(image_file)
 
# CSS added in streamlit  for design, css done by soniya and aniket 
background_style = f"""
<style>
.stApp {{
    background: url(data:image/jpg;base64,{background_image});
    background-size: cover;
    background-position: center;
    height: 100vh;
}}
 
.stButton>button {{
    background-color: #334eac;
    color: white;
    padding: 15px 30px;
    border: none;
    border-radius: 10px;
    font-size: 18px;
    cursor: pointer;
    transition: background-color 0.3s ease;
    width: 100%;
    max-width: 250px;
}}
 
 
.stButton>button:hover {{
    background-color: #0056b3;
}}
 
.stButton>button:active {{
    background-color: #003366;
}}
 
.button-container {{
    display: flex;
    flex-wrap: wrap;  /* Allow buttons to wrap on smaller screens */
    justify-content: space-between;  /* Distribute buttons evenly across available space */ 
    gap: 20px;  /* Space between buttons */
    position: sticky;
    top: 0;
    z-index: 10;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    margin-top: 50px;
    margin-left: 10px;
    margin-right: 10px;
    background-color: #007bff; /* Blue Color */
}}
 
.container {{
    text-align: center;  /* Left-align text */
    padding: 40px;
    border-radius: 15px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    margin-top: None;
    max-width: 1200px;
    margin-left: auto;
    margin-right: auto;
    color:#d1e5f4;
}}
 
h2, h3, p {{
    text-align: left !important;  /* Left-align headers and paragraphs */
}}

</style>
"""
 
 
# here we apply css to streamlit
st.markdown(background_style, unsafe_allow_html=True)
 
# home page buttons
def home_page():
    # This is for the heading
    st.markdown("""<h1 class="container"> WELCOME TO <BR> STOCK PRICE PREDICTOR CUM ANALYST</h1>""", unsafe_allow_html=True)

    # to add buttons on home screen in same line 
    col1, col2, col3, col4, col5= st.columns(5)
 
    with col1:
        if st.button("Stock Price Prediction"):
            st.session_state.page = "Stock Price Prediction"
   
    with col2:
        if st.button("Real-Time Stock Data"):
            st.session_state.page = "Real-Time Stock Data"
   
    with col3:
        if st.button("Stock Price Calculator"):
            st.session_state.page = "Stock Price Calculator"
   
    with col4:
        if st.button("News Alerts"):
            st.session_state.page = "News Alerts"
   
    st.markdown('', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 18px; color: #C8C8C8; ">This model is prepared by Group 15.</p>', unsafe_allow_html=True)
 
 
    # page selection to display
    if "page" in st.session_state:
        if st.session_state.page == "Stock Price Prediction":
            stock_prediction()
        elif st.session_state.page == "Real-Time Stock Data":
            real_time_stock()
        elif st.session_state.page == "Stock Price Calculator":
            future_price()
        elif st.session_state.page == "News Alerts":
            news()
        else:
            home_page()
 
# Function to run model
def stock_prediction():
    model = load_model('Stock Price Predictions Model.keras')  # to load model
    st.header('Stock Price Prediction')
    stock = st.text_input('Enter Stock Symbol', 'AMZN')
    start = '2013-12-31'
    end = datetime.today().strftime('%Y-%m-%d')
    data = yf.download(stock, start, end)
   
    # to display stock data
    st.subheader('Stock Data')
    st.write(data)
 
    # for prediction(data preparing)
    data_train = pd.DataFrame(data.Close[0: int(len(data) * 0.80)])
    data_test = pd.DataFrame(data.Close[int(len(data) * 0.80): len(data)])
    scaler = MinMaxScaler(feature_range=(0, 1))
   
    pas_100_days = data_train.tail(100)
    data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
    data_test_scale = scaler.fit_transform(data_test)
 
    # Moving averages plot
    def plot_moving_averages(data, ma_days, title, colors):
        fig = go.Figure()
        for ma, color in zip(ma_days, colors):
            fig.add_trace(go.Scatter(x=data.index, y=data.Close.rolling(ma).mean(), mode='lines', name=f'MA {ma} days', line=dict(color=color)))
        fig.add_trace(go.Scatter(x=data.index, y=data.Close, mode='lines', name='Actual Price', line=dict(color='green')))
        fig.update_layout(title=title, xaxis_title='Time', yaxis_title='Price', hovermode='x')
        st.plotly_chart(fig)
 # plot moving average
    plot_moving_averages(data, [50], 'Price vs MA50', ['red'])
    plot_moving_averages(data, [50, 100], 'Price vs MA50 vs MA100', ['red', 'blue'])
    plot_moving_averages(data, [100, 200], 'Price vs MA100 vs MA200', ['red', 'blue'])
   
    # Prediction logic
    x = []
    y = []
   
    for i in range(100, data_test_scale.shape[0]):
        x.append(data_test_scale[i - 100:i])
        y.append(data_test_scale[i, 0])
   
    x, y = np.array(x), np.array(y)
    predict = model.predict(x)
    scale = 1 / scaler.scale_
    predict = predict * scale
    y = y * scale
   
    # Plot predictions vs original
    st.subheader('Original vs Predicted Price')
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(predict, 'r', label='Predicted Price')
    ax.plot(y, 'g', label='Original Price')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.legend()
    ax.legend(facecolor='black', edgecolor='white')
    ax.grid(False)
    st.pyplot(fig)
   
    # Future stock predictions
    future_dates = pd.date_range(end, periods=25, freq='15D')
    future_predictions = []
    last_100_days = data_test_scale[-100:]
   
    for _ in range(25):
        future_x = last_100_days[-100:]
        future_x = np.expand_dims(future_x, axis=0)
        future_pred = model.predict(future_x)
        future_predictions.append(future_pred[0][0])
        last_100_days = np.append(last_100_days, future_pred[0][0]).reshape(-1, 1)
   
    future_predictions = np.array(future_predictions) * scale

    # Plot future predictions
    st.subheader('Future Stock Price Predictions')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=future_dates, y=future_predictions, mode='lines', name='Future Predicted Price', line=dict(color='blue')))
    fig.update_layout(title='Future Stock Price Predictions', xaxis_title='Date', yaxis_title='Price', hovermode='x')
    st.plotly_chart(fig)
 
    # Conclusion and recommendation
    st.subheader('Conclusion and Recommendation')
    latest_prediction = future_predictions[-1]
    current_price = data['Close'].iloc[-1]
   
    if latest_prediction > current_price:
        st.write(f"The predicted future price for {stock} is higher than the current price. It may be a good idea to consider buying this stock.")
    else:
        st.write(f"The predicted future price for {stock} is lower than or close to the current price. It may be better to wait or look for other opportunities.")
    
# Real-time stock data display
def real_time_stock():
    st.header('Real-Time Stock Data')
    ticker = st.text_input("Enter Stock Ticker", "AMZN")
   
    if ticker:
        try:
            stock_data = yf.Ticker(ticker).history(period="1d", interval="1m")
            real_time_price = stock_data['Close'].iloc[-1]
            volume = stock_data['Volume'].iloc[-1]
            market_cap = yf.Ticker(ticker).info['marketCap']
            previous_close = stock_data['Close'].iloc[-2]
            info = yf.Ticker(ticker).info
           
            # Display stock info
            st.subheader("Stock Information")
            st.write(f"**Name:** {info.get('shortName', 'N/A')}")
            st.write(f"**Sector:** {info.get('sector', 'N/A')}")
            st.write(f"**Industry:** {info.get('industry', 'N/A')}")
           
            # Real-time stock price
            st.subheader(f"Real-Time Price for {ticker}")
            st.write(f"Current Price: ${real_time_price:.2f}")
            st.write(f"Volume: {volume:,}")
            st.write(f"Market Cap: ${market_cap:,}")
            st.write(f"Previous Close: ${previous_close:.2f}")
           
            # Performance metrics
            st.subheader("Performance")
            st.write(f"**52 Week High:** {info.get('fiftyTwoWeekHigh', 'N/A')}")
            st.write(f"**52 Week Low:** {info.get('fiftyTwoWeekLow', 'N/A')}")
           
            # Price change
            price_change = real_time_price - previous_close
            if price_change > 0:
                st.markdown(f"**Price Change:** +${price_change:.2f} (+{(price_change / previous_close) * 100:.2f}%)")
            else:
                st.markdown(f"**Price Change:** ${price_change:.2f} ({(price_change / previous_close) * 100:.2f}%)")
           
            # Summary
            summary = info.get('longBusinessSummary', 'Summary not available')
            st.subheader("Brief Overview")
            st.write(summary)
           
            # Plot real-time graph
            st.subheader("Real-Time Graph")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close'))
            fig.update_layout(title=f"Real-Time Graph for {ticker}", xaxis_title='Time', yaxis_title='Price')
            st.plotly_chart(fig)
           
        except Exception as e:
            st.error(f"Error fetching real-time data for {ticker}: {str(e)}")
 
# Stock Price Calculator
def calculate_future_price(stock_code, years):
    stock = yf.Ticker(stock_code)
    history = stock.history(period='5y')  # Fetch historical data for the past 5 years

    if history.empty:
        raise ValueError(f"No price data found for {stock_code}")

    current_price = history['Close'].iloc[-1]
    annual_returns = history['Close'].pct_change().dropna().resample('Y').mean()
    average_growth_rate = np.mean(annual_returns)

    future_price = current_price * (1 + average_growth_rate) ** years
    return current_price, average_growth_rate, future_price, history


def future_price():
    st.header('Stock Price Calculator')
    # ticker = st.text_input('Enter Stock Ticker', 'AMZN')
    stock_code = st.text_input('Stock Code', 'AMZN')
    years = st.number_input('Years', min_value=1, max_value=100, value=5)
    num_stocks = st.number_input('Number of Stocks', min_value=1, value=10)

 
    # Fetch current price using yfinance
    try:
            current_price, average_growth_rate, future_price, history = calculate_future_price(stock_code, years)
            
            price_difference = future_price - current_price
            current_investment = current_price * num_stocks
            future_value = future_price * num_stocks
            profit = future_value - current_investment

            st.write(f'**Current Price (USD) :** ${current_price:.2f}')
            st.write(f'**Average Annual Growth Rate:** {average_growth_rate:.2%}')
            st.write(f'**Future Price after {years} years (USD):** ${future_price:.2f}')
            st.write(f'**Price Difference (USD):** ${price_difference:.2f}')
            st.write(f'**Current Investment for {num_stocks} stocks (USD):** ${current_investment:.2f}')
            st.write(f'**Future Value of {num_stocks} stocks after {years} years (USD):** ${future_value:.2f}')
            st.write(f'**Profit (USD):** ${profit:.2f}')

    except Exception as e:
            st.error(f'Error: {e}')
 
# News
def fetch_stock_news(ticker):
    # Using yfinance to fetch news
    try:
        stock = yf.Ticker(ticker)
        news = stock.news  # Fetching news articles for the stock
        return news
    except Exception as e:
        raise Exception(f"Error fetching news for {ticker}: {str(e)}")
 
def news():
    st.header("News Alerts")
    st.write("""
    Stay updated with the latest stock news and alerts.
    You can configure news notifications for your selected stock symbols.
    """)
 
    ticker = st.text_input("Enter Stock Ticker for News", "AMZN")
    try:
        news = fetch_stock_news(ticker)
        st.subheader(f"Latest News for {ticker}")
        for article in news:
            st.markdown(f"**{article['title']}**")
            st.markdown(f"[Read more]({article['link']})")
    except Exception as e:
        st.error(f"Error fetching news for {ticker}: {str(e)}")
 
# call  home page function
home_page()
 
