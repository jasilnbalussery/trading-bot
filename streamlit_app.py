import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
import datetime
import time

# Streamlit app configuration
st.set_page_config(page_title="Live Trading Platform (Indian Stocks)", layout="wide")

# Function to fetch stock data
def fetch_data(ticker, interval='1m', period='1d'):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        if df.empty:
            raise ValueError(f"No data found for ticker {ticker}. Ensure the ticker is valid (e.g., RELIANCE.NS for NSE, RELIANCE.BO for BSE).")
        return df
    except Exception as e:
        raise Exception(f"Failed to fetch data for {ticker}: {str(e)}")

# Function to calculate indicators
def calculate_indicators(df):
    df['SMA20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
    df['EMA12'] = EMAIndicator(close=df['Close'], window=12).ema_indicator()
    df['EMA26'] = EMAIndicator(close=df['Close'], window=26).ema_indicator()
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    return df

# Function to detect simple candlestick patterns
def detect_candlestick_patterns(df):
    df['Bullish_Engulfing'] = (
        (df['Close'].shift(1) < df['Open'].shift(1)) & 
        (df['Close'] > df['Open']) & 
        (df['Close'] > df['Open'].shift(1)) & 
        (df['Open'] < df['Close'].shift(1))
    )
    df['Bearish_Engulfing'] = (
        (df['Close'].shift(1) > df['Open'].shift(1)) & 
        (df['Close'] < df['Open']) & 
        (df['Close'] < df['Open'].shift(1)) & 
        (df['Open'] > df['Close'].shift(1))
    )
    return df

# Function to generate trading signals
def generate_signals(df):
    df['Signal'] = 0
    df.loc[(df['RSI'] < 30) | (df['Bullish_Engulfing']) | 
           ((df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))), 'Signal'] = 1
    df.loc[(df['RSI'] > 70) | (df['Bearish_Engulfing']) | 
           ((df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1))), 'Signal'] = -1
    return df

# Function to calculate signal strength
def calculate_signal_strength(df):
    latest_rsi = df['RSI'].iloc[-1]
    latest_macd_diff = df['MACD'].iloc[-1] - df['MACD_Signal'].iloc[-1]
    # RSI strength: proximity to 30 (buy) or 70 (sell)
    rsi_buy_strength = max(0, (30 - latest_rsi) / 30 * 100) if latest_rsi < 50 else 0
    rsi_sell_strength = max(0, (latest_rsi - 70) / 30 * 100) if latest_rsi > 50 else 0
    # MACD strength: normalized difference
    macd_buy_strength = max(0, latest_macd_diff / df['Close'].iloc[-1] * 1000) if latest_macd_diff > 0 else 0
    macd_sell_strength = max(0, -latest_macd_diff / df['Close'].iloc[-1] * 1000) if latest_macd_diff < 0 else 0
    return {
        'RSI_Buy_Strength': rsi_buy_strength,
        'RSI_Sell_Strength': rsi_sell_strength,
        'MACD_Buy_Strength': macd_buy_strength,
        'MACD_Sell_Strength': macd_sell_strength
    }

# Function to simulate trading
def simulate_trade(df, capital=10000, position_size=0.1):
    position = 0
    balance = capital
    trades = []
    for i in range(1, len(df)):
        if df['Signal'].iloc[i] == 1 and position == 0:  # Buy
            shares = (balance * position_size) // df['Close'].iloc[i]
            cost = shares * df['Close'].iloc[i]
            balance -= cost
            position += shares
            trades.append({'Time': df.index[i], 'Type': 'Buy', 'Price': df['Close'].iloc[i], 'Shares': shares, 'Balance': balance})
        elif df['Signal'].iloc[i] == -1 and position > 0:  # Sell
            revenue = position * df['Close'].iloc[i]
            balance += revenue
            trades.append({'Time': df.index[i], 'Type': 'Sell', 'Price': df['Close'].iloc[i], 'Shares': position, 'Balance': balance})
            position = 0
    return trades, balance, position

# Plotting function
def plot_data(df):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Candlestick'
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], name='SMA20', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA12'], name='EMA12', line=dict(color='orange')))
    buy_signals = df[df['Signal'] == 1]
    sell_signals = df[df['Signal'] == -1]
    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Low'], mode='markers', 
                            name='Buy', marker=dict(symbol='triangle-up', size=10, color='green')))
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['High'], mode='markers', 
                            name='Sell', marker=dict(symbol='triangle-down', size=10, color='red')))
    fig.update_layout(title='Stock Price and Indicators', xaxis_title='Time', yaxis_title='Price', 
                      height=600, template='plotly_dark')
    return fig

# Main app
def main():
    st.title("Live Trading Platform with Trend Prediction (Indian Stocks)")
    
    # Sidebar for user inputs
    st.sidebar.header("Settings")
    st.sidebar.write("Note: For Indian stocks, use .NS for NSE (e.g., RELIANCE.NS) or .BO for BSE (e.g., RELIANCE.BO)")
    st.sidebar.write("Signals are based on the latest data, not future predictions. Check signal strength for trends.")
    st.sidebar.warning("Indian markets are open 9:15 AM–3:30 PM IST, Mon–Fri. Data may be stale outside these hours.")
    ticker = st.sidebar.text_input("Ticker Symbol", value="RELIANCE.NS")
    interval = st.sidebar.selectbox("Time Interval", ['1m', '5m', '15m', '1h'], index=0)
    capital = st.sidebar.number_input("Initial Capital (₹)", min_value=1000, value=10000)
    position_size = st.sidebar.slider("Position Size (% of capital)", 0.1, 1.0, 0.1)
    
    # Initialize session state for notification log and last signal
    if 'notification_log' not in st.session_state:
        st.session_state.notification_log = []
    if 'last_signal' not in st.session_state:
        st.session_state.last_signal = None
    
    # Fetch and process data
    try:
        df = fetch_data(ticker, interval=interval)
        df = calculate_indicators(df)
        df = detect_candlestick_patterns(df)
        df = generate_signals(df)
        
        # Display signal strength
        st.subheader("Signal Strength (Latest Data)")
        strength = calculate_signal_strength(df)
        st.write(f"RSI Buy Strength: {strength['RSI_Buy_Strength']:.2f}% (higher means closer to buy)")
        st.write(f"RSI Sell Strength: {strength['RSI_Sell_Strength']:.2f}% (higher means closer to sell)")
        st.write(f"MACD Buy Strength: {strength['MACD_Buy_Strength']:.2f} (higher means stronger buy trend)")
        st.write(f"MACD Sell Strength: {strength['MACD_Sell_Strength']:.2f} (higher means stronger sell trend)")
        
        # Display chart with unique key
        st.subheader(f"{ticker} Price Chart")
        st.plotly_chart(plot_data(df), use_container_width=True, key="initial_chart")
        
        # Simulate trading
        trades, final_balance, final_position = simulate_trade(df, capital, position_size)
        
        # Display trading results
        st.subheader("Trading Simulation Results")
        st.write(f"Initial Capital: ₹{capital:.2f}")
        st.write(f"Final Balance: ₹{final_balance:.2f}")
        st.write(f"Open Position: {final_position} shares")
        st.write(f"Profit/Loss: ₹{(final_balance + final_position * df['Close'].iloc[-1] - capital):.2f}")
        
        # Display trades
        st.subheader("Trade History")
        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            st.dataframe(trades_df)
        else:
            st.write("No trades executed.")
        
        # Notification log display
        st.subheader("Notification Log")
        if st.session_state.notification_log:
            st.dataframe(pd.DataFrame(st.session_state.notification_log))
        else:
            st.write("No notifications yet.")
        
        # Live update toggle
        if st.sidebar.checkbox("Enable Live Updates", value=False):
            st.subheader("Live Mode")
            placeholder = st.empty()
            while True:
                df = fetch_data(ticker, interval=interval)
                df = calculate_indicators(df)
                df = detect_candlestick_patterns(df)
                df = generate_signals(df)
                
                # Check for new signal
                latest_signal = df['Signal'].iloc[-1]
                if latest_signal != st.session_state.last_signal and latest_signal != 0:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    if latest_signal == 1:
                        st.success(f"BUY Signal for {ticker} at ₹{df['Close'].iloc[-1]:.2f} on {timestamp}")
                        st.session_state.notification_log.append({
                            'Time': timestamp,
                            'Type': 'Buy',
                            'Price': df['Close'].iloc[-1],
                            'Ticker': ticker
                        })
                    elif latest_signal == -1:
                        st.warning(f"SELL Signal for {ticker} at ₹{df['Close'].iloc[-1]:.2f} on {timestamp}")
                        st.session_state.notification_log.append({
                            'Time': timestamp,
                            'Type': 'Sell',
                            'Price': df['Close'].iloc[-1],
                            'Ticker': ticker
                        })
                    st.session_state.last_signal = latest_signal
                
                with placeholder.container():
                    st.plotly_chart(plot_data(df), use_container_width=True, key=f"live_chart_{int(time.time())}")
                time.sleep(60)  # Update every minute
    
    except Exception as e:
        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
