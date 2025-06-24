import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import datetime
import pytz
import requests
import time
from ta.trend import EMAIndicator, SMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
import warnings
warnings.filterwarnings('ignore')

# Telegram Configuration
TELEGRAM_BOT_TOKEN = "7807998203:AAFn7dhHmnIMod6r6akNFfia5hGfCAGUTfY"
TELEGRAM_CHAT_ID = "5251360456"

class CandlestickPatterns:
    """Advanced Candlestick Pattern Recognition"""
    
    @staticmethod
    def identify_patterns(df):
        """Identify multiple candlestick patterns"""
        patterns = {}
        
        try:
            # Single candlestick patterns
            patterns['Doji'] = CandlestickPatterns.doji(df)
            patterns['Hammer'] = CandlestickPatterns.hammer(df)
            patterns['Shooting_Star'] = CandlestickPatterns.shooting_star(df)
            patterns['Marubozu'] = CandlestickPatterns.marubozu(df)
            
            # Multi-candlestick patterns
            patterns['Bullish_Engulfing'] = CandlestickPatterns.bullish_engulfing(df)
            patterns['Bearish_Engulfing'] = CandlestickPatterns.bearish_engulfing(df)
            patterns['Morning_Star'] = CandlestickPatterns.morning_star(df)
            patterns['Evening_Star'] = CandlestickPatterns.evening_star(df)
            patterns['Three_White_Soldiers'] = CandlestickPatterns.three_white_soldiers(df)
            patterns['Three_Black_Crows'] = CandlestickPatterns.three_black_crows(df)
            
            return patterns
        except Exception as e:
            st.error(f"Pattern recognition error: {e}")
            return {}
    
    @staticmethod
    def doji(df, tolerance=0.1):
        """Identify Doji patterns"""
        body_size = abs(df['Close'] - df['Open'])
        range_size = df['High'] - df['Low']
        return (body_size <= tolerance * range_size) & (range_size > 0)
    
    @staticmethod
    def hammer(df):
        """Identify Hammer patterns"""
        body_size = abs(df['Close'] - df['Open'])
        upper_shadow = df['High'] - df[['Open', 'Close']].max(axis=1)
        lower_shadow = df[['Open', 'Close']].min(axis=1) - df['Low']
        
        return (lower_shadow >= 2 * body_size) & (upper_shadow <= 0.1 * body_size)
    
    @staticmethod
    def shooting_star(df):
        """Identify Shooting Star patterns"""
        body_size = abs(df['Close'] - df['Open'])
        upper_shadow = df['High'] - df[['Open', 'Close']].max(axis=1)
        lower_shadow = df[['Open', 'Close']].min(axis=1) - df['Low']
        
        return (upper_shadow >= 2 * body_size) & (lower_shadow <= 0.1 * body_size)
    
    @staticmethod
    def marubozu(df, tolerance=0.05):
        """Identify Marubozu patterns"""
        body_size = abs(df['Close'] - df['Open'])
        range_size = df['High'] - df['Low']
        
        return body_size >= (1 - tolerance) * range_size
    
    @staticmethod
    def bullish_engulfing(df):
        """Identify Bullish Engulfing patterns"""
        if len(df) < 2:
            return pd.Series([False] * len(df), index=df.index)
        
        prev_bearish = df['Close'].shift(1) < df['Open'].shift(1)
        curr_bullish = df['Close'] > df['Open']
        engulfing = (df['Open'] < df['Close'].shift(1)) & (df['Close'] > df['Open'].shift(1))
        
        return prev_bearish & curr_bullish & engulfing
    
    @staticmethod
    def bearish_engulfing(df):
        """Identify Bearish Engulfing patterns"""
        if len(df) < 2:
            return pd.Series([False] * len(df), index=df.index)
        
        prev_bullish = df['Close'].shift(1) > df['Open'].shift(1)
        curr_bearish = df['Close'] < df['Open']
        engulfing = (df['Open'] > df['Close'].shift(1)) & (df['Close'] < df['Open'].shift(1))
        
        return prev_bullish & curr_bearish & engulfing
    
    @staticmethod
    def morning_star(df):
        """Identify Morning Star patterns"""
        if len(df) < 3:
            return pd.Series([False] * len(df), index=df.index)
        
        first_bearish = df['Close'].shift(2) < df['Open'].shift(2)
        small_body = abs(df['Close'].shift(1) - df['Open'].shift(1)) < abs(df['Close'].shift(2) - df['Open'].shift(2)) * 0.3
        third_bullish = df['Close'] > df['Open']
        gap_down = df['High'].shift(1) < df['Low'].shift(2)
        gap_up = df['Low'] > df['High'].shift(1)
        
        return first_bearish & small_body & third_bullish & gap_down & gap_up
    
    @staticmethod
    def evening_star(df):
        """Identify Evening Star patterns"""
        if len(df) < 3:
            return pd.Series([False] * len(df), index=df.index)
        
        first_bullish = df['Close'].shift(2) > df['Open'].shift(2)
        small_body = abs(df['Close'].shift(1) - df['Open'].shift(1)) < abs(df['Close'].shift(2) - df['Open'].shift(2)) * 0.3
        third_bearish = df['Close'] < df['Open']
        gap_up = df['Low'].shift(1) > df['High'].shift(2)
        gap_down = df['High'] < df['Low'].shift(1)
        
        return first_bullish & small_body & third_bearish & gap_up & gap_down
    
    @staticmethod
    def three_white_soldiers(df):
        """Identify Three White Soldiers pattern"""
        if len(df) < 3:
            return pd.Series([False] * len(df), index=df.index)
        
        bullish_1 = df['Close'].shift(2) > df['Open'].shift(2)
        bullish_2 = df['Close'].shift(1) > df['Open'].shift(1)
        bullish_3 = df['Close'] > df['Open']
        
        higher_closes = (df['Close'].shift(1) > df['Close'].shift(2)) & (df['Close'] > df['Close'].shift(1))
        higher_opens = (df['Open'].shift(1) > df['Open'].shift(2)) & (df['Open'] > df['Open'].shift(1))
        
        return bullish_1 & bullish_2 & bullish_3 & higher_closes & higher_opens
    
    @staticmethod
    def three_black_crows(df):
        """Identify Three Black Crows pattern"""
        if len(df) < 3:
            return pd.Series([False] * len(df), index=df.index)
        
        bearish_1 = df['Close'].shift(2) < df['Open'].shift(2)
        bearish_2 = df['Close'].shift(1) < df['Open'].shift(1)
        bearish_3 = df['Close'] < df['Open']
        
        lower_closes = (df['Close'].shift(1) < df['Close'].shift(2)) & (df['Close'] < df['Close'].shift(1))
        lower_opens = (df['Open'].shift(1) < df['Open'].shift(2)) & (df['Open'] < df['Open'].shift(1))
        
        return bearish_1 & bearish_2 & bearish_3 & lower_closes & lower_opens

class TechnicalAnalysis:
    """Advanced Technical Analysis with Multiple Indicators"""
    
    @staticmethod
    def calculate_all_indicators(df):
        """Calculate comprehensive technical indicators"""
        try:
            # Trend Indicators
            df['EMA12'] = EMAIndicator(close=df['Close'], window=12).ema_indicator()
            df['EMA26'] = EMAIndicator(close=df['Close'], window=26).ema_indicator()
            df['SMA20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
            df['SMA50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
            
            # MACD
            macd = MACD(close=df['Close'])
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            df['MACD_Hist'] = macd.macd_diff()
            
            # ADX (Trend Strength)
            adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'])
            df['ADX'] = adx.adx()
            df['DI_Plus'] = adx.adx_pos()
            df['DI_Minus'] = adx.adx_neg()
            
            # Momentum Indicators
            df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
            
            # Stochastic
            stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'])
            df['Stoch_K'] = stoch.stoch()
            df['Stoch_D'] = stoch.stoch_signal()
            
            # Bollinger Bands
            bb = BollingerBands(close=df['Close'])
            df['BB_Upper'] = bb.bollinger_hband()
            df['BB_Middle'] = bb.bollinger_mavg()
            df['BB_Lower'] = bb.bollinger_lband()
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
            
            # ATR (Volatility)
            df['ATR'] = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close']).average_true_range()
            
            # Volume Indicators
            df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()

            df['OBV'] = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
            
            # Support and Resistance
            df = TechnicalAnalysis.calculate_support_resistance(df)
            
            return df.dropna()
        
        except Exception as e:
            st.error(f"Error calculating indicators: {e}")
            return df
    
    @staticmethod
    def calculate_support_resistance(df, window=20):
        """Calculate dynamic support and resistance levels"""
        df['Resistance'] = df['High'].rolling(window=window).max()
        df['Support'] = df['Low'].rolling(window=window).min()
        return df

class SignalGenerator:
    """Advanced Signal Generation with Multiple Confirmations"""
    
    @staticmethod
    def generate_advanced_signals(df, patterns):
        """Generate buy/sell signals with multiple confirmations"""
        df['Signal'] = 0
        df['Signal_Strength'] = 0
        df['Signal_Reason'] = ''
        
        try:
            for i in range(50, len(df)):  # Start after indicators stabilize
                buy_score = 0
                sell_score = 0
                reasons = []
                
                # Current data
                curr = df.iloc[i]
                prev = df.iloc[i-1]
                
                # 1. Trend Analysis (Weight: 25%)
                if curr['EMA12'] > curr['SMA20'] and curr['Close'] > curr['EMA12']:
                    buy_score += 2
                    reasons.append("Uptrend")
                elif curr['EMA12'] < curr['SMA20'] and curr['Close'] < curr['EMA12']:
                    sell_score += 2
                    reasons.append("Downtrend")
                
                # 2. Momentum Analysis (Weight: 20%)
                if curr['RSI'] < 35 and curr['RSI'] > prev['RSI']:
                    buy_score += 2
                    reasons.append("RSI Oversold Recovery")
                elif curr['RSI'] > 65 and curr['RSI'] < prev['RSI']:
                    sell_score += 2
                    reasons.append("RSI Overbought Decline")
                
                # 3. MACD Analysis (Weight: 20%)
                if (curr['MACD'] > curr['MACD_Signal'] and 
                    prev['MACD'] <= prev['MACD_Signal']):
                    buy_score += 2
                    reasons.append("MACD Bullish Cross")
                elif (curr['MACD'] < curr['MACD_Signal'] and 
                      prev['MACD'] >= prev['MACD_Signal']):
                    sell_score += 2
                    reasons.append("MACD Bearish Cross")
                
                # 4. Volume Confirmation (Weight: 15%)
                if curr['Volume'] > curr['Volume_SMA'] * 1.2:
                    if buy_score > sell_score:
                        buy_score += 1
                        reasons.append("High Volume")
                    elif sell_score > buy_score:
                        sell_score += 1
                        reasons.append("High Volume")
                
                # 5. Bollinger Bands (Weight: 10%)
                if curr['Close'] <= curr['BB_Lower'] and curr['Close'] > prev['Close']:
                    buy_score += 1
                    reasons.append("BB Oversold Bounce")
                elif curr['Close'] >= curr['BB_Upper'] and curr['Close'] < prev['Close']:
                    sell_score += 1
                    reasons.append("BB Overbought Rejection")
                
                # 6. ADX Trend Strength (Weight: 10%)
                if curr['ADX'] > 25:  # Strong trend
                    if curr['DI_Plus'] > curr['DI_Minus'] and buy_score > 0:
                        buy_score += 1
                        reasons.append("Strong Uptrend (ADX)")
                    elif curr['DI_Minus'] > curr['DI_Plus'] and sell_score > 0:
                        sell_score += 1
                        reasons.append("Strong Downtrend (ADX)")
                
                # 7. Candlestick Patterns
                pattern_signals = SignalGenerator.evaluate_patterns(patterns, i)
                if pattern_signals['buy'] > 0:
                    buy_score += pattern_signals['buy']
                    reasons.extend(pattern_signals['buy_reasons'])
                if pattern_signals['sell'] > 0:
                    sell_score += pattern_signals['sell']
                    reasons.extend(pattern_signals['sell_reasons'])
                
                # Generate final signal
                if buy_score >= 4 and buy_score > sell_score:
                    df.iloc[i, df.columns.get_loc('Signal')] = 1
                    df.iloc[i, df.columns.get_loc('Signal_Strength')] = min(buy_score, 10)
                    df.iloc[i, df.columns.get_loc('Signal_Reason')] = '; '.join(reasons)
                elif sell_score >= 4 and sell_score > buy_score:
                    df.iloc[i, df.columns.get_loc('Signal')] = -1
                    df.iloc[i, df.columns.get_loc('Signal_Strength')] = min(sell_score, 10)
                    df.iloc[i, df.columns.get_loc('Signal_Reason')] = '; '.join(reasons)
            
            # Remove consecutive same signals
            df = SignalGenerator.filter_consecutive_signals(df)
            
            return df
        
        except Exception as e:
            st.error(f"Error generating signals: {e}")
            return df
    
    @staticmethod
    def evaluate_patterns(patterns, index):
        """Evaluate candlestick patterns for signals"""
        result = {'buy': 0, 'sell': 0, 'buy_reasons': [], 'sell_reasons': []}
        
        try:
            # Bullish patterns
            bullish_patterns = ['Hammer', 'Bullish_Engulfing', 'Morning_Star', 'Three_White_Soldiers']
            for pattern in bullish_patterns:
                if pattern in patterns and index < len(patterns[pattern]) and patterns[pattern].iloc[index]:
                    result['buy'] += 1
                    result['buy_reasons'].append(f"{pattern.replace('_', ' ')}")
            
            # Bearish patterns
            bearish_patterns = ['Shooting_Star', 'Bearish_Engulfing', 'Evening_Star', 'Three_Black_Crows']
            for pattern in bearish_patterns:
                if pattern in patterns and index < len(patterns[pattern]) and patterns[pattern].iloc[index]:
                    result['sell'] += 1
                    result['sell_reasons'].append(f"{pattern.replace('_', ' ')}")
            
            return result
        except:
            return result
    
    @staticmethod
    def filter_consecutive_signals(df):
        """Remove consecutive signals of the same type"""
        for i in range(1, len(df)):
            if (df.iloc[i]['Signal'] != 0 and 
                df.iloc[i]['Signal'] == df.iloc[i-1]['Signal']):
                df.iloc[i, df.columns.get_loc('Signal')] = 0
                df.iloc[i, df.columns.get_loc('Signal_Strength')] = 0
                df.iloc[i, df.columns.get_loc('Signal_Reason')] = ''
        
        return df

def send_telegram_alert(message):
    """Send alert to Telegram"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        response = requests.post(url, data=data, timeout=10)
        response.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Telegram Error: {e}")
        return False

def fetch_stock_data(ticker, interval='5m', period='5d'):
    """Fetch stock data with error handling"""
    try:
        ticker = ticker.upper()
        data = yf.download(ticker, interval=interval, period=period, progress=False)
        
        if data.empty:
            raise ValueError(f"No data found for {ticker}")
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        
        data = data.dropna()
        
        if len(data) < 50:
            raise ValueError(f"Insufficient data: {len(data)} candles. Need at least 50.")
        
        return data
    
    except Exception as e:
        raise Exception(f"Error fetching {ticker}: {str(e)}")

def create_advanced_chart(df, patterns):
    """Create comprehensive trading chart"""
    try:
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            row_heights=[0.4, 0.2, 0.2, 0.2],
            vertical_spacing=0.02,
            subplot_titles=("Price Chart with Patterns & Signals", "MACD", "RSI & Stochastic", "Volume & ATR")
        )
        
        # Main price chart
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'], name='Price'
        ), row=1, col=1)
        
        # Moving averages
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA12'], name='EMA12', 
                               line=dict(color='orange', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], name='SMA20', 
                               line=dict(color='blue', width=1)), row=1, col=1)
        
        # Bollinger Bands
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper', 
                               line=dict(color='gray', dash='dot'), opacity=0.5), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower', 
                               line=dict(color='gray', dash='dot'), opacity=0.5), row=1, col=1)
        
        # Support and Resistance
        fig.add_trace(go.Scatter(x=df.index, y=df['Resistance'], name='Resistance', 
                               line=dict(color='red', dash='dash'), opacity=0.3), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Support'], name='Support', 
                               line=dict(color='green', dash='dash'), opacity=0.3), row=1, col=1)
        
        # Buy/Sell signals
        buy_signals = df[df['Signal'] == 1]
        sell_signals = df[df['Signal'] == -1]
        
        if not buy_signals.empty:
            fig.add_trace(go.Scatter(
                x=buy_signals.index, y=buy_signals['Low'] * 0.995,
                mode='markers', name='Buy Signal',
                marker=dict(symbol='triangle-up', color='lime', size=15,
                          line=dict(color='darkgreen', width=2)),
                text=[f"Strength: {s}<br>Reason: {r}" for s, r in 
                      zip(buy_signals['Signal_Strength'], buy_signals['Signal_Reason'])],
                hovertemplate='<b>BUY SIGNAL</b><br>%{text}<extra></extra>'
            ), row=1, col=1)
        
        if not sell_signals.empty:
            fig.add_trace(go.Scatter(
                x=sell_signals.index, y=sell_signals['High'] * 1.005,
                mode='markers', name='Sell Signal',
                marker=dict(symbol='triangle-down', color='red', size=15,
                          line=dict(color='darkred', width=2)),
                text=[f"Strength: {s}<br>Reason: {r}" for s, r in 
                      zip(sell_signals['Signal_Strength'], sell_signals['Signal_Reason'])],
                hovertemplate='<b>SELL SIGNAL</b><br>%{text}<extra></extra>'
            ), row=1, col=1)
        
        # Add pattern markers
        add_pattern_markers(fig, df, patterns)
        
        # MACD
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', 
                               line=dict(color='blue')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal', 
                               line=dict(color='red')), row=2, col=1)
        colors = ['red' if val < 0 else 'green' for val in df['MACD_Hist']]
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='Histogram',
                           marker_color=colors, opacity=0.7), row=2, col=1)
        
        # RSI and Stochastic
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', 
                               line=dict(color='purple')), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_K'], name='Stoch %K', 
                               line=dict(color='orange')), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_D'], name='Stoch %D', 
                               line=dict(color='cyan')), row=3, col=1)
        
        # RSI levels
        fig.add_hline(y=70, line_dash='dash', line_color='red', row=3, col=1)
        fig.add_hline(y=30, line_dash='dash', line_color='green', row=3, col=1)
        fig.add_hline(y=50, line_dash='dot', line_color='gray', row=3, col=1)
        
        # Volume and ATR
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume',
                           marker_color='lightblue', opacity=0.7), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['ATR'], name='ATR', 
                               line=dict(color='red'), yaxis='y2'), row=4, col=1)
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Advanced Intraday Trading Dashboard",
            template='plotly_dark',
            hovermode='x unified',
            xaxis_rangeslider_visible=False,
            showlegend=False
        )
        
        # Update y-axis titles
        fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
        fig.update_yaxes(title_text="MACD", row=2, col=1)
        fig.update_yaxes(title_text="RSI/Stoch", row=3, col=1, range=[0, 100])
        fig.update_yaxes(title_text="Volume", row=4, col=1)
        
        return fig
    
    except Exception as e:
        st.error(f"Chart creation error: {e}")
        return None

def add_pattern_markers(fig, df, patterns):
    """Add candlestick pattern markers to chart"""
    pattern_colors = {
        'Doji': 'yellow',
        'Hammer': 'lime',
        'Shooting_Star': 'red',
        'Bullish_Engulfing': 'green',
        'Bearish_Engulfing': 'red',
        'Morning_Star': 'lightgreen',
        'Evening_Star': 'pink',
        'Three_White_Soldiers': 'darkgreen',
        'Three_Black_Crows': 'darkred'
    }
    
    for pattern_name, pattern_series in patterns.items():
        if pattern_series.any():
            pattern_points = df[pattern_series]
            if not pattern_points.empty:
                fig.add_trace(go.Scatter(
                    x=pattern_points.index,
                    y=pattern_points['High'] * 1.01,
                    mode='markers',
                    name=pattern_name.replace('_', ' '),
                    marker=dict(
                        symbol='star',
                        color=pattern_colors.get(pattern_name, 'white'),
                        size=8,
                        line=dict(color='black', width=1)
                    ),
                    text=pattern_name.replace('_', ' '),
                    hovertemplate=f'<b>{pattern_name.replace("_", " ")}</b><extra></extra>'
                ), row=1, col=1)

def format_alert_message(ticker, signal_type, price, strength, reason, timestamp):
    """Format comprehensive alert message"""
    ist = pytz.timezone("Asia/Kolkata")
    time_str = timestamp.astimezone(ist).strftime('%d-%m-%Y %H:%M:%S IST')
    
    signal_emoji = "🟢" if signal_type == 1 else "🔴"
    action = "BUY" if signal_type == 1 else "SELL"
    strength_stars = "⭐" * min(int(strength), 5)
    
    message = f"""
<b>{signal_emoji} {action} SIGNAL ALERT!</b>

📊 <b>Ticker:</b> {ticker}
💰 <b>Price:</b> ₹{price:.2f}
⏰ <b>Time:</b> {time_str}
💪 <b>Strength:</b> {strength}/10 {strength_stars}

📈 <b>Analysis:</b>
{reason}

⚡ <b>Action Required:</b> {action} NOW!

🤖 <i>Advanced Trading Bot</i>
    """.strip()
    
    return message

@st.cache_data(ttl=30)
def get_market_data(ticker, interval, period):
    """Cached data fetching"""
    return fetch_stock_data(ticker, interval, period)

def main():
    st.set_page_config(
        page_title="🚀 Advanced Intraday Trading Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .metric-card {
        background-color: #1e1e1e;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #00ff00;
    }
    .alert-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .buy-alert { 
        background-color: #0d4f3c; 
        border-left: 4px solid #00ff00; 
        color: #00ff00;
    }
    .sell-alert { 
        background-color: #4f0d0d; 
        border-left: 4px solid #ff0000; 
        color: #ff0000;
    }
    .stSelectbox > div > div { 
        background-color: #2b2b2b; 
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🚀 Advanced Intraday Trading Dashboard")
    st.markdown("### Real-time Technical Analysis & Signal Generation")
    
    # Sidebar Configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Stock selection
    ticker = st.sidebar.text_input("📈 Stock Ticker", value="RELIANCE.NS", help="Enter NSE ticker (e.g., RELIANCE.NS)")
    
    # Timeframe selection
    interval_options = {
        "1 minute": "1m",
        "5 minutes": "5m", 
        "15 minutes": "15m",
        "30 minutes": "30m",
        "1 hour": "1h"
    }
    
    selected_interval = st.sidebar.selectbox("⏰ Timeframe", list(interval_options.keys()), index=1)
    interval = interval_options[selected_interval]
    
    # Period selection
    period_options = {
        "1 day": "1d",
        "5 days": "5d",
        "1 month": "1mo",
        "3 months": "3mo"
    }
    
    selected_period = st.sidebar.selectbox("📊 Period", list(period_options.keys()), index=1)
    period = period_options[selected_period]
    
    # Alert settings
    st.sidebar.header("🔔 Alert Settings")
    enable_alerts = st.sidebar.checkbox("Enable Telegram Alerts", value=True)
    min_signal_strength = st.sidebar.slider("Minimum Signal Strength", 1, 10, 5)
    
    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=False)
    if auto_refresh:
        time.sleep(0)
        st.rerun()
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    try:
        # Fetch and process data
        with st.spinner(f"📡 Fetching data for {ticker}..."):
            df = get_market_data(ticker, interval, period)
            
            if df is None or df.empty:
                st.error("❌ No data available for the selected ticker")
                return
            
            # Calculate technical indicators
            df = TechnicalAnalysis.calculate_all_indicators(df)
            
            # Identify patterns
            patterns = CandlestickPatterns.identify_patterns(df)
            
            # Generate signals
            df = SignalGenerator.generate_advanced_signals(df, patterns)
        
        # Display current metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        current_price = df['Close'].iloc[-1]
        price_change = df['Close'].iloc[-1] - df['Close'].iloc[-2]
        price_change_pct = (price_change / df['Close'].iloc[-2]) * 100
        current_volume = df['Volume'].iloc[-1]
        avg_volume = df['Volume'].mean()
        
        with col1:
            st.metric("💰 Current Price", f"₹{current_price:.2f}", 
                     f"{price_change:+.2f} ({price_change_pct:+.2f}%)")
        
        with col2:
            rsi_value = df['RSI'].iloc[-1]
            rsi_status = "Overbought" if rsi_value > 70 else "Oversold" if rsi_value < 30 else "Neutral"
            st.metric("📊 RSI", f"{rsi_value:.1f}", rsi_status)
        
        with col3:
            macd_value = df['MACD'].iloc[-1]
            macd_signal = df['MACD_Signal'].iloc[-1]
            macd_status = "Bullish" if macd_value > macd_signal else "Bearish"
            st.metric("📈 MACD", f"{macd_value:.3f}", macd_status)
        
        with col4:
            volume_ratio = current_volume / avg_volume
            volume_status = "High" if volume_ratio > 1.5 else "Low" if volume_ratio < 0.5 else "Normal"
            st.metric("📊 Volume", f"{current_volume:,.0f}", f"{volume_ratio:.1f}x ({volume_status})")
        
        with col5:
            atr_value = df['ATR'].iloc[-1]
            volatility = "High" if atr_value > df['ATR'].quantile(0.8) else "Low" if atr_value < df['ATR'].quantile(0.2) else "Medium"
            st.metric("⚡ ATR", f"{atr_value:.2f}", volatility)
        
        # Recent signals
        recent_signals = df[df['Signal'] != 0].tail(5)
        
        if not recent_signals.empty:
            st.subheader("🎯 Recent Signals")
            
            for idx, signal in recent_signals.iterrows():
                signal_type = "BUY" if signal['Signal'] == 1 else "SELL"
                alert_class = "buy-alert" if signal['Signal'] == 1 else "sell-alert"
                signal_emoji = "🟢" if signal['Signal'] == 1 else "🔴"
                
                ist = pytz.timezone("Asia/Kolkata")
                signal_time = idx.astimezone(ist).strftime('%d-%m-%Y %H:%M:%S IST')
                
                st.markdown(f"""
                <div class="alert-box {alert_class}">
                    <strong>{signal_emoji} {signal_type} SIGNAL</strong><br>
                    📅 Time: {signal_time}<br>
                    💰 Price: ₹{signal['Close']:.2f}<br>
                    💪 Strength: {signal['Signal_Strength']}/10<br>
                    📋 Reason: {signal['Signal_Reason']}
                </div>
                """, unsafe_allow_html=True)
                
                # Send telegram alert for strong signals
                if (enable_alerts and 
                    signal['Signal_Strength'] >= min_signal_strength and
                    (datetime.datetime.now(pytz.timezone("Asia/Kolkata")) - idx.astimezone(pytz.timezone("Asia/Kolkata"))).seconds < 300):
                    
                    alert_message = format_alert_message(
                        ticker, signal['Signal'], signal['Close'], 
                        signal['Signal_Strength'], signal['Signal_Reason'], idx
                    )
                    
                    if send_telegram_alert(alert_message):
                        st.success("✅ Alert sent to Telegram!")
        
        # Main chart
        st.subheader("📊 Advanced Trading Chart")
        chart = create_advanced_chart(df, patterns)
        if chart:
            st.plotly_chart(chart, use_container_width=True)
        
        # Pattern analysis
        st.subheader("🕯️ Candlestick Pattern Analysis")
        
        pattern_cols = st.columns(3)
        pattern_count = 0
        
        for pattern_name, pattern_series in patterns.items():
            if pattern_series.any():
                recent_patterns = df[pattern_series].tail(10)
                if not recent_patterns.empty:
                    col_idx = pattern_count % 3
                    with pattern_cols[col_idx]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>🕯️ {pattern_name.replace('_', ' ')}</h4>
                            <p>Last seen: {recent_patterns.index[-1].strftime('%H:%M')}</p>
                            <p>Occurrences: {pattern_series.sum()}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    pattern_count += 1
        
        if pattern_count == 0:
            st.info("ℹ️ No significant patterns detected in recent data")
        
        # Technical analysis summary
        st.subheader("📈 Technical Analysis Summary")
        
        analysis_cols = st.columns(2)
        
        with analysis_cols[0]:
            st.markdown("#### 🎯 Key Levels")
            
            current_resistance = df['Resistance'].iloc[-1]
            current_support = df['Support'].iloc[-1]
            
            st.write(f"**Resistance:** ₹{current_resistance:.2f}")
            st.write(f"**Support:** ₹{current_support:.2f}")
            st.write(f"**Distance to Resistance:** {((current_resistance - current_price) / current_price * 100):+.2f}%")
            st.write(f"**Distance to Support:** {((current_support - current_price) / current_price * 100):+.2f}%")
        
        with analysis_cols[1]:
            st.markdown("#### 📊 Indicator Status")
            
            # Trend analysis
            ema12 = df['EMA12'].iloc[-1]
            sma20 = df['SMA20'].iloc[-1]
            trend = "Uptrend" if ema12 > sma20 and current_price > ema12 else "Downtrend" if ema12 < sma20 and current_price < ema12 else "Sideways"
            
            st.write(f"**Trend:** {trend}")
            st.write(f"**RSI Status:** {rsi_status} ({rsi_value:.1f})")
            st.write(f"**MACD Status:** {macd_status}")
            st.write(f"**Volatility:** {volatility}")
        
        # Data table
        if st.checkbox("📋 Show Raw Data"):
            st.subheader("📊 Recent Data")
            display_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'Signal', 'Signal_Strength']
            st.dataframe(df[display_columns].tail(20).round(2), use_container_width=True)
        
        # Footer
        st.markdown("---")
        st.markdown("### ⚡ Dashboard Features")
        
        feature_cols = st.columns(4)
        
        with feature_cols[0]:
            st.markdown("""
            **🎯 Signal Generation**
            - Multi-indicator confirmation
            - Pattern recognition
            - Strength scoring
            """)
        
        with feature_cols[1]:
            st.markdown("""
            **📊 Technical Analysis**
            - 15+ indicators
            - Support/Resistance
            - Trend analysis
            """)
        
        with feature_cols[2]:
            st.markdown("""
            **🕯️ Pattern Recognition**
            - 10+ candlestick patterns  
            - Real-time detection
            - Visual markers
            """)
        
        with feature_cols[3]:
            st.markdown("""
            **🔔 Smart Alerts**
            - Telegram integration
            - Customizable strength
            - Real-time notifications
            """)
        
        # Update timestamp
        ist = pytz.timezone("Asia/Kolkata")
        current_time = datetime.datetime.now(ist).strftime('%d-%m-%Y %H:%M:%S IST')
        st.caption(f"🕐 Last updated: {current_time}")
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("💡 Please check your ticker symbol and try again")
        
        # Debug information
        if st.checkbox("🔍 Show Debug Info"):
            st.exception(e)

if __name__ == "__main__":
    main()
