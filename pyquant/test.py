import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta  # Pure-Python technical analysis library
from datetime import datetime, timedelta

def analyze_stock(ticker="AAPL"):
    """
    Analyzes a stock's recent price action to identify potential short-term trading opportunities.

    Args:
        ticker (str): The stock ticker symbol (default: "AAPL").
    """

    # 1. Load or fetch the last 14 days of OHLCV data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=20)  # Fetch a bit more for MA calculation
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            print(f"No data found for ticker {ticker}.")
            return
        df = df.tail(14) # Keep only the last 14 days
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return

    # 2. Compute 5-day and 10-day moving averages
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()

    # 3. Identify candlestick patterns and label bullish/bearish signals
    # (Simplified pattern recognition - can be expanded)
    df['Hammer'] = ta.cdl_hammer(df['Open'], df['High'], df['Low'], df['Close'])
    df['Engulfing'] = ta.cdl_engulfing(df['Open'], df['High'], df['Low'], df['Close'])
    df['Doji'] = ta.cdl_doji(df['Open'], df['High'], df['Low'], df['Close'])

    bullish_signals = df['Hammer'].fillna(0) + (df['Engulfing'] > 0).fillna(0)
    bearish_signals = (df['Engulfing'] < 0).fillna(0) + df['Doji'].fillna(0)

    # 4. Evaluate moving average crossovers and momentum
    df['MACD'] = ta.macd(df['Close'])['MACD_12_26_9']
    df['Signal'] = ta.macd(df['Close'])['MACDs_12_26_9']
    df['ROC'] = ta.momentum.roc(df['Close'], length=10)

    # 5. Assess volume trends alongside price moves
    df['Volume_MA'] = df['Volume'].rolling(window=5).mean()

    # 6. Determine support and resistance from recent highs/lows
    # (Simplified - using recent max/min)
    recent_high = df['High'].max()
    recent_low = df['Low'].min()

    # 7. Determine potential 3-5% gain setup
    last_close = df['Close'].iloc[-1]
    potential_gain = 0.03 * last_close  # 3% gain target
    target_price = last_close + potential_gain

    # Check for bullish signals, MA crossover, and volume confirmation
    bullish_condition = (bullish_signals.iloc[-1] > 0) and (df['MA5'].iloc[-1] > df['MA10'].iloc[-1]) and (df['Volume'].iloc[-1] > df['Volume_MA'].iloc[-1])

    if bullish_condition:
        # Calculate optimal buy price, profit target, and stop-loss
        buy_price = last_close  # Buy at the current price
        stop_loss = last_close - 0.5 * potential_gain # Risk/reward 2:1
        if stop_loss < recent_low:
            stop_loss = recent_low # Ensure stop loss is not below recent low

        risk = buy_price - stop_loss
        reward = target_price - buy_price

        if risk > 0 and reward / risk >= 2:
            print(f"Optimal Buy Price: {buy_price:.2f}")
            print(f"Profit Target: {target_price:.2f}")
            print(f"Stop-Loss: {stop_loss:.2f}")
        else:
            print(f"No short-term trade recommended for {ticker} due to insufficient risk/reward ratio.")

    else:
        print(f"No short-term trade recommended for {ticker}.  Reasons: Insufficient bullish signals, no MA crossover, or weak volume.")

if __name__ == "__main__":
    analyze_stock("AAPL")  # Example: Analyze Apple stock
    # analyze_stock("MSFT") # Example: Analyze Microsoft stock