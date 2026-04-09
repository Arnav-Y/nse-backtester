"""
================================================
  NSE Advanced Backtester v3
  MA Crossover + RSI + ADX + Risk Sizing
  With Parameter Optimizer & Trading Costs
================================================
"""

import sys
import itertools
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, timedelta
import yfinance as yf
warnings.filterwarnings("ignore")

# ==============================================================================
#  CONFIGURATION
# ==============================================================================

SYMBOL     = "RELIANCE"          
START_DATE = date(2019, 1, 1)
END_DATE   = date(2024, 1, 1)

# --- Strategy Parameters ---
SHORT_WINDOW   = 5
LONG_WINDOW    = 13
RSI_PERIOD     = 14
RSI_BUY_MAX    = 70
RSI_SELL_MIN   = 35
ATR_PERIOD     = 14
ATR_MULTIPLIER = 3.0
ADX_PERIOD     = 14
ADX_MIN        = 15    
RR_RATIO       = 3.0   

# --- Risk Management ---
STARTING_CASH     = 100_000   
RISK_PER_TRADE    = 0.02      
MIN_HOLD_DAYS     = 3         

# --- Realistic Trading Costs (NSE) ---
BROKERAGE_PCT     = 0.0003    
STT_PCT           = 0.001     
EXCHANGE_PCT      = 0.0000345 
SLIPPAGE_PCT      = 0.0005    

TOTAL_BUY_COST    = BROKERAGE_PCT + EXCHANGE_PCT + SLIPPAGE_PCT
TOTAL_SELL_COST   = BROKERAGE_PCT + STT_PCT + EXCHANGE_PCT + SLIPPAGE_PCT

# --- Optimizer Search Space ---
OPTIMIZE_PARAMS = {
    "SHORT_WINDOW":   [5, 9, 12],
    "LONG_WINDOW":    [21, 26, 50],
    "RSI_BUY_MAX":    [60, 65, 70],
    "ATR_MULTIPLIER": [2.0, 2.5, 3.0],
    "ADX_MIN":        [15, 20, 25],
}
OPTIMIZE_METRIC = "sharpe"   

# ==============================================================================
#  DATA FETCHING
# ==============================================================================

def fetch_data(symbol, start, end, warmup_days=250):
    yf_symbol   = f"{symbol}.NS"
    warmup_start = start - timedelta(days=warmup_days)
    print(f"Fetching {yf_symbol}...")
    df = yf.download(yf_symbol, start=str(warmup_start), end=str(end), progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data for {yf_symbol}.")
    df = df.reset_index()
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[["date", "open", "high", "low", "close", "volume"]]
    return df.sort_values("date").reset_index(drop=True)

# ==============================================================================
#  INDICATORS
# ==============================================================================

def add_indicators(df, short_w, long_w, rsi_p, atr_p, adx_p):
    df = df.copy()
    df["ma_short"] = df["close"].rolling(short_w).mean()
    df["ma_long"]  = df["close"].rolling(long_w).mean()
    df["cross_up"]   = (df["ma_short"].shift(1) <= df["ma_long"].shift(1)) & (df["ma_short"] > df["ma_long"])
    df["cross_down"] = (df["ma_short"].shift(1) >= df["ma_long"].shift(1)) & (df["ma_short"] < df["ma_long"])
    
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=rsi_p, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_p, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    tr = pd.concat([df["high"]-df["low"], (df["high"]-df["close"].shift(1)).abs(), (df["low"]-df["close"].shift(1)).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(atr_p).mean()

    # ADX Calculation
    upmove = df["high"] - df["high"].shift(1)
    downmove = df["low"].shift(1) - df["low"]
    plus_dm = np.where((upmove > downmove) & (upmove > 0), upmove, 0)
    minus_dm = np.where((downmove > upmove) & (downmove > 0), downmove, 0)
    plus_di = 100 * (pd.Series(plus_dm).ewm(span=adx_p).mean() / df["atr"])
    minus_di = 100 * (pd.Series(minus_dm).ewm(span=adx_p).mean() / df["atr"])
    df["adx"] = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di)).ewm(span=adx_p).mean()
    
    return df.dropna().reset_index(drop=True)

# ==============================================================================
#  POSITION SIZING & ENGINE
# ==============================================================================

def calc_position_size(cash, entry_price, stop_price, risk_pct):
    risk_amount = cash * risk_pct
    risk_per_share = abs(entry_price - stop_price)
    if risk_per_share <= 0: return 0
    shares = int(risk_amount / risk_per_share)
    return min(shares, int((cash * 0.95) / entry_price))

def run_backtest(df, params):
    trade_df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)].reset_index(drop=True)
    cash, shares, trades, portfolio = STARTING_CASH, 0, [], []
    entry_p, stop_p, tp_p, highest_p, hold_days = None, None, None, None, 0

    for _, row in trade_df.iterrows():
        price = row["close"]
        if shares > 0:
            hold_days += 1
            highest_p = max(highest_p, price)
            stop_p = max(stop_p, highest_p - params["ATR_MULTIPLIER"] * row["atr"])

            # Exits
            if (price >= tp_p or price <= stop_p or (row["cross_down"] and row["rsi"] >= RSI_SELL_MIN)) and hold_days >= MIN_HOLD_DAYS:
                sell_price = price * (1 - TOTAL_SELL_COST)
                cash += shares * sell_price
                trades.append({"date": row["date"], "pnl": (sell_price - entry_p) * shares, "action": "SELL"})
                shares = 0
        
        # Entry
        elif row["cross_up"] and price > row["ma_long"] and row["rsi"] <= params["RSI_BUY_MAX"] and row["adx"] >= params["ADX_MIN"]:
            stop_p = price - params["ATR_MULTIPLIER"] * row["atr"]
            tp_p = price + RR_RATIO * (price - stop_p)
            buy_price = price * (1 + TOTAL_BUY_COST)
            shares = calc_position_size(cash, buy_price, stop_p, RISK_PER_TRADE)
            if shares > 0:
                entry_p, highest_p, hold_days = buy_price, price, 0
                cash -= shares * buy_price
                trades.append({"date": row["date"], "action": "BUY"})

        portfolio.append({"date": row["date"], "portfolio_value": cash + (shares * price)})

    return pd.DataFrame(trades), pd.DataFrame(portfolio)

# ==============================================================================
#  MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    data = fetch_data(SYMBOL, START_DATE, END_DATE)
    final_params = {
        "SHORT_WINDOW": SHORT_WINDOW, "LONG_WINDOW": LONG_WINDOW, 
        "RSI_BUY_MAX": RSI_BUY_MAX, "ATR_MULTIPLIER": ATR_MULTIPLIER, "ADX_MIN": ADX_MIN
    }
    
    # Run Backtest
    df_ind = add_indicators(data, final_params["SHORT_WINDOW"], final_params["LONG_WINDOW"], RSI_PERIOD, ATR_PERIOD, ADX_PERIOD)
    trades, portfolio = run_backtest(df_ind, final_params)
    
    print(f"Backtest Complete for {SYMBOL}. Final Value: Rs.{portfolio['portfolio_value'].iloc[-1]:,.2f}")
    portfolio.plot(x="date", y="portfolio_value", title=f"Equity Curve - {SYMBOL}")
    plt.show()