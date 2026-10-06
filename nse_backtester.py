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

# --- Train / Test Split ---
TRAIN_END_DATE  = date(2022, 12, 31)
TEST_START_DATE = TRAIN_END_DATE + timedelta(days=1)

# --- Strategy Parameters ---
SHORT_WINDOW   = 5
LONG_WINDOW    = 13
TREND_MA       = 50    # long-term trend filter: only enter when price is above this MA
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

# --- Optimizer ---
OPTIMIZE_PARAMS = False     # set True to run the grid search instead of a single backtest
OPTIMIZE_METRIC = "sharpe"  # primary ranking metric (tiebreak is total return)

PARAM_GRID = {
    "SHORT_WINDOW":   [7, 9, 11],
    "LONG_WINDOW":    [18, 21, 25],
    "TREND_MA":       [44, 50, 55],
    "RSI_PERIOD":     [12, 14],
    "ATR_MULTIPLIER": [3.5, 4.0, 4.5],
}

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

def add_indicators(df, short_w, long_w, rsi_p, atr_p, adx_p, trend_ma_p):
    df = df.copy()
    df["ma_short"] = df["close"].rolling(short_w).mean()
    df["ma_long"]  = df["close"].rolling(long_w).mean()
    df["trend_ma"] = df["close"].rolling(trend_ma_p).mean()
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

def run_backtest(df, params, start_date, end_date):
    trade_df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].reset_index(drop=True)
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
        elif row["cross_up"] and price > row["ma_long"] and price > row["trend_ma"] and row["rsi"] <= params["RSI_BUY_MAX"] and row["adx"] >= params["ADX_MIN"]:
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
#  METRICS
# ==============================================================================

def compute_sharpe(portfolio_df, periods_per_year=252):
    if portfolio_df.empty or len(portfolio_df) < 2:
        return 0.0
    returns = portfolio_df["portfolio_value"].pct_change().dropna()
    if returns.empty or returns.std() == 0:
        return 0.0
    return float((returns.mean() / returns.std()) * np.sqrt(periods_per_year))

def compute_cagr(portfolio_df, periods_per_year=252):
    if portfolio_df.empty or len(portfolio_df) < 2:
        return 0.0
    years = (len(portfolio_df) - 1) / periods_per_year
    if years <= 0:
        return 0.0
    start_val = portfolio_df["portfolio_value"].iloc[0]
    end_val = portfolio_df["portfolio_value"].iloc[-1]
    if start_val <= 0:
        return 0.0
    return float(((end_val / start_val) ** (1 / years) - 1) * 100)

def compute_total_return(portfolio_df):
    if portfolio_df.empty or len(portfolio_df) < 2:
        return 0.0
    start_val = portfolio_df["portfolio_value"].iloc[0]
    end_val = portfolio_df["portfolio_value"].iloc[-1]
    return float((end_val / start_val - 1) * 100)

# ==============================================================================
#  GRID SEARCH OPTIMIZER
# ==============================================================================

def run_grid_search(data, param_grid, start_date, end_date, metric=OPTIMIZE_METRIC):
    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))
    total = len(combos)
    print(f"Grid search: {total} parameter combinations to test ({start_date} to {end_date})")

    results = []
    for i, combo in enumerate(combos, 1):
        combo_params = dict(zip(keys, combo))
        params = {
            "SHORT_WINDOW":   combo_params["SHORT_WINDOW"],
            "LONG_WINDOW":    combo_params["LONG_WINDOW"],
            "TREND_MA":       combo_params["TREND_MA"],
            "RSI_PERIOD":     combo_params["RSI_PERIOD"],
            "ATR_MULTIPLIER": combo_params["ATR_MULTIPLIER"],
            "RSI_BUY_MAX":    RSI_BUY_MAX,
            "ADX_MIN":        ADX_MIN,
        }
        try:
            df_ind = add_indicators(
                data, params["SHORT_WINDOW"], params["LONG_WINDOW"],
                params["RSI_PERIOD"], ATR_PERIOD, ADX_PERIOD, params["TREND_MA"]
            )
            trades, portfolio = run_backtest(df_ind, params, start_date, end_date)
        except Exception:
            continue

        if portfolio.empty or len(portfolio) < 2:
            continue

        sharpe = compute_sharpe(portfolio)
        total_return = compute_total_return(portfolio)
        n_trades = int((trades["action"] == "SELL").sum()) if not trades.empty else 0

        results.append({
            **combo_params,
            "sharpe": round(sharpe, 3),
            "total_return_pct": round(total_return, 2),
            "n_trades": n_trades,
        })

        if i % 20 == 0 or i == total:
            print(f"  [{i}/{total}] combos tested...")

    results_df = pd.DataFrame(results)
    if results_df.empty:
        print("Grid search produced no valid results (no trades triggered in any config).")
        return results_df

    results_df = results_df.sort_values(
        ["sharpe", "total_return_pct"], ascending=[False, False]
    ).reset_index(drop=True)
    top10 = results_df.head(10).copy()
    top10.insert(0, "rank", range(1, len(top10) + 1))

    print("\n" + "=" * 100)
    print(f"TOP {len(top10)} CONFIGS — ranked by Sharpe ratio (tiebreak: total return)  [{total} combos tested]")
    print("=" * 100)
    print(top10.to_string(index=False))
    print("=" * 100)

    return top10

# ==============================================================================
#  MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    data = fetch_data(SYMBOL, START_DATE, END_DATE)

    if OPTIMIZE_PARAMS:
        top10 = run_grid_search(data, PARAM_GRID, START_DATE, TRAIN_END_DATE)
        if top10.empty:
            sys.exit("No valid grid search configs found; aborting.")
        best = top10.iloc[0]
        best_params = {
            "SHORT_WINDOW":   int(best["SHORT_WINDOW"]),
            "LONG_WINDOW":    int(best["LONG_WINDOW"]),
            "TREND_MA":       int(best["TREND_MA"]),
            "RSI_PERIOD":     int(best["RSI_PERIOD"]),
            "ATR_MULTIPLIER": float(best["ATR_MULTIPLIER"]),
            "RSI_BUY_MAX":    RSI_BUY_MAX,
            "ADX_MIN":        ADX_MIN,
        }
        print(f"\nBest config selected from grid search: {best_params}")
    else:
        best_params = {
            "SHORT_WINDOW":   SHORT_WINDOW,
            "LONG_WINDOW":    LONG_WINDOW,
            "TREND_MA":       TREND_MA,
            "RSI_PERIOD":     RSI_PERIOD,
            "ATR_MULTIPLIER": ATR_MULTIPLIER,
            "RSI_BUY_MAX":    RSI_BUY_MAX,
            "ADX_MIN":        ADX_MIN,
        }

    # --- Train / Test evaluation (prevents in-sample overfitting) ---
    df_ind = add_indicators(
        data, best_params["SHORT_WINDOW"], best_params["LONG_WINDOW"],
        best_params["RSI_PERIOD"], ATR_PERIOD, ADX_PERIOD, best_params["TREND_MA"]
    )

    train_trades, train_portfolio = run_backtest(df_ind, best_params, START_DATE, TRAIN_END_DATE)
    test_trades, test_portfolio   = run_backtest(df_ind, best_params, TEST_START_DATE, END_DATE)

    in_sample_sharpe  = compute_sharpe(train_portfolio)
    out_sample_sharpe = compute_sharpe(test_portfolio)
    in_sample_return  = compute_total_return(train_portfolio)
    out_sample_return = compute_total_return(test_portfolio)
    in_sample_cagr    = compute_cagr(train_portfolio)
    out_sample_cagr   = compute_cagr(test_portfolio)

    train_years = (len(train_portfolio) - 1) / 252 if len(train_portfolio) > 1 else 0.0
    test_years  = (len(test_portfolio) - 1) / 252 if len(test_portfolio) > 1 else 0.0
    train_range = f"{START_DATE} to {TRAIN_END_DATE} (~{train_years:.2f} yrs)"
    test_range  = f"{TEST_START_DATE} to {END_DATE} (~{test_years:.2f} yrs)"

    print("\n" + "=" * 70)
    print(f"In-Sample (Train):     {train_range}")
    print(f"Out-of-Sample (Test):  {test_range}")
    print("-" * 70)
    print(f"{'Metric':<25}{'In-Sample':>20}{'Out-of-Sample':>20}")
    print("-" * 70)
    print(f"{'Sharpe Ratio':<25}{in_sample_sharpe:>20.3f}{out_sample_sharpe:>20.3f}")
    print(f"{'Total Return %':<25}{in_sample_return:>20.2f}{out_sample_return:>20.2f}")
    print(f"{'CAGR % (annualized)':<25}{in_sample_cagr:>20.2f}{out_sample_cagr:>20.2f}")
    print("=" * 70)
    print("Note: Total Return is cumulative over each window (different lengths); "
          "compare CAGR for an apples-to-apples annualized read.")

    if in_sample_sharpe != 0:
        degradation = (in_sample_sharpe - out_sample_sharpe) / abs(in_sample_sharpe)
        if degradation > 0.40:
            print(f"\nWARNING: out-of-sample Sharpe dropped {degradation * 100:.1f}% vs in-sample — "
                  f"possible overfitting to the training window.")

    print(f"\nPrepared by Surekha Yadav")

    full_portfolio = pd.concat([train_portfolio, test_portfolio]).reset_index(drop=True)
    print(f"\nBacktest Complete for {SYMBOL}. Final Value: Rs.{full_portfolio['portfolio_value'].iloc[-1]:,.2f}")

    full_portfolio["date"] = pd.to_datetime(full_portfolio["date"])
    full_portfolio.plot(x="date", y="portfolio_value", title=f"Equity Curve - {SYMBOL}")
    plt.axvline(x=pd.Timestamp(TRAIN_END_DATE), color="red", linestyle="--", label="Train/Test Split")
    plt.legend()
    plt.show()
