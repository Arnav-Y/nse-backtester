# NSE Advanced Backtester v3

A multi-indicator stock backtesting engine for Indian equities (NSE), combining **MA Crossover**, **RSI**, **ADX**, and **ATR-based risk sizing** with realistic trading cost simulation.

---

## Features

- **Multi-indicator signal generation** — MA crossover filtered by RSI, ADX, and a long-term trend MA to reduce false entries
- **Dynamic position sizing** — Risk a fixed % of capital per trade based on ATR stop distance
- **Trailing stop-loss** — ATR-based trailing stop that locks in profits as price moves up
- **Realistic NSE cost model** — Brokerage, STT, exchange charges, and slippage all factored in
- **Grid search optimizer** — Exhaustive search over a configurable parameter grid, ranked by Sharpe ratio (tiebreak: total return)
- **Train/test split** — Evaluates the strategy on an out-of-sample window separate from the one used to pick parameters, with an automatic overfitting warning
- **Equity curve output** — Portfolio value plotted over the backtest period, with the train/test split marked

---

## Strategy Logic

| Signal | Condition |
|--------|-----------|
| **Entry** | Short MA crosses above Long MA, price > Long MA, price > Trend MA, RSI ≤ threshold, ADX ≥ minimum |
| **Exit (Stop)** | Price hits ATR-based trailing stop |
| **Exit (Target)** | Price hits risk/reward target (default 3:1 RR) |
| **Exit (Signal)** | Short MA crosses below Long MA and RSI ≥ sell threshold |

A minimum hold period prevents premature exits on noisy crossovers. The Trend MA (`TREND_MA`) is a slower-moving filter on top of the crossover — it blocks entries unless price is above the longer-term trend.

---

## Tech Stack

- **Python 3.8+**
- `yfinance` — Historical OHLCV data from Yahoo Finance
- `pandas` / `numpy` — Data manipulation and indicator calculation
- `matplotlib` — Equity curve visualization

---

## Installation

```bash
git clone https://github.com/Arnav-Y/nse-backtester.git
cd nse-backtester
python3 -m venv .venv
source .venv/bin/activate
pip install yfinance pandas numpy matplotlib
```

---

## Usage

Edit the configuration block at the top of `nse_backtester.py`:

```python
SYMBOL     = "RELIANCE"          # NSE ticker (without .NS suffix)
START_DATE = date(2019, 1, 1)
END_DATE   = date(2024, 1, 1)

TRAIN_END_DATE  = date(2022, 12, 31)   # TEST_START_DATE is auto-set to the day after

SHORT_WINDOW   = 5
LONG_WINDOW    = 13
TREND_MA       = 50
RSI_PERIOD     = 14
RSI_BUY_MAX    = 70
ADX_MIN        = 15
ATR_MULTIPLIER = 3.0
```

Then run a single backtest:

```bash
python nse_backtester.py
```

This evaluates the configured parameters on the train window (`START_DATE` → `TRAIN_END_DATE`) and the test window (`TEST_START_DATE` → `END_DATE`) separately, prints a side-by-side comparison, and plots the equity curve with the split marked.

### Running the grid search optimizer

Set `OPTIMIZE_PARAMS = True` and run the script again:

```python
OPTIMIZE_PARAMS = True
```

This runs every combination in `PARAM_GRID` (via `itertools.product`) over the train window only, ranks all results by Sharpe ratio (tiebreak: total return), prints the top 10 configs, and then re-evaluates the best config on the held-out test window — so the reported out-of-sample numbers were never used to pick the parameters.

```python
PARAM_GRID = {
    "SHORT_WINDOW":   [7, 9, 11],
    "LONG_WINDOW":    [18, 21, 25],
    "TREND_MA":       [44, 50, 55],
    "RSI_PERIOD":     [12, 14],
    "ATR_MULTIPLIER": [3.5, 4.0, 4.5],
}
```

---

## Configuration Reference

### Strategy Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SHORT_WINDOW` | 5 | Fast moving average period |
| `LONG_WINDOW` | 13 | Slow moving average period |
| `TREND_MA` | 50 | Long-term trend filter — entries require price above this MA |
| `RSI_PERIOD` | 14 | RSI lookback period |
| `RSI_BUY_MAX` | 70 | Maximum RSI allowed on entry (avoids overbought entries) |
| `RSI_SELL_MIN` | 35 | Minimum RSI required for signal-based exit |
| `ATR_PERIOD` | 14 | ATR lookback for stop and sizing |
| `ATR_MULTIPLIER` | 3.0 | Stop distance as multiple of ATR |
| `ADX_PERIOD` | 14 | ADX lookback period |
| `ADX_MIN` | 15 | Minimum ADX to confirm trend strength |
| `RR_RATIO` | 3.0 | Risk/reward ratio for take-profit target |
| `MIN_HOLD_DAYS` | 3 | Minimum days to hold before allowing exit |

### Train / Test Split

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TRAIN_END_DATE` | 2022-12-31 | Last date included in the in-sample (train) window |
| `TEST_START_DATE` | auto (day after `TRAIN_END_DATE`) | First date of the out-of-sample (test) window |

Each window is evaluated independently from the same `STARTING_CASH` (not compounded), so Sharpe/return/CAGR are directly comparable between train and test. If out-of-sample Sharpe drops more than 40% versus in-sample, the script prints an overfitting warning.

### Optimizer

| Parameter | Default | Description |
|-----------|---------|-------------|
| `OPTIMIZE_PARAMS` | `False` | Set `True` to run the grid search instead of a single backtest |
| `OPTIMIZE_METRIC` | `"sharpe"` | Primary ranking metric for the grid search (tiebreak: total return) |
| `PARAM_GRID` | see above | Parameter grid searched via `itertools.product` |

### Risk Management

| Parameter | Default | Description |
|-----------|---------|-------------|
| `STARTING_CASH` | ₹1,00,000 | Initial capital (used independently for both train and test windows) |
| `RISK_PER_TRADE` | 2% | Max capital risked per trade |

### NSE Cost Model

| Cost | Rate |
|------|------|
| Brokerage | 0.03% per side |
| STT | 0.1% on sell side |
| Exchange charges | 0.00345% per side |
| Slippage | 0.05% per side |

---

## Sample Output

### Single backtest (`OPTIMIZE_PARAMS = False`)

```
Fetching RELIANCE.NS...

======================================================================
In-Sample (Train):     2019-01-01 to 2022-12-31 (~3.92 yrs)
Out-of-Sample (Test):  2023-01-01 to 2024-01-01 (~0.97 yrs)
----------------------------------------------------------------------
Metric                              In-Sample       Out-of-Sample
----------------------------------------------------------------------
Initial Value (Rs.)                100,000.00          100,000.00
Final Value (Rs.)                  118,341.84          102,187.09
Sharpe Ratio                            0.990               0.630
Total Return %                          18.34                2.19
CAGR % (annualized)                      4.38                2.26
======================================================================
Note: each window starts fresh from the same Initial Value (independent evaluation, not compounded).
Note: Total Return is cumulative over each window (different lengths); compare CAGR for an apples-to-apples annualized read.

Backtest Complete for RELIANCE. Initial Value: Rs.100,000.00 -> Final Value: Rs.102,187.09
```

An equity curve window will open showing portfolio value over time, with a dashed line marking the train/test split.

### Grid search (`OPTIMIZE_PARAMS = True`)

```
Grid search: 162 parameter combinations to test (2019-01-01 to 2022-12-31)
  [20/162] combos tested...
  ...
====================================================================================================
TOP 10 CONFIGS — ranked by Sharpe ratio (tiebreak: total return)  [162 combos tested]
====================================================================================================
 rank  SHORT_WINDOW  LONG_WINDOW  TREND_MA  RSI_PERIOD  ATR_MULTIPLIER  sharpe  total_return_pct  n_trades
    1             7           25        55          14             3.5   0.994             10.67         5
    ...
```

The best config from the grid search is then re-evaluated on the held-out test window, using the same side-by-side report shown above.

---

## Disclaimer

This project is for **educational and research purposes only**. Past performance of a backtested strategy does not guarantee future results. This is not financial advice. Always do your own research before trading.
