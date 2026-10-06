# NSE Advanced Backtester v3

A multi-indicator stock backtesting engine for Indian equities (NSE), combining **MA Crossover**, **RSI**, **ADX**, and **ATR-based risk sizing** with realistic trading cost simulation.

---

## Features

- **Multi-indicator signal generation** — MA crossover filtered by RSI and ADX to reduce false entries
- **Dynamic position sizing** — Risk a fixed % of capital per trade based on ATR stop distance
- **Trailing stop-loss** — ATR-based trailing stop that locks in profits as price moves up
- **Realistic NSE cost model** — Brokerage, STT, exchange charges, and slippage all factored in
- **Parameter optimizer** — Grid search over configurable parameter space, ranked by Sharpe ratio
- **Equity curve output** — Portfolio value plotted over the backtest period

---

## Strategy Logic

| Signal | Condition |
|--------|-----------|
| **Entry** | Short MA crosses above Long MA, RSI ≤ threshold, ADX ≥ minimum, price > Long MA |
| **Exit (Stop)** | Price hits ATR-based trailing stop |
| **Exit (Target)** | Price hits risk/reward target (default 3:1 RR) |
| **Exit (Signal)** | Short MA crosses below Long MA and RSI ≥ sell threshold |

A minimum hold period prevents premature exits on noisy crossovers.

---

## Tech Stack

- **Python 3.8+**
- `yfinance` — Historical OHLCV data from Yahoo Finance
- `pandas` / `numpy` — Data manipulation and indicator calculation
- `matplotlib` — Equity curve visualization

---

## Installation

```bash
git clone https://github.com/yourusername/nse-backtester.git
cd nse-backtester
pip install yfinance pandas numpy matplotlib
```

---

## Usage

Edit the configuration block at the top of `nse_backtester.py`:

```python
SYMBOL     = "RELIANCE"          # NSE ticker (without .NS suffix)
START_DATE = date(2019, 1, 1)
END_DATE   = date(2024, 1, 1)

SHORT_WINDOW   = 5
LONG_WINDOW    = 13
RSI_PERIOD     = 14
RSI_BUY_MAX    = 70
ADX_MIN        = 15
ATR_MULTIPLIER = 3.0
```

Then run:

```bash
python nse_backtester.py
```

The script will print the final portfolio value and display an equity curve chart.

---

## Configuration Reference

### Strategy Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SHORT_WINDOW` | 5 | Fast moving average period |
| `LONG_WINDOW` | 13 | Slow moving average period |
| `RSI_PERIOD` | 14 | RSI lookback period |
| `RSI_BUY_MAX` | 70 | Maximum RSI allowed on entry (avoids overbought entries) |
| `RSI_SELL_MIN` | 35 | Minimum RSI required for signal-based exit |
| `ATR_PERIOD` | 14 | ATR lookback for stop and sizing |
| `ATR_MULTIPLIER` | 3.0 | Stop distance as multiple of ATR |
| `ADX_PERIOD` | 14 | ADX lookback period |
| `ADX_MIN` | 15 | Minimum ADX to confirm trend strength |
| `RR_RATIO` | 3.0 | Risk/reward ratio for take-profit target |
| `MIN_HOLD_DAYS` | 3 | Minimum days to hold before allowing exit |

### Risk Management

| Parameter | Default | Description |
|-----------|---------|-------------|
| `STARTING_CASH` | ₹1,00,000 | Initial capital |
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

```
Fetching RELIANCE.NS...
Backtest Complete for RELIANCE. Final Value: Rs.1,84,320.45
```

An equity curve window will open showing portfolio value over time.

---

## Disclaimer

This project is for **educational and research purposes only**. Past performance of a backtested strategy does not guarantee future results. This is not financial advice. Always do your own research before trading.
