# Enhanced EMA Portfolio Strategy

## Overview
This project implements an advanced **trend-following trading strategy** using **Exponential Moving Average (EMA) crossovers** combined with **portfolio optimization techniques**. It integrates individual asset strategies into a Sharpe-weighted portfolio while providing visualizations and key performance metrics.

The strategy is designed to minimize noise, adapt dynamically to market conditions, and combine quantitative trading principles with practical risk management.

---

## Key Features

1. **Adaptive EMA Crossover Strategy per Ticker**
   - Calculates short-term and long-term EMAs.
   - Trades signals are generated when EMA_short crosses EMA_long with a significance threshold to reduce whipsaw trades.
   - Applies a **volatility filter** to avoid trading in high-volatility periods.
   - Accounts for **transaction costs** when positions change.

2. **Parameter Grid Search Optimization**
   - Automatically searches for the best EMA short and long spans per ticker.
   - Optimizes strategy performance using total return as the objective metric.

3. **Sharpe-Weighted Multi-Asset Portfolio**
   - Combines individual ticker strategies into a single portfolio.
   - Weights are proportional to the Sharpe ratio of each strategy.
   - Ensures assets with poor performance contribute minimally to overall portfolio.

4. **Mean-Variance Portfolio Optimization**
   - Computes the **expected annual return** and **covariance matrix** for the selected tickers.
   - Finds the **Maximum Sharpe Ratio portfolio** using constrained optimization.
   - Generates the **Efficient Frontier** for visualization and analysis.

5. **Visualization**
   - Plots EMA crossovers and price for each ticker.
   - Compares individual EMA strategy portfolios, combined EMA portfolio, and buy & hold portfolio.
   - Efficient Frontier visualization with maximum Sharpe portfolio highlighted.

6. **Performance Metrics**
   - Calculates **Total Return**, **Sharpe Ratio**, and **Maximum Drawdown** for the combined portfolio.
   - Displays per-ticker best EMA parameters and strategy returns.

---

## Project Structure

```
enhanced-ema-portfolio/
│
├─ enhanced_ema_portfolio.py      # Main Python script with strategy and portfolio optimization
├─ README.md                       # High-level design and documentation
├─ .gitignore                      # Ignore cache, pyc, and temporary files
```

---

## Technical Details

### EMA Crossover Logic
- **Signal Generation**:
  - Long position: `EMA_short > EMA_long * (1 + SIGNIFICANCE_PCT)`
  - Short/flat position: `EMA_short < EMA_long * (1 - SIGNIFICANCE_PCT)`
  - No trade if realized volatility > threshold.
- **Position Lag**: Positions applied one day after signal to avoid look-ahead bias.
- **Transaction Costs**: Modeled as a fixed proportion per trade when position changes.

### Grid Search
- Explores combinations of short and long EMA spans from defined grids.
- Chooses the combination with maximum total return for each ticker.
- Ensures short < long constraint for meaningful crossovers.

### Portfolio Construction
- **Sharpe-Weighted Combined Portfolio**:
  - Calculates Sharpe ratio for each asset strategy.
  - Converts Sharpe ratios into weights for combined returns.
- **Buy & Hold Benchmark**: Equal-weighted portfolio across tickers for comparison.

### Mean-Variance Optimization
- Uses annualized mean returns and covariance matrix.
- Optimizes weights to **maximize Sharpe ratio** under full investment and non-negativity constraints.
- Generates Efficient Frontier for risk-return trade-off visualization.

### Visualization
- EMA crossovers plotted per ticker with portfolio growth.
- Combined EMA portfolio vs Buy & Hold vs individual EMA portfolios.
- Efficient Frontier showing Maximum Sharpe Portfolio.

---

## Dependencies
- Python 3.8+
- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- `yfinance`

Install dependencies using:
```bash
pip install numpy pandas matplotlib scipy yfinance
```

---

## Usage
```bash
python enhanced_ema_portfolio.py
```
- The script downloads historical price data from Yahoo Finance.
- Performs per-ticker EMA strategy optimization.
- Constructs combined portfolio and calculates performance metrics.
- Generates plots for analysis.

---

## Design Philosophy
- **Adaptive & Data-Driven**: Uses grid search to adapt EMA spans to each asset.
- **Noise Reduction**: Significance threshold and volatility filter reduce false signals.
- **Risk-Aware**: Transaction costs, drawdown tracking, and Sharpe-weighted portfolio ensure realistic and risk-conscious performance.
- **Modular & Extensible**: Easy to extend with additional strategies, risk models, or portfolio constraints.

---

This repository demonstrates a professional and quantitative approach to **algorithmic trading and portfolio optimization** suitable for research, backtesting, and educational purposes.

