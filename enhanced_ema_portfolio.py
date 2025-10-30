"""
Enhanced SMA/EMA Strategy + Portfolio Optimization
- Longer lookback (start 2017)
- EMA default (50,200) and optional SMA (20,50) options
- Volatility filter and significance threshold to prevent whipsaws
- Parameter grid-search (for EMA spans) per ticker to find best-performing combo
- Sharpe-weighted portfolio combination of single-asset strategies
- Mean-variance (max Sharpe) portfolio optimization and efficient frontier
- Plots and performance metrics

Usage: run the script. It fetches data via yfinance and shows plots. Adjust PARAMETERS section below.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from itertools import product
import warnings

warnings.filterwarnings("ignore")

# ---------------- PARAMETERS ----------------
TICKERS = ["JNJ", "MSFT", "PG", "JPM", "NEE"]
BENCH = "^GSPC"
START = "2008-01-01"        # extended lookback
END = "2025-10-29"
TRADING_DAYS = 252
RISK_FREE = 0.02
INITIAL_CAPITAL = 100000

# Strategy hyper-parameters (defaults)
EMA_SHORT_DEFAULT = 50
EMA_LONG_DEFAULT = 200
SMA_SHORT_DEFAULT = 20
SMA_LONG_DEFAULT = 50

# Trade filters
SIGNIFICANCE_PCT = 0.01   # require short > long * (1+SIGNIFICANCE_PCT) to go long
VOL_WINDOW = 20           # window to compute realized volatility
VOL_THRESHOLD = 0.05      # if realized vol > this, skip trading for that asset
TRANSACTION_COST = 0.0005 # proportional cost per trade (0.05%) - applied as slippage

# Grid-search settings to optimize EMA spans per ticker (keeps runtime moderate)
EMA_SHORT_GRID = [10, 20, 50]
EMA_LONG_GRID = [100, 150, 200]

# ---------------- HELPERS ----------------

def download_data(tickers, start, end):
    print("Downloading data...")
    all_tickers = tickers + [BENCH]
    data = yf.download(all_tickers, start=start, end=end, auto_adjust=True)["Close"]
    data = data.dropna(how='all', axis=1)
    if BENCH not in data.columns:
        raise ValueError(f"Benchmark {BENCH} data could not be fetched.")
    data = data.dropna()
    return data


def compute_log_returns(price_df):
    lr = np.log(price_df / price_df.shift(1)).dropna()
    return lr


# Portfolio optimization helpers

def portfolio_stats(weights, mu, Sigma, rf=RISK_FREE):
    port_ret = weights.dot(mu)
    port_vol = np.sqrt(weights.dot(Sigma).dot(weights))
    sharpe = (port_ret - rf) / port_vol if port_vol != 0 else 0
    return port_ret, port_vol, sharpe


def max_sharpe_portfolio(mu, Sigma):
    n = len(mu)
    def neg_sharpe(w):
        return -portfolio_stats(w, mu, Sigma)[2]
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},)
    bounds = tuple((0, 1) for _ in range(n))
    init = np.repeat(1/n, n)
    res = minimize(neg_sharpe, init, method='SLSQP', bounds=bounds, constraints=constraints)
    if not res.success:
        raise RuntimeError('Max Sharpe optimization failed: ' + res.message)
    w = res.x
    ret, vol, sharpe = portfolio_stats(w, mu, Sigma)
    return pd.Series(w), ret, vol, sharpe


# ---------------- STRATEGY FUNCTIONS ----------------

def apply_ema_strategy(price_series, short_span, long_span,
                       significance_pct=SIGNIFICANCE_PCT,
                       vol_window=VOL_WINDOW, vol_threshold=VOL_THRESHOLD,
                       tx_cost=TRANSACTION_COST):
    """Return a DataFrame with signals, positions, and strategy returns for an EMA crossover strategy.
    - Position = 1 when EMA_short > EMA_long * (1+significance)
    - Position = -1 when EMA_short < EMA_long * (1-significance)
    - If volatility above threshold, position = 0 (stay flat)
    - Apply transaction cost as a reduction when position changes (simple approximation)
    """
    df = price_series.to_frame(name='Close').copy()
    df['EMA_Short'] = df['Close'].ewm(span=short_span, adjust=False).mean()
    df['EMA_Long'] = df['Close'].ewm(span=long_span, adjust=False).mean()
    df['Volatility'] = df['Close'].pct_change().rolling(vol_window).std()

    df['Signal'] = 0
    df.loc[df['EMA_Short'] > df['EMA_Long'] * (1 + significance_pct), 'Signal'] = 1
    df.loc[df['EMA_Short'] < df['EMA_Long'] * (1 - significance_pct), 'Signal'] = -1

    # apply volatility filter
    df.loc[df['Volatility'] > vol_threshold, 'Signal'] = 0

    df['Position'] = df['Signal'].shift(1).fillna(0)  # act on yesterday's signal
    df['Daily_Return'] = df['Close'].pct_change().fillna(0)

    df['Strategy_Return'] = df['Daily_Return'] * df['Position']

    # approximate transaction cost: when position changes, subtract cost proportional to abs(change)
    pos_change = df['Position'].diff().abs().fillna(0)
    df['Strategy_Return_after_costs'] = df['Strategy_Return'] - pos_change * tx_cost

    df['Portfolio_Value'] = INITIAL_CAPITAL * (1 + df['Strategy_Return_after_costs']).cumprod()
    return df


# ---------------- GRID SEARCH (per-ticker) ----------------

def grid_search_best_ema(price_series, short_grid, long_grid):
    """Return best (short,long) pair from grid based on total return (after costs).
    Keeps short < long.
    """
    best_ret = -np.inf
    best_pair = (None, None)
    results = {}
    for s, l in product(short_grid, long_grid):
        if s >= l:
            continue
        df = apply_ema_strategy(price_series, s, l)
        total_ret = df['Portfolio_Value'].iloc[-1] / INITIAL_CAPITAL - 1
        results[(s, l)] = total_ret
        if total_ret > best_ret:
            best_ret = total_ret
            best_pair = (s, l)
    return best_pair, best_ret, results


# ---------------- MAIN WORKFLOW ----------------

def main():
    data = download_data(TICKERS, START, END)

    log_returns = compute_log_returns(data)
    asset_returns = log_returns[TICKERS]
    bench_ret = log_returns[BENCH]

    # === Portfolio optimization (mean-variance) ===
    mean_daily = asset_returns.mean()
    expected_annual_return = mean_daily * TRADING_DAYS
    cov_annual = asset_returns.cov() * TRADING_DAYS

    mu = expected_annual_return.values
    Sigma = cov_annual.values

    # Print covariance for debugging
    print("Annual covariance matrix:\n", cov_annual.round(6))

    # Max-Sharpe portfolio
    try:
        w_mvo, ret_opt, vol_opt, sharpe_opt = max_sharpe_portfolio(mu, Sigma)
    except RuntimeError as e:
        print("MVO failed:", e)
        w_mvo = pd.Series(np.repeat(1/len(TICKERS), len(TICKERS)), index=TICKERS)
        ret_opt, vol_opt, sharpe_opt = portfolio_stats(w_mvo.values, mu, Sigma)

    print("\nOptimized Portfolio (Max Sharpe):")
    print(w_mvo)
    print(f"Expected Return: {ret_opt:.2%}, Volatility: {vol_opt:.2%}, Sharpe: {sharpe_opt:.2f}")

    # Efficient frontier (quick approximation across target returns)
    target_returns = np.linspace(mu.min(), mu.max(), 50)
    ef_vols = []
    n = len(TICKERS)
    bounds = tuple((0, 1) for _ in range(n))
    cons_sum = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

    for tr in target_returns:
        cons = (
            cons_sum,
            {'type': 'eq', 'fun': lambda w, tr=tr: w.dot(mu) - tr}
        )
        res = minimize(lambda w: w.dot(Sigma).dot(w), np.repeat(1/n, n), method='SLSQP', bounds=bounds, constraints=cons)
        ef_vols.append(np.sqrt(res.fun) if res.success else np.nan)

    # === Per-ticker strategy: grid search then apply best EMA ===
    results = {}
    best_params = {}

    for ticker in TICKERS:
        print(f"\nTuning {ticker} EMA spans via grid search...")
        price_series = data[ticker]
        best_pair, best_ret, grid_results = grid_search_best_ema(price_series, EMA_SHORT_GRID, EMA_LONG_GRID)
        s_span, l_span = best_pair
        best_params[ticker] = {"short": s_span, "long": l_span, "total_return": best_ret}
        print(f"Best EMA for {ticker}: short={s_span}, long={l_span}, total_return={best_ret:.2%}")

        df = apply_ema_strategy(price_series, s_span, l_span)
        results[ticker] = df

        # Plot price + EMAs
        plt.figure(figsize=(10, 4))
        plt.plot(df['Close'], label=f"{ticker} Price")
        plt.plot(df['EMA_Short'], label=f"EMA {s_span}", linestyle='--')
        plt.plot(df['EMA_Long'], label=f"EMA {l_span}", linestyle='--')
        plt.title(f"{ticker} EMA Strategy (short={s_span}, long={l_span})")
        plt.legend()
        plt.grid(True)
        plt.show()

    # === Combine strategies using Sharpe-weighting ===
    strategy_returns = pd.DataFrame({t: results[t]['Strategy_Return_after_costs'] for t in TICKERS})
    # align indexes and drop initial NaNs
    strategy_returns = strategy_returns.dropna()

    per_asset_sharpes = {}
    for t in TICKERS:
        mean_r = strategy_returns[t].mean()
        std_r = strategy_returns[t].std()
        sr = (mean_r / std_r) * np.sqrt(TRADING_DAYS) if std_r != 0 else 0
        per_asset_sharpes[t] = max(sr, 0)  # clip negatives to zero so they don't get positive weight

    weights_sharpe = pd.Series(per_asset_sharpes)
    if weights_sharpe.sum() == 0:
        weights_sharpe = pd.Series(np.repeat(1/len(TICKERS), len(TICKERS)), index=TICKERS)
    else:
        weights_sharpe /= weights_sharpe.sum()

    print("\nSharpe-based weights for combining strategies:")
    print(weights_sharpe.round(4))

    combined = pd.DataFrame({t: results[t]['Strategy_Return_after_costs'] for t in TICKERS}).dropna()
    combined['Portfolio_Return'] = sum(combined[t] * weights_sharpe[t] for t in TICKERS)
    combined['Portfolio_Value'] = INITIAL_CAPITAL * (1 + combined['Portfolio_Return']).cumprod()

    # Buy & hold benchmark on same dates
    buy_hold = data[TICKERS].pct_change().loc[combined.index].mean(axis=1)
    buy_hold_value = INITIAL_CAPITAL * (1 + buy_hold).cumprod()

    # === PLOTS: portfolio comparison ===
    plt.figure(figsize=(12, 6))
    for t in TICKERS:
        plt.plot(results[t].loc[combined.index, 'Portfolio_Value'], label=f"{t} EMA Portfolio", alpha=0.7)
    plt.plot(combined['Portfolio_Value'], color='black', linewidth=2, label="Combined EMA Portfolio")
    plt.plot(buy_hold_value, color='orange', linewidth=2, linestyle='--', label='Buy & Hold (equal-weight)')
    plt.title('Portfolio Value Comparison')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Efficient frontier plot
    plt.figure(figsize=(8, 5))
    plt.plot(ef_vols, target_returns, label='Efficient Frontier')
    plt.scatter(vol_opt, ret_opt, c='red', label='Max Sharpe Portfolio', s=80)
    plt.xlabel('Volatility (Annual)')
    plt.ylabel('Expected Return (Annual)')
    plt.title('Efficient Frontier with Max-Sharpe Portfolio')
    plt.legend()
    plt.grid(True)
    plt.show()

    # === PERFORMANCE METRICS ===
    sharpe_combined = combined['Portfolio_Return'].mean() / combined['Portfolio_Return'].std() * np.sqrt(TRADING_DAYS)
    total_return_combined = combined['Portfolio_Value'].iloc[-1] / INITIAL_CAPITAL - 1
    max_drawdown = (combined['Portfolio_Value'] / combined['Portfolio_Value'].cummax() - 1).min()

    print("\nCombined EMA Portfolio Performance:")
    print(f"Total Return (since {combined.index[0].date()}): {total_return_combined:.2%}")
    print(f"Sharpe Ratio: {sharpe_combined:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")

    # print per-ticker best params summary
    print("\nPer-ticker best EMA parameters and returns:")
    for t in TICKERS:
        p = best_params[t]
        print(f"{t}: short={p['short']}, long={p['long']}, total_return={p['total_return']:.2%}")


if __name__ == '__main__':
    main()
