# Capital Asset Pricing Model & Markowitz Portfolio Optimizer

An end-to-end portfolio analysis pipeline for 10 NIFTY 50 stocks, combining CAPM-based risk/return estimation with Markowitz mean-variance portfolio optimization, Monte Carlo efficient frontier visualization, and a monthly-rebalanced backtest benchmarked against a buy-and-hold equal-weight strategy.

## Contents

- `CAPM.py` — Estimates alpha, beta, and CAPM-implied expected return for each stock against the Nifty 50 index.
- `markowitz_model.py` — Builds constrained max-Sharpe portfolios, visualizes the efficient frontier, backtests with monthly rebalancing, and evaluates performance via Sharpe ratio, max drawdown, and turnover, with sensitivity testing across rebalance frequencies.

## Stocks Used

```
RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS, ICICIBANK.NS,
HINDUNILVR.NS, ITC.NS, SBIN.NS, BHARTIARTL.NS, KOTAKBANK.NS
```

Market benchmark: `^NSEI` (Nifty 50 index). Risk-free rate: fixed at 7% (approx. Indian 10-year G-Sec yield).

Period: 2015-01-01 to 2025-01-01.

## 1. CAPM (`CAPM.py`)

For each stock, monthly log returns are regressed against the Nifty 50's monthly log returns to estimate:

- **Beta** — computed two independent ways (covariance formula, and OLS regression slope) as a consistency check.
- **Alpha** — the regression intercept.
- **Expected return** — via the CAPM formula:

  ```
  E(R_i) = R_f + beta * (E(R_m) - R_f)
  ```

Run:

```bash
python CAPM.py
```

This prints beta (both methods), alpha, and expected return for each stock, and plots each stock's regression line against the market (equal-aspect axes, so a 45° line visually corresponds to beta = 1).

**Sanity check:** the simple average beta across all 10 stocks comes out close to 1.0 (~1.007), consistent with theory — the Nifty 50 index itself has beta = 1 against itself, so a broad-enough, representative sample of its constituents should average close to 1. (Note: this is only exact for a *market-cap-weighted* average across *all* 50 constituents; a simple average over a 10-stock subset is an approximation, not a guaranteed identity.)

## 2. Markowitz Optimization & Backtest (`markowitz_model.py`)

### Portfolio optimization
- Monte Carlo simulation of 10,000 random portfolios to visualize the efficient frontier.
- Constrained optimization (SLSQP) for the max-Sharpe portfolio, subject to:
  - **Long-only**: all weights in `[0, max_weight]`
  - **Fully-invested**: weights sum to 1
  - **Max-weight**: no single stock exceeds 30% (configurable via `max_weight`)

### Backtest
- `backtest_optimized()`: re-optimizes portfolio weights at each rebalance date using only a trailing 2-year lookback window of *past* data (no look-ahead bias), then holds those weights through the next period before re-optimizing again. Default rebalance frequency: monthly (`'ME'`).
- `backtest_buy_and_hold_equal_weight()`: buys an equal rupee amount of each stock once at the start and never rebalances, letting weights drift with price movements — the baseline the optimized strategy is compared against.

### Evaluation metrics
- **Sharpe ratio** — annualized, computed on the realized daily returns of each strategy's resulting value series (not just the in-sample optimizer objective).
- **Max drawdown** — largest peak-to-trough decline in portfolio value.
- **Turnover** — average sum of absolute weight changes per rebalance event, a proxy for trading/transaction-cost intensity.

### Sensitivity testing
`sensitivity_test()` reruns the full backtest at monthly, quarterly, and annual rebalancing frequencies, comparing Sharpe, max drawdown, and turnover across each — to check whether performance is robust to this choice or sensitive to it (a check against overfitting to one arbitrary parameter).

Run:

```bash
python markowitz_model.py
```

This prints the optimal portfolio weights, optimized vs. buy-and-hold Sharpe/drawdown, average turnover, and the sensitivity results across rebalance frequencies, alongside the efficient frontier and Sharpe-ratio plots.

## Known Limitations

- **Transaction costs are not modeled.** Turnover is reported as a proxy for trading intensity, but no fees/slippage are subtracted from returns.
- **Equal-weight (rebalanced) is not a separate benchmark.** Only two strategies are compared: SLSQP-optimized (monthly-rebalanced) vs. buy-and-hold equal-weight (never rebalanced). A separately-rebalanced equal-weight benchmark was considered but merged into buy-and-hold for scope reasons.
- **Risk-free rate is a fixed constant (7%)**, not a time-varying series pulled from actual bond yield data.
- Some stocks with a shorter trading history within the sample window are handled by trimming to the overlapping data available (see `CAPM.py`'s per-stock alignment), rather than requiring a uniform window across all stocks.

## Dependencies

```
numpy
pandas
yfinance
matplotlib
scikit-learn
scipy
```
