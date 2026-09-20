import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.optimize as optimization

num_of_trading_days = 252
no_of_portfolios = 10000
max_weight = 0.30
risk_free_rate = 0.07

nifty50_stocks = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
    'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS'
]

start_date = '2015-01-01'
end_date = '2025-01-01'

def fetch_data(tickers):
    stock_data = {}
    for stock in tickers:
        t = yf.Ticker(stock)
        stock_data[stock] = t.history(start=start_date, end=end_date)['Close']

    df = pd.DataFrame(stock_data)
    df.index = df.index.tz_localize(None)
    return df.dropna()

def calculate_returns(stock_data):
    log_returns = np.log(stock_data / stock_data.shift(1))
    return log_returns[1:]

def statistics(weights, returns, rf=risk_free_rate):
    portfolio_return = np.sum(returns.mean() * weights) * num_of_trading_days
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * num_of_trading_days, weights)))
    sharpe_ratio = (portfolio_return - rf) / portfolio_volatility
    return np.array([portfolio_return, portfolio_volatility, sharpe_ratio])

def min_sharpe_func(weights, returns, rf=risk_free_rate):
    return -statistics(weights, returns, rf)[2]

def optimize_portfolio(initial_weights, returns, rf=risk_free_rate, cap=max_weight):
    n = len(initial_weights)
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0, cap) for _ in range(n))
    return optimization.minimize(fun=min_sharpe_func, x0=initial_weights, args=(returns, rf),
                                  method='SLSQP', bounds=bounds, constraints=constraints)

def print_optimal_portfolio(optimum, returns):
    print('Optimal portfolio:', optimum['x'].round(6))
    print('return, volatility, sharpe:', statistics(optimum['x'], returns))

# ---------- MONTE CARLO / EFFICIENT FRONTIER ----------
def generate_portfolios(returns, n_stocks):
    portfolio_weights = []
    portfolio_means = []
    portfolio_volatility = []
    for _ in range(no_of_portfolios):
        w = np.random.random(n_stocks)
        w /= np.sum(w)
        portfolio_weights.append(w)
        portfolio_means.append(np.sum(returns.mean() * w) * num_of_trading_days)
        portfolio_volatility.append(np.sqrt(np.dot(w.T, np.dot(returns.cov() * num_of_trading_days, w))))
    return np.array(portfolio_weights), np.array(portfolio_means), np.array(portfolio_volatility)

def show_portfolios(returns, risks):
    plt.figure(figsize=(10, 6))
    plt.scatter(x=risks, y=returns, c=(returns - risk_free_rate) / risks, marker='o')
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Expected volatility')
    plt.ylabel('Expected Returns')
    plt.title('Efficient Frontier (Monte Carlo)')
    plt.grid(True)
    plt.show()

def show_optimized_sharpe_ratio(opt, ret, port_returns, port_risks):
    plt.figure(figsize=(10, 6))
    plt.scatter(x=port_risks, y=port_returns, c=(port_returns - risk_free_rate) / port_risks, marker='o')
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Expected volatility')
    plt.ylabel('Expected Returns')
    plt.grid(True)
    stats = statistics(opt['x'], ret)
    plt.plot(stats[1], stats[0], 'g*', markersize=15, label='Max Sharpe Portfolio')
    plt.legend()
    plt.show()

# ---------- METRICS ----------
def max_drawdown(portfolio_values):
    running_peak = np.maximum.accumulate(portfolio_values)
    drawdowns = (portfolio_values - running_peak) / running_peak
    return drawdowns.min()

def compute_turnover(weight_history):
    turnovers = []
    for t in range(1, len(weight_history)):
        turnovers.append(np.sum(np.abs(weight_history[t] - weight_history[t - 1])))
    return np.array(turnovers)

def sharpe_of_series(returns_series, rf=risk_free_rate):
    ann_return = returns_series.mean() * num_of_trading_days
    ann_vol = returns_series.std() * np.sqrt(num_of_trading_days)
    return (ann_return - rf) / ann_vol

# ---------- BACKTESTING ----------
def backtest_optimized(price_data, rebalance_freq='ME', lookback_days=252 * 2, cap=max_weight):
    log_returns = calculate_returns(price_data)
    rebalance_dates = price_data.resample(rebalance_freq).last().index
    rebalance_dates = rebalance_dates[rebalance_dates >= log_returns.index[0] + pd.Timedelta(days=lookback_days)]

    portfolio_value = [1.0]
    portfolio_dates = [rebalance_dates[0]]
    weight_history = []

    for i in range(len(rebalance_dates) - 1):
        rebal_date = rebalance_dates[i]
        next_date = rebalance_dates[i + 1]

        window = log_returns[log_returns.index <= rebal_date].tail(lookback_days)
        if len(window) < 60:
            continue

        n = window.shape[1]
        init_weights = np.ones(n) / n
        result = optimize_portfolio(init_weights, window, cap=cap)
        current_weights = result['x']
        weight_history.append(current_weights)

        period_returns = log_returns[(log_returns.index > rebal_date) & (log_returns.index <= next_date)]
        period_portfolio_log_returns = period_returns.dot(current_weights)

        for daily_log_ret in period_portfolio_log_returns:
            portfolio_value.append(portfolio_value[-1] * np.exp(daily_log_ret))
        portfolio_dates.extend(period_returns.index.tolist())

    return np.array(portfolio_value), portfolio_dates, np.array(weight_history)

def backtest_buy_and_hold_equal_weight(price_data):
    n = price_data.shape[1]
    start_prices = price_data.iloc[0]
    units = (1.0 / n) / start_prices

    portfolio_value = (price_data * units).sum(axis=1)
    return portfolio_value.values, price_data.index.tolist()

# ---------- SENSITIVITY ----------
def sensitivity_test(price_data, frequencies=('ME', 'QE', 'YE')):
    results = {}
    for freq in frequencies:
        values, dates, weights_hist = backtest_optimized(price_data, rebalance_freq=freq)
        rets = pd.Series(values).pct_change().dropna()
        results[freq] = {
            'sharpe': sharpe_of_series(rets),
            'max_drawdown': max_drawdown(values),
            'avg_turnover': compute_turnover(weights_hist).mean() if len(weights_hist) > 1 else np.nan
        }
    return results

if __name__ == '__main__':
    price_data = fetch_data(nifty50_stocks)
    log_returns = calculate_returns(price_data)

    weights, m_returns, m_risks = generate_portfolios(log_returns, len(price_data.columns))
    show_portfolios(m_returns, m_risks)
    optimum = optimize_portfolio(weights[0], log_returns)
    print_optimal_portfolio(optimum, log_returns)
    show_optimized_sharpe_ratio(optimum, log_returns, m_returns, m_risks)

    opt_values, opt_dates, opt_weights_hist = backtest_optimized(price_data, rebalance_freq='ME')
    bh_values, bh_dates = backtest_buy_and_hold_equal_weight(price_data)

    opt_returns = pd.Series(opt_values).pct_change().dropna()
    bh_returns = pd.Series(bh_values).pct_change().dropna()

    print('Optimized strategy - Sharpe:', sharpe_of_series(opt_returns),
          'Max Drawdown:', max_drawdown(opt_values))
    print('Buy-and-hold (equal-weight) strategy - Sharpe:', sharpe_of_series(bh_returns),
          'Max Drawdown:', max_drawdown(bh_values))

    turnover = compute_turnover(opt_weights_hist)
    print('Average turnover per rebalance:', turnover.mean())

    sens = sensitivity_test(price_data)
    for freq, res in sens.items():
        print(freq, res)