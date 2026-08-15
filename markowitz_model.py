import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.optimize as optimization


num_of_trading_days = 252
no_of_portfolios = 10000
stocks = ['AAPL', 'TSLA', 'GE', 'AMZN']

start_date = '2015-01-01'
end_date = '2025-01-01'

def fetch_data():
    stock_data = {}

    for stock in stocks:
        ticker = yf.Ticker(stock)
        stock_data[stock] = ticker.history(start= start_date, end = end_date)['Close']

    return pd.DataFrame(stock_data)

def show_data(data):
    data.plot(figsize=(10, 6))
    plt.show()

def calculate_returns(stock_data):
    log_returns = np.log(stock_data/stock_data.shift(1))
    return log_returns[1:]


def statistics(weights, returns):
    portfolio_return = np.sum(returns.mean()* weights) * num_of_trading_days
    portfolio_volatility = np.sqrt(np.dot(weights.T , np.dot(returns.cov() * num_of_trading_days, weights)))
    return np.array([portfolio_return, portfolio_volatility, portfolio_return/portfolio_volatility])

def min_sharpe_func(weights, returns):
    return -statistics(weights, returns)[2]

def optimize_portfolio(weights, returns):
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x)-1}
    bounds = tuple((0,1) for _ in range(len(stocks)))
    return optimization.minimize(fun=min_sharpe_func, x0=weights[0], args=returns, method='SLSQP',
                          bounds= bounds, constraints=constraints)

def print_optimal_portfolio(optimum, returns):
    print('Optimal portfolio :', optimum['x'].round(3))
    print('return, volatility and sharpe ratio : ', statistics(optimum['x'], returns))

def show_optimized_sharpe_ratio(opt, ret, port_returns, port_risks):
    plt.figure(figsize=(10, 6))
    plt.scatter(x=port_risks, y=port_returns, c=port_risks / port_returns, marker='o')
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Expected volatility')
    plt.ylabel('Expected Returns')
    plt.grid(True)
    plt.plot(statistics(opt['x'], ret)[1], statistics(opt['x'], ret)[0], 'g*', markersize=10)
    plt.show()
def generate_portfolios(returns):
    portfolio_means = []
    portfolio_volatility = []
    portfolio_weights = []

    for _ in range(no_of_portfolios):
        w = np.random.random(len(stocks))
        w /= np.sum(w)
        portfolio_weights.append(w)
        portfolio_means.append(np.sum(returns.mean() * w) * num_of_trading_days)
        portfolio_volatility.append(np.sqrt(np.dot(w.T , np.dot(returns.cov() * num_of_trading_days, w))))

    return np.array(portfolio_weights), np.array(portfolio_means), np.array(portfolio_volatility)


def show_portfolios(returns, risks):
    plt.figure(figsize=(10, 6))
    plt.scatter(x=risks, y=returns, c = returns/risks, marker='o')
    plt.colorbar(label = 'Sharpe Ratio')
    plt.xlabel('Expected volatility')
    plt.ylabel('Expected Returns')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':

    stock_data = fetch_data()
    log_returns = calculate_returns(stock_data)
    weights, returns, risks = generate_portfolios(log_returns)
    show_portfolios(returns, risks)
    optimum = optimize_portfolio(weights, log_returns)
    print_optimal_portfolio(optimum, log_returns)
    show_optimized_sharpe_ratio(optimum, log_returns, returns, risks)
