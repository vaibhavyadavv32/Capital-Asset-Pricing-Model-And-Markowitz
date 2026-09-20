import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class CAPM:
    def __init__(self, stocks, market, start_date, end_date, risk_free_rate):
        self.data = None
        self.stocks = stocks
        self.market = market
        self.start_date = start_date
        self.end_date = end_date
        self.risk_free_rate = risk_free_rate

    def download_data(self):

        tickers = self.stocks + [self.market]
        data = {}
        for ticker in tickers:
            t = yf.Ticker(ticker)
            data[ticker] = t.history(start=self.start_date, end=self.end_date)['Close']

        df  = pd.DataFrame(data)
        df.index = df.index.tz_localize(None)

        return df.dropna()

    def initialize(self):
        price_data = self.download_data()
        price_data = price_data.resample('ME').last()
        log_returns = np.log(price_data / price_data.shift(1))
        self.data = log_returns[1:]

    def run_capm(self):
        market_returns = self.data[self.market]
        expected_market_return = market_returns.mean() * 12

        results = {}
        for stock in self.stocks:
            stock_returns = self.data[stock]

            combined = pd.concat([stock_returns, market_returns], axis=1).dropna()
            if combined.empty or len(combined) < 12:
                print(f"Warning: insufficient overlapping data for {stock}, skipping")
                continue

            s_ret = combined[stock]
            m_ret = combined[self.market]

            covariance_matrix = np.cov(s_ret, m_ret)
            beta = covariance_matrix[0, 1] / covariance_matrix[1, 1]

            lr = LinearRegression()
            lr.fit(m_ret.values.reshape(-1, 1), s_ret.values)
            alpha = lr.intercept_
            beta_regression = lr.coef_[0]
            expected_return = self.risk_free_rate + beta * (expected_market_return - self.risk_free_rate)
            beta_diff = abs(beta - beta_regression)

            results[stock] = {
                'beta': beta,
                'beta_regression': beta_regression,
                'beta_diff': beta_diff,
                'alpha': alpha,
                'expected_return': expected_return
            }

        return results

    def plot_graph(self, stock, alpha, beta):
        stock_returns = self.data[stock]
        market_returns = self.data[self.market]

        fig, axis = plt.subplots(1, figsize=(6, 6))
        axis.scatter(x=market_returns, y=stock_returns)
        axis.plot(market_returns, beta * market_returns + alpha, color='red')
        axis.grid()
        axis.set_aspect('equal')
        plt.title(f'CAPM: {stock} vs {self.market}')
        plt.xlabel('Market Returns')
        plt.ylabel(f'{stock} Returns')
        plt.show()

if __name__ == '__main__':
    nifty50_stocks = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
                    'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS']
    market_index = '^NSEI'
    risk_free_rate = 0.07

    capm = CAPM(nifty50_stocks, market_index, '2015-01-01', '2025-01-01', risk_free_rate)
    capm.initialize()
    results = capm.run_capm()

    for stock, metrics in results.items():
        print(f"{stock}: beta_cov={metrics['beta']:.4f}, beta_reg={metrics['beta_regression']:.4f}, "
            f"diff={metrics['beta_diff']:.6f}, alpha={metrics['alpha']:.4f}, "
            f"expected_return={metrics['expected_return']:.4f}")

        capm.plot_graph(stock, results[stock]['alpha'], results[stock]['beta_regression'])

    # capm.plot_graph('RELIANCE.NS', results['RELIANCE.NS']['alpha'], results['RELIANCE.NS']['beta_regression'])