import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


class CAPM:

    def __init__(self, stocks, start_date, end_date):
        self.data = None
        self.stocks = stocks
        self.start_date = start_date
        self.end_date = end_date

    def download_data(self):

        data = {}

        for stock in self.stocks:
            ticker = yf.Ticker(stock)
            data[stock] = ticker.history(start=self.start_date, end=self.end_date)['Close']

        return pd.DataFrame(data)


    def initialize(self):
        stock_data = self.download_data()

        #we take monthly returns into account,So modifying the data accordingly
        stock_data = stock_data.resample('ME').last()

        self.data = pd.DataFrame({'s_close': stock_data[self.stocks[0]], 'm_close': stock_data[self.stocks[1]]})
        #calculating log returns
        self.data[['s_returns', 'm_returns']] = np.log(self.data[['s_close', 'm_close']]/
                                                 self.data[['s_close', 'm_close']].shift(1))
        #Removing the NaN values
        self.data = self.data[1:]

    def calculate_beta(self):
        #we need covariance matrix to calculate beta
        covariance_matrix = np.cov(self.data['s_returns'], self.data['m_returns'])
        #calculating beta from the formula
        beta = covariance_matrix[0,1]/ covariance_matrix[1,1]
        print('Beta from the formula :', beta)

    def regression(self):
        lr = LinearRegression()
        lr.fit(self.data['m_returns'].values.reshape(-1, 1), self.data['s_returns'].values)
        print('Beta from regression:', lr.coef_)
        self.plot_graph(lr.intercept_, lr.coef_)

    def plot_graph(self, alpha, beta):
        fig, axis = plt.subplots(1, figsize=(10,6))
        axis.scatter(x= self.data['m_returns'], y= self.data['s_returns'])
        axis.plot(self.data['m_returns'], beta* self.data['m_returns'] + alpha)
        plt.title('CAPM model, Finding Alpha and Beta')
        plt.xlabel('Market Returns')
        plt.ylabel('Expected return')
        plt.show()



if __name__ == '__main__':

    capm = CAPM(['IBM', '^GSPC'], '2015-01-01', '2025-01-01')
    capm.initialize()
    capm.calculate_beta()
    capm.regression()
    