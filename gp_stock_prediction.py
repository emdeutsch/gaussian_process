import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm
from sklearn.metrics.pairwise import euclidean_distances as distance
import requests


def main():
    ticker = 'AAPl'
    start_year = '1980'

    n = 150

    data_frame = generate_data(ticker, start_year)

    train_n = 4900
    test_n = len(data_frame[f'{ticker}_close_returns']) - train_n


    x1_train = np.array(data_frame[f'{ticker}_close_returns'][1:train_n-2].array)
    x2_train = np.array(data_frame[f'{ticker}_close_returns'][2:train_n-1].array)

    train_x = []
    train_x.append(x1_train)
    train_x.append(x2_train)
    train_x = np.array(train_x).T
    train_y = np.asarray(data_frame[f'{ticker}_close_returns'][3:train_n].array)


    x1_test = data_frame[f'{ticker}_close_returns'][train_n:train_n+test_n - 2].array
    x2_test = data_frame[f'{ticker}_close_returns'][train_n+1:train_n+test_n - 1].array
    test_x = []
    test_x.append(x1_test)
    test_x.append(x2_test)
    test_x = np.array(test_x).T
    test_y = np.asarray(data_frame[f'{ticker}_close_returns'][train_n+2:].array)

    print(train_n+test_n)


    gp = GP(0, n, twod=True)

    gp.add_all(train_x, train_y)

    plot_mu = []
    plot_q1 = []
    plot_q2 = []

    mu = gp.mu.reshape(n, n)
    q1 = gp.q1.reshape(n, n)
    q2 = gp.q2.reshape(n, n)

    for i in range(test_n - 2):
        plot_mu.append(mu[int(np.round(test_x[i][0]*(n/2))) + int(n/2)][int(np.round(test_x[i][1]*(n/2))) + int(n/2)])
        plot_q1.append(q1[int(np.round(test_x[i][0]*(n/2))) + int(n/2)][int(np.round(test_x[i][1]*(n/2))) + int(n/2)])
        plot_q2.append(q2[int(np.round(test_x[i][0]*(n/2))) + int(n/2)][int(np.round(test_x[i][1]*(n/2))) + int(n/2)])


    plt.plot(plot_mu, 'k')
    plt.plot(test_y, 'b')
    plt.plot(plot_q1, 'r--')
    plt.plot(plot_q2, 'r--')
    plt.show()


def run_1d(data_frame):
    ticker = 'AMZN'
    start_year = '2019'

    n = 500

    data_frame = generate_data(ticker, start_year)

    train_x = np.asarray(data_frame[f'{ticker}_close_returns'][1:508])
    train_y = np.asarray(data_frame[f'{ticker}_close_returns'][2:508])
    test_y = np.asarray(data_frame[f'{ticker}_close_returns'][508:])

    train_n = len(train_x)
    test_n = len(test_y)
    print(test_n)

    gp = GP(0, n)

    for i in range(train_n - 1):
        print(i)
        gp.add([train_x[i]], train_x[i+1])

    plot_mu = []
    plot_q1 = []
    plot_q2 = []

    for i in range(test_n):
        plot_mu.append(gp.mu[int(np.round(train_y[i] * (n / 2))) + int(n / 2)])
        plot_q1.append(gp.q1[int(np.round(train_y[i] * (n / 2))) + int(n / 2)])
        plot_q2.append(gp.q2[int(np.round(train_y[i] * (n / 2))) + int(n / 2)])
        print(i)
        print(int(np.round(train_y[i] * (n / 2))) + int(n / 2))

    plt.plot(plot_mu, 'k')
    plt.plot(test_y, 'b')
    plt.plot(plot_q1, 'r--')
    plt.plot(plot_q2, 'r--')
    plt.show()

def run_time_covariance(data_frame):
    n = len(data_frame['AAPL_close'])

    X = np.linspace(0, n, n).reshape(-1, 1)
    Y = data_frame['AAPL_close']

    gp = GP(1, n)

    for i in range(n - 30):
        gp.add(X[i], Y[i])
        # gp.add(np.array([data_frame['AAPL_close'][i]]), data_frame['AAPL_close'][i + 1])

    plt.plot(gp.mu, 'k')
    plt.plot(Y, 'b')
    plt.plot(gp.q1, 'r--')
    plt.plot(gp.q2, 'r--')
    plt.show()


def generate_data(ticker, start_year):
    api_key = '3b2bdb6cee3045dabbec01da8c4248cf'
    ticker = f'{ticker}'
    interval = '1day'
    api_url = f'https://api.twelvedata.com/time_series?symbol={ticker}&start_date={start_year}-01-01&end_date=2021-03-21&order=ASC&interval={interval}&apikey={api_key}'

    data = requests.get(api_url).json()
    data_frame = pd.DataFrame(data['values'])
    # data_frame.datetime = pd.to_datetime(data_frame.datetime, dayfirst=True)
    # data_frame.set_index("datetime", inplace=True)
    data_frame['open'] = pd.to_numeric(data_frame['open'])
    data_frame[f'{ticker}_open_returns'] = data_frame.open.pct_change(1)
    data_frame[f'{ticker}_open'] = data_frame['open']
    data_frame['close'] = pd.to_numeric(data_frame['close'])
    data_frame[f'{ticker}_close_returns'] = data_frame.close.pct_change(1)
    data_frame[f'{ticker}_close'] = data_frame['close']
    del data_frame['high'], data_frame['low'], data_frame['volume'], data_frame['close'], data_frame['open']

    return data_frame

class GP:
    def __init__(self, expected_mu, n, l=1.0, noise=0.0, twod=False):
        self.jitter = 0.001
        self.n = n
        self.X = []
        self.Y = []
        self.l = l
        self.noise = noise
        self.expected_mu = expected_mu
        self.mu = np.zeros(n) + expected_mu
        self.sig = np.identity(n)
        if twod:
            self.XX = np.array([((x/(8*n)) - 0.0625, ((y/(8*n)) - 0.0625)) for x in range(n) for y in range(n)])
        else:
            self.XX = np.linspace(-1, 1, n).reshape(-1, 1)
        print('XX ' + str(self.XX.shape))
        self.acqu_val = self.mu.reshape(-1) - 1.8*np.sqrt(np.diag(self.sig)).reshape(-1)

    def add(self, X, Y):
        self.X.append(X)
        self.Y.append(Y - self.expected_mu)

        X = np.array(self.X)
        Y = np.array(self.Y)
        XX = self.XX

        D = distance(X, X)
        Sigma = np.exp(-0.5*(1/np.power(self.l, 2))*D) + np.diag(np.repeat(0.00000001490116, len(X))) + np.power(self.noise, 2)*np.identity(len(X))


        DXX = distance(XX, XX)
        SXX = np.exp(-0.5*(1/np.power(self.l, 2))*DXX) + np.diag(np.repeat(0.00000001490116, self.n))

        DX = distance(XX, X)
        SX = np.exp(-0.5*(1/np.power(self.l, 2))*DX)

        Si = np.linalg.inv(Sigma)

        self.mu = SX.dot(Si).dot(Y) + self.expected_mu
        self.sig = SXX - SX.dot(Si).dot(SX.transpose())
        self.acqu_val = self.mu.reshape(-1) - 1.8*np.sqrt(np.diag(self.sig)).reshape(-1)
        self.q1 = norm.ppf(0.05, self.mu.reshape(-1), np.sqrt(np.diag(self.sig)))
        self.q2 = norm.ppf(0.95, self.mu.reshape(-1), np.sqrt(np.diag(self.sig)))

    def add_all(self, X, Y):
        self.X = X
        self.Y = Y
        print('X ' + str(X.shape))

        XX = self.XX

        D = distance(X, X)
        Sigma = np.exp(-0.5 * (1 / np.power(self.l, 2)) * D) + np.diag(np.repeat(0.00000001490116, len(X))) + np.power(
            self.noise, 2) * np.identity(len(X))

        DXX = distance(XX, XX)
        print('DXX ' + str(DXX.shape))
        SXX = np.exp(-0.5 * (1 / np.power(self.l, 2)) * DXX) #+ np.diag(np.repeat(0.00000001490116, self.n ** self.n))
        print('SXX ' + str(SXX.shape))


        DX = distance(XX, X)
        print('DX ' + str(DX.shape))
        SX = np.exp(-0.5 * (1 / np.power(self.l, 2)) * DX)

        Si = np.linalg.inv(Sigma)

        self.mu = SX.dot(Si).dot(Y) + self.expected_mu
        print('mu ' + str(self.mu.shape))
        self.sig = SXX - SX.dot(Si).dot(SX.transpose())
        print('sig ' + str(self.sig.shape))
        self.acqu_val = self.mu.reshape(-1) - 1.8 * np.sqrt(np.diag(self.sig)).reshape(-1)
        self.q1 = norm.ppf(0.05, self.mu.reshape(-1), np.sqrt(np.diag(self.sig)))
        self.q2 = norm.ppf(0.95, self.mu.reshape(-1), np.sqrt(np.diag(self.sig)))

    def conditional_gaussian(self, X):
        tempX = self.X
        tempX.append(X)

        X = np.array(tempX)
        X_avg = np.average(X)

        XX = self.XX

        D = distance(X, X)
        Sigma = np.exp(-0.5 * (1 / np.power(self.l, 2)) * D) + np.diag(np.repeat(0.00000001490116, len(X))) + np.power(
            self.noise, 2) * np.identity(len(X))

        DXX = distance(XX, XX)
        SXX = np.exp(-0.5 * (1 / np.power(self.l, 2)) * DXX) + np.diag(np.repeat(0.00000001490116, self.n))

        DX = distance(XX, X)
        SX = np.exp(-0.5 * (1 / np.power(self.l, 2)) * DX)

        Si = np.linalg.inv(Sigma)

        return (SX.dot(Si).dot(X - X_avg), SXX - SX.dot(Si).dot(SX.transpose()))



    def plot_gp(self, objective_X, objective_Y):
        X = np.array(self.X)
        Y = np.array(self.Y)
        XX = self.XX

        plt.plot(X, Y + self.expected_mu, 'ko')
        plt.plot(XX, self.mu, 'k-')
        plt.plot(XX, self.q1, 'r--')
        plt.plot(XX, self.q2, 'r--')
        plt.plot(objective_X, objective_Y, 'c-')
        plt.show()

    def plot_acquisiton(self):
        plt.plot(self.XX, self.acqu_val, 'y-')
        plt.show()

    def max_acquisition(self):
        return np.argmin(self.acqu_val)








if __name__ == "__main__":
    main()