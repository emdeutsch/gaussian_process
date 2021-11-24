import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from sklearn.metrics.pairwise import euclidean_distances as distance


def main():
    n = 500
    expected_mu = -5
    objective_func = objective(expected_mu, n)
    X = objective_func[0]
    Y = objective_func[1].transpose()
    objective_noise = True

    gp = GP(expected_mu, n, 1, 0.25)
    i = 0
    while i < 100:
        gp.plot_gp(X, Y)
        gp.plot_acquisiton()
        index = gp.max_acquisition()
        if objective_noise:
            gp.add(X[index], np.random.normal(loc=Y[index], scale=0.25))
        else:
            gp.add(X[index], Y[index])
        i += 1

    #plot_predict_gaussian()


def plot_predict_gaussian():
    n = 24
    X = np.linspace(-4*np.pi, 4*np.pi, n).reshape(-1, 1)
    Y = np.sin(X) - np.sin(X/2)

    D = distance(X, X)
    Sigma = np.exp(-D)

    XX = np.linspace(-5*np.pi, 5*np.pi, 500).reshape(-1, 1)
    DXX = distance(XX, XX)
    SXX = np.exp(-DXX)

    DX = distance(XX, X)
    SX = np.exp(-DX)

    Si = np.linalg.inv(Sigma)
    mu = SX.dot(Si).dot(Y)
    Sig = SXX - SX.dot(Si).dot(SX.transpose())

    YY = np.random.multivariate_normal(mu.reshape(-1), Sig, 100)

    q1 = norm.ppf(0.05, mu.reshape(-1), np.sqrt(np.diag(Sig)))
    q2 = norm.ppf(0.95, mu.reshape(-1), np.sqrt(np.diag(Sig)))

    plt.plot(XX, YY.transpose(), 'c-')
    plt.plot(X, Y, 'ko')
    plt.plot(XX, np.sin(XX) - np.sin(XX/2), 'b-')
    plt.plot(XX, mu, 'k-')
    plt.plot(XX, q1, 'r--')
    plt.plot(XX, q2, 'r--')
    plt.show()


def plot_random_gaussian(num):
    n = 100
    X = np.linspace(0, 10, n).reshape(-1, 1)
    print(X.shape)

    D = distance(X, X)
    Sigma = np.exp(-D)

    Y = np.random.multivariate_normal(np.zeros(n), Sigma, num)
    print(Y.shape)

    plt.plot(X, Y.transpose())

    plt.show()


def objective(expected_mu, n):
    X = np.linspace(-3, 3, n).reshape(-1, 1)

    D = distance(X, X)
    Sigma = np.exp(-D)

    Y = np.random.multivariate_normal(np.zeros(n) + expected_mu, Sigma, 1)

    return X, Y

class GP:
    def __init__(self, expected_mu, n, l=0.0, noise=0.0):
        self.jitter = 0.001
        self.n = n
        self.X = []
        self.Y = []
        self.l = l
        self.noise = noise
        self.expected_mu = expected_mu
        self.mu = np.zeros(n) + expected_mu
        self.sig = np.identity(n)
        self.XX = np.linspace(-3, 3, self.n).reshape(-1, 1)
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

    def plot_gp(self, objective_X, objective_Y):
        X = np.array(self.X)
        Y = np.array(self.Y)
        XX = self.XX

        q1 = norm.ppf(0.05, self.mu.reshape(-1), np.sqrt(np.diag(self.sig)))
        q2 = norm.ppf(0.95, self.mu.reshape(-1), np.sqrt(np.diag(self.sig)))

        plt.plot(X, Y + self.expected_mu, 'ko')
        plt.plot(XX, self.mu, 'k-')
        plt.plot(XX, q1, 'r--')
        plt.plot(XX, q2, 'r--')
        plt.plot(objective_X, objective_Y, 'c-')
        plt.show()

    def plot_acquisiton(self):
        plt.plot(self.XX, self.acqu_val, 'y-')
        plt.show()

    def max_acquisition(self):
        return np.argmin(self.acqu_val)








if __name__ == "__main__":
    main()