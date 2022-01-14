import random
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import scale
import numpy as np


def rbf_dot2(p1, p2, deg):
    if p1.ndim == 1:
        p1 = p1[:, np.newaxis]
        p2 = p2[:, np.newaxis]

    size1 = p1.shape
    size2 = p2.shape

    G = np.sum(p1 * p1, axis=1)[:, np.newaxis]
    H = np.sum(p2 * p2, axis=1)[:, np.newaxis]
    Q = np.tile(G, (1, size2[0]))
    R = np.tile(H.T, (size1[0], 1))
    H = Q + R - 2.0 * np.dot(p1, p2.T)
    H = np.exp(-H / 2.0 / (deg ** 2))

    return H


def rbf_dot(X, deg):
    if X.ndim == 1:
        X = X[:, np.newaxis]
    m = X.shape[0]
    G = np.sum(X * X, axis=1)[:, np.newaxis]
    Q = np.tile(G, (1, m))
    H = Q + Q.T - 2.0 * np.dot(X, X.T)
    if deg == -1:
        dists = (H - np.tril(H)).flatten()
        deg = np.sqrt(0.5 * np.median(dists[dists > 0]))
    H = np.exp(-H / 2.0 / (deg ** 2))

    return H


def FastHsicTestGamma(X, Y, sig=[-1, -1], maxpnt=200):
    m = X.shape[0]
    if m > maxpnt:
        indx = np.floor(np.r_[0:m:float(m - 1) / (maxpnt - 1)]).astype(int)
        #       indx = np.r_[0:maxpnt]
        Xm = X[indx].astype(float)
        Ym = Y[indx].astype(float)
        m = Xm.shape[0]
    else:
        Xm = X.astype(float)
        Ym = Y.astype(float)

    H = np.eye(m) - 1.0 / m * np.ones((m, m))

    K = rbf_dot(Xm, sig[0])
    L = rbf_dot(Ym, sig[1])

    Kc = np.dot(H, np.dot(K, H))
    Lc = np.dot(H, np.dot(L, H))

    testStat = (1.0 / m) * (Kc.T * Lc).sum()
    if ~np.isfinite(testStat):
        testStat = 0

    return testStat


def normalized_hsic(x, y):
    x = (x - np.mean(x)) / np.std(x)
    y = (y - np.mean(y)) / np.std(y)
    h = FastHsicTestGamma(x, y)

    return h


def create_pairdata(func='exp', n=500, sigma=0.09, seed=1):
    np.random.seed(seed)
    random.seed(seed)
    n_samples = n
    sigma = sigma

    x = np.random.uniform(-1.2, 1.2, n_samples)
    # x = np.random.normal(0, 1, n_samples)
    alpha = np.random.uniform(-1, 1)
    noise = np.random.normal(0, sigma, n_samples)

    if func == 'linear':
        y = alpha * x + noise

    elif func == 'exp':
        y = alpha * np.exp(x) + noise

    elif func == 'sin':
        y = alpha * np.sin(x) + noise

    elif func == 'tan':
        y = alpha * np.tan(x) + noise

    return np.c_[x, y]


def fanm_coreset(x, y, e=0.01, t=0.03, seed=1):
    np.random.seed(seed)
    random.seed(seed)
    N = x.shape[0]
    sampling = np.arange(e * N / 2, N, e * N).astype(np.int)
    data = np.c_[x, y]

    data_sampling = data[sampling]

    head = int(N * t)
    tail = int(N * (1 - t))
    p_head = data[:head, :]
    p_tail = data[tail:, :]

    w = 0.25
    if head < w / e:
        data_head = p_head
        data_tail = p_tail
    else:
        sampling_h = np.arange(0, head - 1, e * head / w).astype(np.int)
        data_head = p_head[sampling_h]
        data_tail = p_tail[sampling_h]
    data_coreset = np.r_[data_head, data_sampling, data_tail]
    x_coreset, y_coreset = data_coreset[:, 0], data_coreset[:, 1]
    x_coreset = x_coreset.reshape((-1, 1))
    y_coreset = y_coreset.reshape((-1, 1))
    gp = GaussianProcessRegressor().fit(x_coreset, y_coreset)

    return gp, data_coreset


def fast_ANM(data, t=0.03, e=0.01):
    data = scale(data)
    data_sorted0 = data[data[:, 0].argsort()]
    data_sorted1 = data[data[:, 1].argsort()]
    gp0, data0 = fanm_coreset(data_sorted0[:, 0], data_sorted0[:, 1], e=e, t=t, seed=1)
    gp1, data1 = fanm_coreset(data_sorted1[:, 1], data_sorted1[:, 0], e=e, t=t, seed=1)
    # forward
    x0, y0 = data0[:, 0].reshape((-1, 1)), data0[:, 1].reshape((-1, 1))
    y_predict = gp0.predict(x0)
    indepscore0 = normalized_hsic(y_predict - y0, x0)

    # backward
    y1, x1 = data1[:, 0].reshape((-1, 1)), data1[:, 1].reshape((-1, 1))
    x_predict = gp1.predict(y1)
    indepscore1 = normalized_hsic(x_predict - x1, y1)

    fanm_result = indepscore1 - indepscore0

    return fanm_result


if __name__ == '__main__':
    data = create_pairdata('tan', seed=18, n=1000)
    result = fast_ANM(data, t=0.03, e=0.01)
    print(result)
