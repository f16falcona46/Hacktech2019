import numpy as np

def tdoa_locate(source_loc, dist):
    d = dist[1:] - dist[0]
    orig = source_loc[0]
    S = source_loc[1:] - source_loc[0]
    delta = np.sum(np.multiply(S, S), axis=1) - np.multiply(d, d)
    W = np.eye(d.shape[0])
    P_1_d = np.eye(d.shape[0]) - np.outer(d, d)/np.dot(d, d)

    PWP = P_1_d @ W @ P_1_d

    x_estimate = 0.5 * np.linalg.inv(S.T @ PWP @ S) @ S.T @ PWP @ delta
    return x_estimate + orig

def run_test(receive_dist, noise, num_recs):
    x = np.array([0, 0, 0])

    locs = []
    delays = []
    for i in range(num_recs):
        l = receive_dist * 2.0 * (np.random.random(3) - np.array([0.5, 0.5, 0.5]))
        d = np.linalg.norm(x - l) + noise * 2.0 * (np.random.random() - 0.5)
        locs.append(l)
        delays.append(d)

    locs.append(l)
    delays.append(d)

    locs = np.array(locs)
    delays = np.array(delays)

    return tdoa_locate(locs, delays)

def run_demo():
    import matplotlib.pyplot as plt
    x = [run_test(10, 1, 10) for x in range(20)]
    plt.plot(x)
    plt.show()
