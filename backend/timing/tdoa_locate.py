import numpy as np
import matplotlib.pyplot as plt
from random import randrange

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

#Meters
def shuffle_tdoa_locate(source_loc, dist, n):
    x_s = []
    for _ in range(n):
        i = randrange(source_loc.shape[0])
        S_1 = source_loc[i]
        S_i = source_loc[0]
        dist_1 = dist[i]
        dist_i = dist[0]

        source_loc[0] = S_1
        source_loc[i] = S_i
        dist[0] = dist_1
        dist[i] = dist_i

        x_s.append(tdoa_locate(source_loc, dist))
    x_s = np.array(x_s)
    return np.mean(x_s, axis=0), np.linalg.norm(np.std(x_s, axis=0))

def prep_test(receive_dist, noise, num_recs):
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

    return locs, delays

def prep_test_uncorrelated(receive_dist, noise, num_recs):
    x = np.array([0, 0, 0])

    locs = []
    delays = []
    for i in range(num_recs):
        l = receive_dist * 2.0 * (np.random.random(3) - np.array([0.5, 0.5, 0.5]))
        l2 = receive_dist * 2.0 * (np.random.random(3) - np.array([0.5, 0.5, 0.5]))
        d = np.linalg.norm(x - l2) + noise * 2.0 * (np.random.random() - 0.5)
        locs.append(l)
        delays.append(d)

    locs.append(l)
    delays.append(d)

    locs = np.array(locs)
    delays = np.array(delays)

    return locs, delays

def run_test(receive_dist, noise, num_recs, n):
    locs, delays = prep_test(receive_dist, noise, num_recs)

    return shuffle_tdoa_locate(locs, delays, n)

def generate_datapoints(loc):
    receiver_locs = []
    delays = []

    for i in range(10):
        receiver_loc = 2.0 * (np.random.random(3) - 0.5)
        deltaR = receiver_loc - loc
        delay = np.linalg.norm(deltaR)
        receiver_locs.append(receiver_loc)
        delays.append(delay)
    
    receiver_locs = np.array(receiver_locs)
    delays = np.array(delays)

    return receiver_locs, delays


def run_demo():
    x = [run_test(10, 1, 10) for x in range(20)]
    import matplotlib.pyplot as plt
    x = [run_test(10, 1, 10)[0] for x in range(20)]
    plt.plot(x)
    plt.show()

def test2(N):
    truelocs = [2. * (np.random.random(3) - 0.5) for i in range(N)] 
    errs = []
    receiver_locs = []
    delay = []
    
    for i, trueloc in enumerate(truelocs):
        receiver_locs, delay = generate_datapoints(trueloc)
        predloc = tdoa_locate(receiver_locs, delay)
        err = np.linalg.norm(predloc - trueloc)
        errs.append(err)

    errs = np.array(errs)
    plt.figure(1)
    plt.hist(errs)
    plt.xlabel('Error = |true loc - pred loc|')
    plt.ylabel('Frequency')
    plt.savefig('errors.png')

    plt.figure(2)
    plt.scatter(receiver_locs[:,0], receiver_locs[:,1])
    plt.title('Receiver Locations')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('reciever_locs.png')
    
    plt.figure(3)
    plt.hist(delay)
    plt.title('Delay')
    plt.xlabel('Delay')
    plt.ylabel('Frequency')
    plt.savefig('delay.png')

    plt.show()

