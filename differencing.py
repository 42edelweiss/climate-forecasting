import random
from math import sin
import matplotlib.pyplot as plt

def differencing(ts):
    # diff[i] = ts[i+1] - ts[i]
    diff_ts = [ts[i+1] - ts[i] for i in range(len(ts) - 1)]
    return diff_ts, ts[0]

def integration(diff_ts, b):
    # reconstitue une série dont la première valeur est b
    int_ts = [b]
    for d in diff_ts:
        int_ts.append(int_ts[-1] + d)
    return int_ts

if __name__ == "__main__":
    random.seed(1)
    ts = [50 + 0.8 * i + 3 * sin(i) + 5 * random.random() for i in range(20)]

    diff_ts, b = differencing(ts)
    int_ts = integration(diff_ts, b)

    fig = plt.figure()

    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    ax1.set_title("Raw Time Series")
    ax1.plot(ts)

    ax2.set_title("Differenced Time Series")
    ax2.plot(diff_ts)

    ax3.set_title("Integrated (Reconstructed) Time Series")
    ax3.plot(int_ts)

    plt.tight_layout()
    plt.show()
