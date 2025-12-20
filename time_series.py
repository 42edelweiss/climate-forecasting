# -*- coding: utf-8 -*-

import random
from math import sin, cos
import matplotlib.pyplot as plt

def get_time_series_data(length):
    a = 0.2
    b = 300
    c = 20
    ls = 5
    ms = 20
    gs = 100
    ts = []
    for i in range(length):
        ts.append(b + a*i + ls*sin(i/5) + ms*cos(i/24) + gs*sin(i/120) + c*random.random())
    return ts  # ✓ CORRECT - retourne après toute la boucle

if __name__ == '__main__':
    ts = get_time_series_data(1000)
    plt.plot(ts)
    plt.title('Time Series Data')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.show()