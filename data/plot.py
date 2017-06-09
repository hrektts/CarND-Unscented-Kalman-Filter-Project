#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('lidar.txt', header=None, skiprows=2)
nis = np.ones(len(data[0]))
chi_squared_value = 5.99  # DF: 2, P: 0.05
nis *= chi_squared_value

plt.plot(data[0], label='Sampled')
plt.plot(nis, label='Chi^2 value: 5.99')
plt.xlabel('Sample')
plt.ylabel('NIS')
plt.legend()
plt.savefig('nis_lidar.png')

plt.clf()

data = pd.read_csv('radar.txt', header=None, skiprows=2)
nis = np.ones(len(data[0]))
chi_squared_value = 7.82  # DF: 3, P: 0.05
nis *= chi_squared_value

plt.plot(data[0], label='Sampled')
plt.plot(nis, label='Chi^2 value: 7.82')
plt.xlabel('Sample')
plt.ylabel('NIS')
plt.legend()
plt.savefig('nis_radar.png')
