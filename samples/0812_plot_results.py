import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

csv_cvs1 = r'C:\Users\Alex\Downloads\results.csv'
csv_cvs3 = r'C:\Users\Alex\Downloads\results(2).csv'

map_key = '     metrics/mAP_0.5'

df1 = pd.read_csv(csv_cvs1)
map1 = df1[map_key].values

plt.scatter(np.arange(0, len(map1), 1), map1)
plt.grid()
plt.xlabel('epochs')
plt.ylabel('mAP 0.5')
plt.show()

df3 = pd.read_csv(csv_cvs3)
map3 = df3[map_key].values

plt.scatter(np.arange(0, len(map3), 1), map3)
plt.grid()
plt.xlabel('epochs')
plt.ylabel('mAP 0.5')
plt.show()

