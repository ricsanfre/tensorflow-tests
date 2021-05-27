# Data time series generator
# This Script generates CSV files for training and testing
# Each line in the CSV will contain a data serie corresponding to 5 weeks hourly data (5*24*7).
# 4 weeks hourly data (1-4*24*7) will be used to predict 1 week data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SEQ_LEN = 5*24*7 # each serie represents the traffic within five weeks (hourly data).
def create_time_series():
  freq = (np.random.random()*0.5) + 0.1  # 0.1 to 0.6
  ampl = np.random.random() + 0.5  # 0.5 to 1.5
  rng = pd.date_range(start='1/1/2018', periods=SEQ_LEN, freq='H')
  sin = pd.Series((np.sin(np.arange(0,len(rng)) * freq) * ampl)+1.5,  index=rng)
  x= sin.round(4)
  return x

def show_graph():
  for i in ['b', 'g',  'r', 'c', 'y']:
    ts= create_time_series()  # 5 series
    ts.plot(c=i,  title='Example Time Series')
  plt.show()
def to_csv(filename, N):
  with open(filename, 'w') as ofp:
    for lineno in range(0, N):
      seq = create_time_series()
      line = ",".join(map(str, seq))
      ofp.write(line + '\n')

to_csv('train.csv', 1000)  # 1000 sequences
to_csv('valid.csv',  50)
#show_graph()



