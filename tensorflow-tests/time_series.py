import pandas as pd
import numpy as np
import random
import matplotlib
import tensorflow as tf

SEQ_LEN = 10
def create_time_series():
  freq = (np.random.random()*0.5) + 0.1  # 0.1 to 0.6
  ampl = np.random.random() + 0.5  # 0.5 to 1.5
  x = np.sin(np.arange(0,SEQ_LEN) * freq) * ampl
  return x

def to_csv(filename, N):
  with open(filename, 'w') as ofp:
    for lineno in range(0, N):
      seq = create_time_series()
      line = ",".join(map(str, seq))
      ofp.write(line + '\n')

#to_csv('train.csv', 1000)  # 1000 sequences
#to_csv('valid.csv',  50)


for i in range(0, 5):
  ts= create_time_series()  # 5 series
  ts.plot(c='b',  title='Example Time Series')
