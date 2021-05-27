# Get BTC coin rates using GDAX or Quandl
import pandas as pd
import numpy as np
import pickle
import quandl
import matplotlib.pyplot as plt
from datetime import datetime
from gdax import GDAX


def btc_usd_30min(start, end):
  return GDAX('BTC-USD').fetch(start, end, 30)

def btc_eur_15min(start, end):
  return GDAX('BTC-EUR').fetch(start, end, 15)

def ltc_eur_1day(start, end):
  return GDAX('LTC-EUR').fetch(start, end, 1440)

def get_quandl_data(quandl_id):
    '''Download and cache Quandl dataseries'''
    cache_path = '{}.pkl'.format(quandl_id).replace('/','-')
    try:
        f = open(cache_path, 'rb')
        df = pickle.load(f)   
        print('Loaded {} from cache'.format(quandl_id))
    except (OSError, IOError) as e:
        print('Downloading {} from Quandl'.format(quandl_id))
        df = quandl.get(quandl_id, returns="pandas")
        df.to_pickle(cache_path)
        print('Cached {} at {}'.format(quandl_id, cache_path))
    return df


# Pull Kraken BTC price exchange data

#btc_usd_price_kraken = quandl.get('BCHARTS/KRAKENUSD', returns="pandas")
#btc_usd_price_kraken.plot(c='b',  title='BTC/USD')
#btc_usd_price_kraken.to_csv("bitcoin.csv")
#plt.show()

data_frame = btc_eur_15min(datetime(2016, 1, 1), datetime(2018, 2, 15))


# Save to CSV.
data_frame.to_csv('bitcoin_data.csv')



