# Get BTC coin rates using GDAX
from datetime import datetime
from gdax import GDAX

# `tf.parse_csv` sets the types of the outputs to match the examples given in
#     the `record_defaults` argument.
CSV_TYPES = [[0.0], [""], [0.0], [0.0], [0.0], [0.0], [0.0]]
CSV_COLUMN_NAMES = ['Time', 'Time_text','Low', 'High', 'Open', 'Close', 'Volume']

CSV_NAME='data/BTC-EUR.csv'


def btc_usd_30min(start, end):
  return GDAX('BTC-USD').fetch(start, end, 30)

def btc_eur_15min(start, end):
  return GDAX('BTC-EUR').fetch(start, end, 15)

def btc_eur_1day(start, end):
  return GDAX('BTC-EUR').fetch(start, end, 1440)

def btc_eur_1hour(start, end):
  return GDAX('BTC-EUR').fetch(start, end, 60)


data_frame = btc_eur_1hour(datetime(2016, 1, 1), datetime(2018, 2, 15))
data_frame.to_csv(CSV_NAME)
    
 



