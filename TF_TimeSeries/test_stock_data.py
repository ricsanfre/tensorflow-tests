import tensorflow as tf
from stock_data_gdax import StockDataSet


def load_data(stock_name, input_size, num_steps):
    stock_dataset = StockDataSet(stock_name, input_size=input_size, num_steps=num_steps,
                                 test_ratio=0.0, close_price_only=True, normalized=False)
    print("Raw sequence size:", stock_dataset.raw_seq.size)
    print("Raw sequence shape:", stock_dataset.raw_seq.shape)
    print("Raw sequence dim:", stock_dataset.raw_seq.ndim)
    print ("Train data size:", len(stock_dataset.train_X))
    print ("Train data shape:",  stock_dataset.train_X.shape)
    print ("Label data size:", len(stock_dataset.train_y))
    print ("Label data shape:",  stock_dataset.train_y.shape)
    
    return stock_dataset


stock_dataset = load_data('BTC-EUR',
                              input_size=1,
                              num_steps=4*24)
                            
dataset = tf.data.Dataset.from_tensor_slices((stock_dataset.train_X,  stock_dataset.train_y))

batched_dataset=dataset.batch(4)

iterator=batched_dataset.make_one_shot_iterator()
X,  Y = iterator.get_next()

print ("First Element X",  X.shape)
print ("First Element Y",  Y.shape)
