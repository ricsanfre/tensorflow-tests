import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data as mnist_data

LOGDIR = "/home/ricardo/CODE/tensorflow-tests/logs/"
DATADIR = "/home/ricardo/CODE/tensorflow-tests/data/"

def main():
    print('Starting main')
    
    print('Loading mnist data')
    # Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
    mnist = mnist_data.read_data_sets(train_dir=DATADIR, one_hot=False, reshape=False, validation_size=0)
#    mnist = tf.contrib.learn.datasets.load_dataset("mnist",  train_dir=DATADIR)
    train_data= mnist.train.images
    train_labels= np.array (mnist.train.labels,  dtype=np.int32)
    eval_data= mnist.test.images
    eval_labels= np.array (mnist.test.labels,  dtype= np.int32)
    print (train_data[0])
    print (train_labels[0])



if __name__ == '__main__':
  main() 
