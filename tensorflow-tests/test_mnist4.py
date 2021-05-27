#import os.path
import tensorflow as tf
import math

LOGDIR = "/home/ricardo/CODE/tensorflow-tests/logs/"
DATADIR = "/home/ricardo/CODE/tensorflow-tests/data/"
### MNIST EMBEDDINGS ###
mnist = tf.contrib.learn.datasets.mnist.read_data_sets(train_dir=DATADIR, one_hot=True,  reshape=False, validation_size=0)

def fc_layer(input, size_in, size_out, name="fc",  activation="sigmoid"):
  with tf.name_scope(name):
    # weights are initilialized ramdomly    
    w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
    if activation == "sigmoid":
        act =  tf.nn.sigmoid(tf.matmul(input, w) + b)
    elif activation == "softmax":
        act = tf.nn.softmax(tf.matmul(input, w)+b)
    elif activation == "relu":
        act = tf.nn.relu(tf.matmul(input, w)+b)
    else:
        act= tf.matmul(input, w)+b
    tf.summary.histogram("weights", w)
    tf.summary.histogram("biases", b)
    tf.summary.histogram("activations", act)
    return act


def mnist_model(activation_function,  hparam,  learning_rate_type):
    tf.reset_default_graph()
    sess = tf.Session()
    # Setup placeholders, input images, labels and learning rate
    x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name="x")
    y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")
    # variable learning rate
    lr = tf.placeholder(tf.float32)

    tf.summary.image('input', x, 3)
    #Reshaping input images
    x_reshape = tf.reshape(x, [-1, 784])
    
    # Number neurons per layer
    layer_neurons1 =200
    layer_neurons2 = 100
    layer_neurons3 = 60
    layer_neurons4 = 30
    layer1 = fc_layer(x_reshape, 784, layer_neurons1, "layer1" , activation_function)
    layer2 = fc_layer(layer1, layer_neurons1, layer_neurons2,  "layer2",  activation_function )
    layer3 =  fc_layer(layer2,  layer_neurons2, layer_neurons3,  "layer3" , activation_function)
    layer4 =  fc_layer(layer3,  layer_neurons3, layer_neurons4,  "layer4",  activation_function )
    logits =  fc_layer(layer4,  layer_neurons4, 10,  "logits",  "none" )
    
    with tf.name_scope("prediction"):
        prediction = tf.nn.softmax(logits)
        tf.summary.histogram("preditions", prediction)
    
    with tf.name_scope("xent"):
        xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=y), name="xent")
        tf.summary.scalar("xent", xent)    
    
    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(lr).minimize(xent)
    
    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)
   
    summ = tf.summary.merge_all()
    
    # Initializing variables
    sess.run(tf.global_variables_initializer())
    
   # TensorBoard - Initializing training and testing writers
    trainer_writer = tf.summary.FileWriter(LOGDIR+ "training_"+hparam)
    test_writer =  tf.summary.FileWriter(LOGDIR+ "testing_"+hparam)
    
    #Display Graph in Tensor Board
    trainer_writer.add_graph(sess.graph)
    
    for i in range(10001):
        batch = mnist.train.next_batch(100)
        if (learning_rate_type == "decay"):
            # learning rate decay
            max_learning_rate = 0.003
            min_learning_rate = 0.0001
            decay_speed = 2000.0 # 0.003-0.0001-2000=>0.9826 done in 5000 iterations
            learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)
        elif (learning_rate_type == "fixed"):
            learning_rate = 0.003
            
        if i % 20 == 0:
            [train_accuracy, s] = sess.run([accuracy, summ], feed_dict={x: batch[0], y: batch[1]})
            trainer_writer.add_summary(s, i)
        if i % 100 == 0:
            [test_accuracy,  s]= sess.run([accuracy, summ], feed_dict={x: mnist.test.images[:1024], y: mnist.test.labels[:1024]})
            test_writer.add_summary(s,i)
        sess.run(train_step, feed_dict={x: batch[0], y: batch[1],  lr: learning_rate})
    
def make_hparam_string(activation_function,  learning_rate_type):
  return "%s_lr_%s" % (activation_function,  learning_rate_type)

def main():
    print('Starting main')
    
    for activation_function in ["sigmoid","relu"]:    
        for learning_rate_type in [ "fixed","decay"]:
            hparam = make_hparam_string(activation_function,  learning_rate_type)
            print('Starting run for %s' % hparam)
            # Actually run with the new settings
            mnist_model(activation_function,  hparam,  learning_rate_type)
    print('Done training!')
    print('Run `tensorboard --logdir=%s` to see the results.' % LOGDIR)
    print('Running on mac? If you want to get rid of the dialogue asking to give '
        'network permissions to TensorBoard, you can provide this flag: '
        '--host=localhost')

if __name__ == '__main__':
  main() 
