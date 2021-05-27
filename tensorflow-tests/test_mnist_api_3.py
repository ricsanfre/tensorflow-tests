import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

LOGDIR = "/home/ricardo/CODE/tensorflow-tests/logs/"
DATADIR = "/home/ricardo/CODE/tensorflow-tests/data/"
EXPORTDIR = "/home/ricardo/CODE/tensorflow-tests/export/"

def cnn_model_fn(
   features, # This is batch_features from input_fn
   labels,   # This is batch_labels from input_fn
   mode,     # An instance of tf.estimator.ModeKeys. The following standard keys are defined:
                            # TRAIN: training mode.
                            # EVAL: evaluation mode.
                            # PREDICT: inference mode.
   params):  # Additional configuration
   
    # Initial Values of return objects
    predictions=None
    loss=None
    train_op=None
    eval_metric_ops=None
    
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
    biasInit = tf.constant_initializer(0.1, dtype=tf.float32)
    conv1 = tf.layers.conv2d(input_layer,  filters=6,  kernel_size=[6, 6], padding="same", activation=tf.nn.relu, bias_initializer=biasInit,  name="conv1")
    conv2 = tf.layers.conv2d(conv1, filters=12, kernel_size=[5, 5], padding="same", strides=2, activation=tf.nn.relu, bias_initializer=biasInit,  name="conv2")
    conv3 = tf.layers.conv2d(conv2, filters=24, kernel_size=[4, 4], padding="same", strides=2, activation=tf.nn.relu, bias_initializer=biasInit,  name="conv3")
    reshape = tf.reshape(conv3, [-1, 24*7*7])
    dense = tf.layers.dense(reshape, 200, activation=tf.nn.relu, bias_initializer=biasInit,  name="dense")
    # to deactivate dropout on the dense layer, set rate=1. The rate is the % of dropped neurons.
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN,  name="dropout")
    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10,  name="logits")
    
    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.cast(tf.argmax(input=logits, axis=1),  tf.uint8),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    
    if mode != tf.estimator.ModeKeys.PREDICT:
        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(labels, 10), logits=logits)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
        
        # Calculate accuracy (for EVAL and TRAINING mode)
        accuracy = tf.metrics.accuracy(labels=labels,  predictions=predictions["classes"],  name="acc_op")
        metrics = {'my_accuracy': accuracy}
        tf.summary.scalar ('accuracy',  accuracy[1])
        # Add evaluation metrics (for EVAL  mode)
        if mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_ops = metrics

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops
    )

def serving_input_receiver_fn():
    feature_spec = {"x": tf.FixedLenSequenceFeature(dtype=tf.int64,  shape=[28, 28, 1],  allow_missing=True)}
    serialized_tf_example = tf.placeholder (dtype= tf.string,  shape=[None])
    received_tensors = {"x": serialized_tf_example}
    features = tf.parse_example (serialized_tf_example,  feature_spec)
    return tf.estimator.export.ServingInputReceiver(features,  received_tensors)

def main():
    
    tf.logging.set_verbosity(tf.logging.INFO)
    print('Starting main')

    print('Loading mnist data')
    
    # Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
    mnist = mnist_data.read_data_sets(train_dir=DATADIR, one_hot=False, reshape=False, validation_size=0)
    train_data= mnist.train.images
    train_labels= np.array (mnist.train.labels,  dtype=np.int32)
    eval_data= mnist.test.images
    eval_labels= np.array (mnist.test.labels,  dtype= np.int32)
    
    # Create the Estimator
    params = tf.contrib.training.HParams(
        learning_rate=0.002,
        train_steps=5000,
        eval_steps=1,
        min_eval_frequency=100
    )
    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir=LOGDIR + "mnist_convnet_model",  params=params)
    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
    # Train the model
    train_input_fn=tf.estimator.inputs.numpy_input_fn (
                    x={"x":train_data}, 
                    y=train_labels, 
                    batch_size=100, 
                    num_epochs=None, 
                    shuffle=True)
                    
    mnist_classifier.train(
                    input_fn=train_input_fn,
                    steps=10000,
                    hooks=[logging_hook])

    # Evaluate model
    eval_input_fn=tf.estimator.inputs.numpy_input_fn (
                    x={"x":eval_data}, 
                    y=eval_labels, 
                    num_epochs=1, 
                    shuffle=False)
                    
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print (eval_results)
    print('Done training!')
    print ('Exporting model')
    
#    mnist_classifier.export_savedmodel(EXPORTDIR, serving_input_receiver_fn)
    
    print('Run `tensorboard --logdir=%s` to see the results.' % LOGDIR)
    print('Running on mac? If you want to get rid of the dialogue asking to give '
        'network permissions to TensorBoard, you can provide this flag: '
        '--host=localhost')

if __name__ == '__main__':
  main() 

