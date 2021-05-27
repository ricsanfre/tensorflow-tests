import tensorflow as tf
import argparse
from stock_data_gdax import StockDataSet

parser= argparse.ArgumentParser()
parser.add_argument('--coin_name',  default='BTC-EUR',  type=str,  help='Coin name to be predicted')
parser.add_argument('--batch_size',  default=100,  type=int,  help='batch size')
parser.add_argument('--prediction_num_steps',  default=30,  type=int, help='Number of steps used in the prediction. Size of the data sequence used for prediction')
parser.add_argument('--train_epochs',  default=100,  type=int,  help='Number of epochs')
parser.add_argument('--hparams',  type=str,  help='Comma separated list of "name=value" pairs')

LOGDIR = "/home/ricardo/CODE/TF_TimeSeries/logs/rnn3_test/"
DATADIR = "/home/ricardo/CODE/TF_TimeSeries/data/"
EXPORTDIR = "/home/ricardo/CODE/TF_TimeSeries/export/"

# Training data single input is a vector containing the data result of applying a non-overlapping sliding window (window_size = num_inputs) durin n-times (num_steps) a number sliding windows of num_imputs data samples.
# Labeled data (the prediction) is a vector of the next not overlapping sliding window in the serie (size = num_inputs)
# If we desire to predict one daily data from the daily data of the previous 30 days, the dimensions should be:
#     num_inputs = 1
#     num_steps = 30

# Tensor shape [BATCH_SIZE. NUM_STEPS, NUM_INPUTS]
# Where BATCH_SIZE is the number of data batches (training + labels). 

def load_data(stock_name, input_size, num_steps):
    stock_dataset = StockDataSet(stock_name, input_size=input_size, num_steps=num_steps,
                                 test_ratio=0.1, normalized=True, close_price_only=True)
    print ("Train data size:", len(stock_dataset.train_X))
    print ("Test data size:", len(stock_dataset.test_X))
    return stock_dataset


def input_fn_train(stock_dataset,   batch_size=10,  num_epochs=1):
    
    #Converting inputs to DataSet
    dataset = tf.data.Dataset.from_tensor_slices((stock_dataset.train_X,  stock_dataset.train_y))
    #Shuffle, repeat and batch the inputs
    dataset= dataset.shuffle(100)
    dataset= dataset.repeat(num_epochs)
    dataset= dataset.batch(batch_size)
    print("DataSet:",  dataset.output_shapes)
    # Return two tensors Train_X and Train_Y data sets.
    return dataset.make_one_shot_iterator().get_next()

# Create prediction model
def rnn_model_fn(
   features, # This is batch_features from input_fn
   labels,   # This is batch_labels from input_fn
   mode,     # An instance of tf.estimator.ModeKeys. The following standard keys are defined:
                            # TRAIN: training mode.
                            # EVAL: evaluation mode.
                            # PREDICT: inference mode.
   params):  # Additional configuration

  # Initializing Hyper Parameters values
  
  learning_rate= params.get(key="learning_rate",  default=0.001)
  nlayers = params.get(key="rnn_layers",  default=[{'cell_size': 128}, {'cell_size':128,  'pkeep': 0.8}])# Stacked LSTM layers configuration
  input_size = params.get(key="input_size",  default=1)
  print ("RNN layers configuration:",  nlayers)
  print ("Parameters: learning rate %s. Input Size (sliding window size): %s", learning_rate,  input_size)
  
  # 0.  Preparing data shape to match rnn fucntion requirements. RNN expects [BATCH_SIZE, num_steps, number_of_inputs]
  
  # rawdata feature input shape [BATCH_SIZE, num_steps, number_of_inputs]
  # labels is [BATCH_SIZE, number_of_inputs]
  # Not need to reshape 'rawdata' 

  X =  tf.cast(features, tf.float32) # [BATCH_SIZE, num_steps, number_of_inputs]
  labels = tf.cast(labels,  tf.float32)
  # 1. configure the RNN
  
  # Building layers of LSTM cells including output dropout
  cells=[0 for i in range(len(nlayers))]
  for i in range(len(nlayers)):
    cell_size= nlayers[i].get('cell_size')
    cell_pkeep = nlayers[i].get('pkeep')
    if cell_pkeep is not None:
       cells[i] = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(cell_size),  output_keep_prob=cell_pkeep) 
#       cells[i] = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(cell_size),  output_keep_prob=cell_pkeep)  
    else:
        cells[i] = tf.nn.rnn_cell.LSTMCell(cell_size)
#        cells[i] = tf.nn.rnn_cell.GRUCell(cell_size)
  # Stacking the LSTM cells layers
  multi_rnn_cell =tf.nn.rnn_cell.MultiRNNCell(cells)
  # Unroll the LSTM cell using the num_steps infered from X (2nd dimension)
  Yr, H = tf.nn.dynamic_rnn(multi_rnn_cell, X, dtype=tf.float32)
    # Yr: [BATCH_SIZE, NUM_STEPS, CELL_SIZE]
    # H: last state of the sequence: [BATCH_SIZE, CELL_SIZE*NLAYERS]
#  H = tf.identity(H,  name="H") # giving it a name
  # Getting last output
  Yr=tf.transpose(Yr,  [1, 0, 2]) # converting Yr output from [BATCH_SIZE, NUM_STEPS, CELL_SIZE] -> [NUM_STEPS, BATCH_SIZE, CELL_SIZE]
  Yt = tf.gather(Yr ,  int(Yr.get_shape()[0]) -1 ,  name='last_lstm_output') # Yt shape [BATCH_SIZE, CELL_SIZE]
  print ("Yt: ",Yt.get_shape() )
  # predictions is result of linear activation of last layer of RNN (Yt)
#  predictions = tf.layers.dense(inputs=Yt, units=input_size, activation=None,  name="YLogits") # [BATCH_SIZE, NUM_IMPUTS/OUTPUTS]
  
  # output is result of linear activation of last layer of RNN
  ws = tf.Variable(tf.truncated_normal([cell_size, input_size]), name="w")
  bias = tf.Variable(tf.constant(0.1, shape=[input_size]), name="b")
  predictions = tf.matmul(Yt, ws) + bias
  
  tf.summary.histogram("pred", predictions)
  # 2. loss function, training/eval ops
  if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
    print("predictions:", predictions.get_shape())
    print("labels:", labels.get_shape())
    loss = tf.losses.mean_squared_error(labels=labels, predictions=predictions)
#    loss =  tf.reduce_sum(tf.square(predictions- labels))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    rme = tf.metrics.root_mean_squared_error(labels=labels,  predictions=predictions,  name="rme_op")
    metrics = {'my_rme': rme}
    avg_sqr_error= rme[1]
    avg_sqr_error = tf.identity(avg_sqr_error,  name="sqr_err")
    tf.summary.scalar ('avgsqr_error',  avg_sqr_error)
    # Add evaluation metrics (for EVAL  mode)
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = metrics
    else:
        eval_metric_ops = None
  else:
    loss = None
    train_op = None
    eval_metric_ops = None

  # 3. Create predictions
  predictions_dict = {"predicted": predictions, 
                                   "avg_sqr_error": avg_sqr_error }
  return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions_dict,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops
  )

def main(argv):
    args = parser.parse_args()
    
    print('Starting main')
    print('Arguments: ',args)
    # Prepare model hyperparameters
    hparams = tf.contrib.training.HParams(
        input_size=1, 
        learning_rate=0.001,
#        rnn_layers = [{'cell_size': 128}, {'cell_size':64}, {'cell_size':32}]
        rnn_layers = [{'cell_size':128,  'pkeep': 0.80}]
#        rnn_layers = [{'cell_size':64}]
    )
    # Overrride hyperparameters values by parsing the command line
    if args.hparams is not None:
        hparams.parse(args.hparams)
    
    print ("Model parameters:",  hparams)
    
    # Load data
    stock_dataset = load_data(args.coin_name,
                              input_size=hparams.get(key="input_size",  default=1),
                              num_steps=args.prediction_num_steps)
    # Create running config object
    # Saving for each step.
    config = tf.estimator.RunConfig (save_summary_steps=10)
    # Create the Estimator
    rnn_predictor = tf.estimator.Estimator(model_fn=rnn_model_fn, model_dir=LOGDIR,  config=config,  params=hparams)
    # Set up logging for predictions
    tensors_to_log = {"avg_sqr_error": "sqr_err"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)
    # Train the model
    rnn_predictor.train(
                    input_fn=lambda: input_fn_train(stock_dataset, batch_size=args.batch_size,  num_epochs=args.train_epochs),
                    hooks=[logging_hook])
    print('Done training!')
    # Evaluate model
#    eval_results = rnn_predictor.evaluate(input_fn=input_fn_eval)
#    print (eval_results)
    
    print ('Exporting model')
    
#    rnn_predictor.export_savedmodel(EXPORTDIR, serving_input_receiver_fn)
    
    print('Run `tensorboard --logdir=%s` to see the results.' % LOGDIR)
    print('Running on mac? If you want to get rid of the dialogue asking to give '
        'network permissions to TensorBoard, you can provide this flag: '
        '--host=localhost')


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
