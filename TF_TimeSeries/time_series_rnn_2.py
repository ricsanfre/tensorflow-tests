import tensorflow as tf
import argparse

parser= argparse.ArgumentParser()
parser.add_argument('--batch_size',  default=100,  type=int,  help='batch size')
parser.add_argument('--train_steps',  default=1000,  type=int,  help='number of traininig steps')
parser.add_argument('--hparams',  type=str,  help='Comma separated list of "name=value" pairs')

LOGDIR = "/home/ricardo/CODE/TF_TimeSeries/logs/"
DATADIR = "/home/ricardo/CODE/TF_TimeSeries/data/"
EXPORTDIR = "/home/ricardo/CODE/TF_TimeSeries/export/"

CSV_LEN = 5*24*7 # each serie represents the traffic within five weeks (hourly data).
N_OUTPUTS = 24*7  # in each sequence, 1-672 are features, and 673-840 (168 samples) are labels
SEQ_LEN = CSV_LEN - N_OUTPUTS
RECORD_DEFAULTS = [[0.0] for x in range(0, CSV_LEN)]
BATCH_SIZE = 20
TIMESERIES_COL = 'rawdata'

# read data and convert to needed format
def read_dataset(filename, mode=tf.contrib.learn.ModeKeys.TRAIN):    
    num_epochs = 100 if mode == tf.contrib.learn.ModeKeys.TRAIN else 1
    # could be a path to one file or a file pattern.
    input_file_names = tf.train.match_filenames_once(filename)
    filename_queue = tf.train.string_input_producer(
        input_file_names, num_epochs=num_epochs, shuffle=True)
    reader = tf.TextLineReader()
    key, value = reader.read_up_to(filename_queue, num_records=BATCH_SIZE)
    value_column = tf.expand_dims(value, -1)
#    print ('readcsv={}'.format(value_column))
    # all_data is a list of tensors
    all_data = tf.decode_csv(value_column, record_defaults=RECORD_DEFAULTS)  
    inputs = all_data[:len(all_data)-N_OUTPUTS]  # first few values
    label = all_data[len(all_data)-N_OUTPUTS : ] # last few values

    # from list of tensors to tensor with one more dimension
    inputs = tf.concat(inputs, axis=1)
    inputs= tf.reshape(inputs, [-1, SEQ_LEN, 1])
    label = tf.concat(label, axis=1)
    label= tf.reshape(label, [-1, N_OUTPUTS, 1])
#    print ('inputs={}'.format(inputs))
#    print ('label={}'.format(label))
    return {TIMESERIES_COL: inputs}, label   # dict of features, label

def input_fn_train():
  return read_dataset('train.csv', mode=tf.contrib.learn.ModeKeys.TRAIN)

def input_fn_eval():
  return read_dataset('valid.csv', mode=tf.contrib.learn.ModeKeys.EVAL)

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
  
  LEARNING_RATE= params.get(key="learning_rate",  default=0.001)
  nlayers = params.get(key="rnn_layers",  default=[{'cell_size': 512}, {'cell_size':512,  'pkeep': 0.8}])# Stacked LSTM layers configuration
  print ("NLAYERS:",  nlayers)
  
  print ("LEARNING_RATE", LEARNING_RATE)
  
  # 0.  Preparing data shape to match rnn fucntion requirements. RNN expects [BATCH_SIZE, SEQ_LEN, number_of_inputs]
  
  # rawdata feature input shape [BATCH_SIZE, SEQ_LEN,1]
  # labels is [BATCH_SIZE, N_OUTPUTS,1]
  # Not need to reshape 'rawdata' 

  X =  features[TIMESERIES_COL]
  
  # 1. configure the RNN
  
  # Building layers of LSTM cells including output dropout
  cells=[0 for i in range(len(nlayers))]
  for i in range(len(nlayers)):
    cell_size= nlayers[i].get('cell_size')
    cell_pkeep = nlayers[i].get('pkeep')
    if cell_pkeep is not None:
       cells[i] = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(cell_size, forget_bias=1.0),  output_keep_prob=cell_pkeep) 
    else:
        cells[i] = tf.nn.rnn_cell.BasicLSTMCell(cell_size, forget_bias=1.0)
  # Stacking the LSTM cells layers
  multi_rnn_cell =tf.nn.rnn_cell.MultiRNNCell(cells)
  Yr, H = tf.nn.dynamic_rnn(multi_rnn_cell, X, dtype=tf.float32)
    # Yr: [BATCH_SIZE, SEQ_LEN, CELL_SIZE]
    # H: last state of the sequence: [BATCH_SIZE, CELL_SIZE*NLAYERS]
  H = tf.identity(H,  name="H") # giving it a name
  # Getting last output
  Yt = Yr[:, -1, :]
  # predictions is result of linear activation of last layer of RNN (Yt)
  Ylogits = tf.layers.dense(inputs=Yt, units=N_OUTPUTS, activation=None,  name="YLogits") # [BATCH_SIZE, N_OUTPUTS]
  predictions = tf.reshape (Ylogits, [-1, N_OUTPUTS, 1])
  # 2. loss function, training/eval ops
  if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
    loss = tf.losses.mean_squared_error(labels, predictions)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
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
    print(args)
    # Prepare model hyperparameters
    hparams = tf.contrib.training.HParams(
        learning_rate=0.002,
        rnn_layers = [{'cell_size': 512}, {'cell_size':512,  'pkeep': 0.8}]
    )
    
    # Overrride hyperparameters values by parsing the command line
    if args.hparams is not None:
        hparams.parse(args.hparams)
    
    print (hparams)
    
    # Create the Estimator
    rnn_predictor = tf.estimator.Estimator(model_fn=rnn_model_fn, model_dir=LOGDIR,  params=hparams)
    # Set up logging for predictions
    tensors_to_log = {"avg_sqr_error": "sqr_err"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)
    # Train the model
    rnn_predictor.train(
                    input_fn=input_fn_train,
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
