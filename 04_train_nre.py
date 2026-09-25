"""04_train_nre.py: train the tensionnet NREs for one data set combination and evaluate them.

*** HPC-GRADE *** At the paper's size each NRE trains on 540 000 simulation pairs
on 4 GPUs for up to 48 hours (see pbs/train_nre.pbs).

The data set combination is set with --first/--second, which picks the
simulation directories to pair up and the output directory
    outputs/nre/{first}_{second}/

Steps:
    1. Pair up the FIRST and SECOND simulations (made by 01a/01b) chunk by chunk,
       and copy them to fast local disk (node SSD on the HPC).
    2. Pre-make all batches for training and validation.
    3. For each of the --runs NREs:
         a. build a fresh model (random initial weights) and train it, restarting
            if it has not learnt anything after 2 epochs, with snapshots every 10
            epochs so an interrupted job can resume;
         b. predict the in concordance log R distribution;
         c. check that the distribution is physical. We have prior knowledge of
            the physical shape of the distribution from validation. If its peak
            density is < 0.13 (too broad - catastrophic initial weights), > 0.25
            (too narrow - mode collapse) or its median is negative, the weights
            are discarded and a new NRE is trained in its place. This repeats
            until --runs NREs have passed.
    4. Evaluate every NRE: training/validation loss (used to weight the
       ensemble in 05_ensemble.py), in concordance distribution plots, and the
       tension T and concordance C for the observed log R.

The NRE hyperparameters (--nside-out, --filters, --layers, --neurons, --dropout,
--activation, --learning-rate, --decay-steps, --decay-rate) default to the values
chosen from the Optuna searches (03a_optuna_search1.py, 03b_optuna_search2.py)
and used in the paper. If running from scratch, review the Optuna outputs and set
them here.

Usage:
    python 04_train_nre.py --first planck --second racs
    python 04_train_nre.py --first planck --second racs --layers 4 --neurons 256 --dropout 0.1
"""

# Import packages
print('Importing packages...')
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 0 = All messages, 1 = Filter out INFO messages, 2 = Filter out WARNING, 3 = Filter out all messages (Fatal errors remain)
import shutil
import glob
import re
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import healpy as hp
import nnhealpix.layers
from scipy.stats import ecdf
from scipy.stats import norm
from tensionnet.utils import calcualte_stats
import matplotlib as mpl
import config
from utils import check_concordance_distribution
mpl.rcParams['text.usetex'] = False # Fix latex errors on cluster

# ---------------------------------------------------------------------------
# Command line arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--first', required=True, choices=['planck', 'racs', 'nvss'], help='FIRST data set')
parser.add_argument('--second', required=True, choices=['racs', 'nvss', 'catwise', 'catsim'], help='SECOND data set')
parser.add_argument('--runs', type=int, default=config.N_RUNS, help='number of NREs that must pass the checks (paper: 5)')
parser.add_argument('--nsamples', type=int, default=config.N_SAMPLES, help='must match 01a/01b (paper: 900000)')
parser.add_argument('--nchunks', type=int, default=config.N_CHUNKS, help='must match 01a/01b (paper: 90)')
# NRE hyperparameters. The defaults are the ones chosen from the Optuna searches (03a, 03b) and
# used in the paper; review the Optuna outputs and change them here if running from scratch.
parser.add_argument('--nside-out', type=int, default=4, help='Nside the convolution towers compress to (paper: 4, i.e. 192 pixels)')
parser.add_argument('--filters', type=int, default=1, help='convolution filters (paper: 1)')
parser.add_argument('--layers', type=int, default=5, help='dense layers (paper: 5)')
parser.add_argument('--neurons', type=int, default=hp.nside2npix(4)*2, help='neurons per dense layer (paper: 384)')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate after each dense layer (paper: 0.2)')
parser.add_argument('--activation', default='relu', choices=['relu', 'leakyrelu', 'gelu', 'elu'], help='dense layer activation (paper: relu)')
parser.add_argument('--learning-rate', type=float, default=1e-3, help='initial learning rate (paper: 1e-3)')
parser.add_argument('--decay-steps', type=int, default=1000, help='learning rate decay steps (paper: 1000)')
parser.add_argument('--decay-rate', type=float, default=0.9, help='learning rate decay rate (paper: 0.9)')
# Training settings
parser.add_argument('--epochs', type=int, default=1000, help='maximum number of epochs (paper: 1000)')
parser.add_argument('--patience', type=int, default=50, help='early stopping patience (paper: 50)')
parser.add_argument('--batch-size', type=int, default=None, help='override the batch size chosen from the GPUs')
parser.add_argument('--min-peak', type=float, default=config.MIN_PEAK_DENSITY, help='discard NREs whose in concordance PDF peaks below this')
parser.add_argument('--max-peak', type=float, default=config.MAX_PEAK_DENSITY, help='discard NREs whose in concordance PDF peaks above this')
parser.add_argument('--min-median', type=float, default=0.0, help='discard NREs whose in concordance distribution has a median below this')
parser.add_argument('--max-discards', type=int, default=20, help='give up after discarding this many NREs for a single run')
args = parser.parse_args()
if (args.first, args.second) not in config.COMBINATIONS:
    parser.error(f'{args.first}-{args.second} is not one of the combinations in the paper: {config.COMBINATIONS}')

# ---------------------------------------------------------------------------
# Tensorflow GPU configuration
# ---------------------------------------------------------------------------
print('Configuring TensorFlow...')
physical_devices = tf.config.list_physical_devices()
print("Available physical devices:", physical_devices) # Print TensorFlow device info
ngpus = len(tf.config.list_physical_devices('GPU'))
if ngpus == 1:
    print(f"{ngpus} GPU available.")
    strategy = tf.distribute.get_strategy()
    batch_size = 1000
    if 'h100' in tf.config.list_physical_devices('GPU')[0].name.lower():
        print('Optimising for H100.')
        tf.config.threading.set_intra_op_parallelism_threads(1) # Set maximum number of CPU cores for TensorFlow
        tf.config.threading.set_inter_op_parallelism_threads(int(os.environ["OMP_NUM_THREADS"])) # Set maximum number of CPU cores for TensorFlow
        batch_size = 3000
elif ngpus > 1:
    print(f"{ngpus} GPUs available.")
    strategy = tf.distribute.MirroredStrategy()
    batch_size = 1500 * ngpus # 1500 works, 1750 works
else:
    print("No GPU available - falling back to CPU.")
    strategy = tf.distribute.get_strategy()
    batch_size = 1000
if args.batch_size is not None:
    batch_size = args.batch_size
tf.config.set_soft_device_placement(True) # allow TensorFlow to fall back to CPU if GPU is not available
if os.uname().sysname == 'Darwin':
    print('Running on macOS - disabling XLA')
    os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
    os.environ["TF_DISABLE_MLIR_BRIDGE"] = "1"
else:
    print('Running on Linux - enabling XLA')
    tf.config.optimizer.set_jit(True) # enable XLA (Accelerated Linear Algebra) for performance improvements

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
# Simulation settings
nChunks = args.nchunks
nSamples = args.nsamples
dtype = np.float32 # np.float32
FIRST_dataset = args.first
SECOND_dataset = args.second

# Network settings (defaults from the Optuna searches, see the arguments above)
nside_in = 64
nside_out = args.nside_out
filters = args.filters
layers = args.layers
neurons = args.neurons
dropout_rate = args.dropout
activation_function = args.activation
runs = args.runs

# Training and evaluation settings
minimum_accuracy_after_2_epochs = 0.55 # This is used to abort training and restart if it is not going well, for whatever reason (I have observed this occuring)
evaluate = True # Choose to evaluate the networks or not
force_evaluation = True

# Set directories
BASE_DIR = config.nre_dir(FIRST_dataset, SECOND_dataset)
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)
SIMULATION_DIR = config.SIMULATION_DIR
FIRST_dataset_dir_mixed = SIMULATION_DIR + FIRST_dataset + '_mixed_first'
SECOND_dataset_dir_mixed = SIMULATION_DIR + SECOND_dataset + '_mixed_second'
FIRST_dataset_dir_concordant = SIMULATION_DIR + FIRST_dataset + '_concordant'
SECOND_dataset_dir_concordant = SIMULATION_DIR + SECOND_dataset + '_concordant'

# Get the R-statistic of the real datasets
R_DIR = config.R_DIR
R = np.load(f'{R_DIR}/R_{FIRST_dataset}_{SECOND_dataset}.npy')
errorR = np.load(f'{R_DIR}/errorR_{FIRST_dataset}_{SECOND_dataset}.npy')/2 # convert to 1 sigma error

# Check and create output directories
if not os.path.exists(BASE_DIR + 'stats/'):
    os.makedirs(BASE_DIR + 'stats/')
if not os.path.exists(BASE_DIR + 'evaluations/'):
    os.makedirs(BASE_DIR + 'evaluations/')
if not os.path.exists(BASE_DIR + 'predictions/'):
    os.makedirs(BASE_DIR + 'predictions/')
SNAPSHOT_DIR = BASE_DIR + 'snapshots/'
if not os.path.exists(SNAPSHOT_DIR):
    os.makedirs(SNAPSHOT_DIR)
STATS_FILE = BASE_DIR + 'stats/' + f'{runs}runs_stats.npy'

# Calculate chunk sizes as in the simulation scripts
chunk_size = nSamples // nChunks
concordant_chunks = int(nChunks * 0.1)
training_chunks = int(nChunks * 0.6)
test_chunks = nChunks - concordant_chunks - training_chunks
mixed_chunks = training_chunks + test_chunks

# ---------------------------------------------------------------------------
# 1. Check the simulations exist, then pair them up on fast local disk
# ---------------------------------------------------------------------------
files = [FIRST_dataset_dir_concordant + f'/chunk{idx+1}.npy' for idx in range(concordant_chunks)]
files += [FIRST_dataset_dir_mixed + f'/chunk{idx+1}.npy' for idx in range(mixed_chunks)]
files += [SECOND_dataset_dir_concordant + f'/chunk{idx+1}.npy' for idx in range(concordant_chunks)]
files += [SECOND_dataset_dir_mixed + f'/chunk{idx+1}.npy' for idx in range(mixed_chunks)]
files += [SIMULATION_DIR + 'labels.npy']
missing = [f for f in files if not os.path.exists(f)]
if missing:
    raise FileNotFoundError(f'{len(missing)} simulation files are missing (e.g. {missing[0]}). '
                            f'Run 01a_make_simulations.py (and 01b_make_catsim_simulations.py for CatSIM) first.')
print('All simulations are present.')

# Copy simulation files to jobfs for faster access and concatenate them:
# each row is [FIRST sky (49152 pixels), SECOND sky (49152 pixels)]
print('Copying simulation files to jobfs for faster access and concatenating them...')
JOBFS_DIR = config.jobfs_dir()
if not os.path.exists(JOBFS_DIR):
    os.makedirs(JOBFS_DIR)
if not os.path.exists(JOBFS_DIR + 'training/'):
    os.makedirs(JOBFS_DIR + 'training/')
if not os.path.exists(JOBFS_DIR + 'validation/'):
    os.makedirs(JOBFS_DIR + 'validation/')
if not os.path.exists(JOBFS_DIR + 'concordant/'):
    os.makedirs(JOBFS_DIR + 'concordant/')
for i in tqdm(range(mixed_chunks)):
    first_data = np.array(np.load(FIRST_dataset_dir_mixed + f'/chunk{i+1}.npy', mmap_mode='r'))
    second_data = np.array(np.load(SECOND_dataset_dir_mixed + f'/chunk{i+1}.npy', mmap_mode='r'))
    if i < training_chunks:
        np.save(JOBFS_DIR + f'training/chunk{i+1}.npy', np.concatenate((first_data, second_data), axis=-1))
    else:
        np.save(JOBFS_DIR + f'validation/chunk{i+1 - training_chunks}.npy', np.concatenate((first_data, second_data), axis=-1))
for i in tqdm(range(concordant_chunks)):
    first_data = np.array(np.load(FIRST_dataset_dir_concordant + f'/chunk{i+1}.npy', mmap_mode='r'))
    second_data = np.array(np.load(SECOND_dataset_dir_concordant + f'/chunk{i+1}.npy', mmap_mode='r'))
    np.save(JOBFS_DIR + f'concordant/chunk{i+1}.npy', np.concatenate((first_data, second_data), axis=-1))
del first_data, second_data
shutil.copy(SIMULATION_DIR + 'labels.npy', JOBFS_DIR + 'labels.npy')

# ---------------------------------------------------------------------------
# 2. Data streaming with a data generator using preconverted+prereshaped chunks and less function calls
# ---------------------------------------------------------------------------
print('Setting up data streaming...')
# Get the list of training and validation chunk files
train_data_files = [JOBFS_DIR + f'training/chunk{i+1}.npy' for i in range(training_chunks)]
val_data_files = [JOBFS_DIR + f'validation/chunk{i+1}.npy' for i in range(test_chunks)]
# Load all data into memory using memory mapping
print('Loading data into memory with memory mapping...')
train_data = [np.load(f, mmap_mode='r') for f in train_data_files]
val_data = [np.load(f, mmap_mode='r') for f in val_data_files]
labels = np.load(JOBFS_DIR + 'labels.npy', mmap_mode='r')
train_labels = labels[:training_chunks * chunk_size]
train_labels = np.split(train_labels, training_chunks)
val_labels = labels[training_chunks * chunk_size:]
val_labels = np.split(val_labels, test_chunks)
# Calculate total number of samples for steps_per_epoch
train_samples = sum(data.shape[0] for data in train_data)
val_samples = sum(data.shape[0] for data in val_data)
steps_per_epoch = int(np.ceil(train_samples / batch_size))
validation_steps = int(np.ceil(val_samples / batch_size))

# Precompute and store reshaped views (no copy) for all data arrays
print('Precomputing reshaped views for data arrays...')
def get_reshaped_views(data_list):
    # Returns a list of views with shape (N, features, 1)
    return [np.expand_dims(data, axis=-1) for data in data_list]

train_data = get_reshaped_views(train_data)
val_data = get_reshaped_views(val_data)

# Pre-make all batches from pre-reshaped arrays without copying data
def premake_batches_inplace(data_views, labels_views, batch_size):
    batches_data = []
    batches_labels = []
    for idx in range(len(data_views)):
        data = data_views[idx]
        labels = labels_views[idx]
        num_samples = data.shape[0]
        num_batches = num_samples // batch_size
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            # Use views (slices) to avoid copying
            batches_data.append(data[start:end])
            batches_labels.append(labels[start:end])
        # Delete reference to data/labels after batching to free RAM
        data_views[idx] = None
        labels_views[idx] = None
    return batches_data, batches_labels

print('Pre-making all batches from data arrays (in-place, no copies)...')
train_batches_data, train_batches_labels = premake_batches_inplace(train_data, train_labels, batch_size)
del train_data, train_labels # Remove original lists to free RAM
val_batches_data, val_batches_labels = premake_batches_inplace(val_data, val_labels, batch_size)
del val_data, val_labels # Remove original lists to free RAM
if len(train_batches_data) == 0 or len(val_batches_data) == 0:
    raise ValueError(f'batch_size={batch_size} is larger than a chunk ({chunk_size} simulations); use a smaller --batch-size.')
steps_per_epoch = len(train_batches_data) # only whole batches are made, so count those
validation_steps = len(val_batches_data)

# Generator yielding pre-made batches (no processing during training)
def static_batch_generator(batches_data, batches_labels):
    while True:
        for data, labels in zip(batches_data, batches_labels):
            yield data, labels

def finite_batch_generator(batches_data, batches_labels):
    for data, labels in zip(batches_data, batches_labels):
        yield data, labels

def tf_static_batch_generator(batches_data, batches_labels, batch_generator_func):
    num_features = batches_data[0].shape[1]
    output_signature = (
        tf.TensorSpec(shape=(None, num_features, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
    )
    gen = lambda: batch_generator_func(batches_data, batches_labels)
    return tf.data.Dataset.from_generator(gen, output_signature=output_signature).prefetch(tf.data.AUTOTUNE)

# Build generators using pre-made batches
print('Building static data generators...')
train_gen = tf_static_batch_generator(train_batches_data, train_batches_labels, static_batch_generator)
val_gen = tf_static_batch_generator(val_batches_data, val_batches_labels, static_batch_generator)

# ---------------------------------------------------------------------------
# The tensionnet: two convolution towers (Nside 64 -> 4) feeding a dense network
# ---------------------------------------------------------------------------
# Define custom binary cross-entropy loss function (the network outputs log R; sigmoid(log R) = p(matched))
def binary_crossentropy(y_true, logR):
    y_pred = tf.sigmoid(logR) # sigmoid activation
    epsilon = 1e-7  # To avoid log(0)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)  # Clip predictions to avoid log(0)
    loss = -tf.reduce_mean(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
    return loss

def build_model():
    print('Building the model...')
    with strategy.scope():
        # Convolution tower: ConvNeighbours then masked average pooling, halving nside each time
        def NBB(x, nside_in, nside_out, filters=1):
            nside = nside_in
            while nside > nside_out:
                x = nnhealpix.layers.ConvNeighbours(nside, filters=filters, kernel_size=9)(x)
                nside = nside // 2
                x = nnhealpix.layers.MaskedAveragePooling(nside*2, nside, mask_value=0.0)(x)
            return x

        # Set parameters
        npix = hp.nside2npix(nside_in)
        # Define input
        inputs = tf.keras.layers.Input(shape=(2 * npix, 1))
        # Split the input into two
        x1 = tf.keras.layers.Lambda(lambda x: x[:, :npix, :])(inputs)  # first half
        x2 = tf.keras.layers.Lambda(lambda x: x[:, npix:, :])(inputs)  # second half
        # Normalize x1 and x2 inputs
        x1 = tf.keras.layers.BatchNormalization(axis=1)(x1) # do not do axis=-1
        x2 = tf.keras.layers.BatchNormalization(axis=1)(x2) # do not do axis=-1
        # Pass each input through its own convolutional tower
        x1 = NBB(x1, nside_in, nside_out, filters)
        x2 = NBB(x2, nside_in, nside_out, filters)
        # Flatten the outputs
        x1 = tf.keras.layers.Flatten()(x1)
        x2 = tf.keras.layers.Flatten()(x2)
        # Concatenate both datasets
        x = tf.keras.layers.Concatenate()([x1, x2])
        # Fully connected layers
        for i in range(layers):
            x = tf.keras.layers.Dense(neurons)(x)
            if activation_function == 'leakyrelu':
                x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
            else:
                x = tf.keras.layers.Activation(activation_function)(x)
            x = tf.keras.layers.Dropout(dropout_rate)(x)
        # Final output layer
        x = tf.keras.layers.Dense(1)(x)
        out = tf.keras.layers.Activation('linear')(x) # linear, to output logR

        # Variable learning rate
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=args.learning_rate,
            decay_steps=args.decay_steps,
            decay_rate=args.decay_rate,
            staircase=True
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        # Compile model
        model = tf.keras.models.Model(inputs=inputs, outputs=out)
        model.compile(loss=binary_crossentropy, optimizer=optimizer, metrics=['accuracy'])

    model.summary()
    print('Model built successfully.')
    return model

# Callback that aborts training if it has not learnt anything after a few epochs
# We found that occasionally the training would not progress, for no apparent reason, so added in this check
class AbortAndRestartCallback(tf.keras.callbacks.Callback):
    def __init__(self, min_acc=0.6, check_epoch=3):
        super().__init__()
        self.min_acc = min_acc
        self.check_epoch = check_epoch
        self.should_abort = False

    def on_epoch_end(self, epoch, logs=None):
        if epoch + 1 == self.check_epoch:
            acc = logs.get('accuracy')
            if acc is not None and acc < self.min_acc:
                print(f"\nTraining accuracy {acc:.3f} below {self.min_acc} after {self.check_epoch} epochs. Aborting and restarting training.")
                self.model.stop_training = True
                self.should_abort = True

# Callback to save snapshots every N epochs (10), so an interrupted job can resume
class SaveSnapshotCallback(tf.keras.callbacks.Callback):
    def __init__(self, run, save_every=10):
        super().__init__()
        self.run = run
        self.save_every = save_every

    def on_epoch_end(self, epoch, logs=None):
        epoch_num = epoch + 1
        if epoch_num % self.save_every == 0:
            fname = os.path.join(SNAPSHOT_DIR, f"run{self.run+1}_snapshot_epoch{epoch_num}.keras")
            try:
                # save full model (architecture + weights + optimizer state)
                self.model.save(fname)
                print(f"Saved training snapshot to {fname}")
            except Exception as e:
                print(f"Failed to save snapshot at epoch {epoch_num}: {e}")

# Predict (or load) the in concordance log R distribution of one run
def predict_concordance(model, run):
    r = []
    for k in range(concordant_chunks):
        if not os.path.exists(BASE_DIR + 'predictions/' + f'run{run+1}_concordance{k+1}.npy'):
            print('Predicting concordance distribution for', f'run {run+1}', f'chunk {k+1}')
            prediction = model.predict(np.load(JOBFS_DIR + f'concordant/chunk{k+1}.npy', mmap_mode='r'), batch_size=batch_size)
            np.save(BASE_DIR + 'predictions/' + f'run{run+1}_concordance{k+1}.npy', prediction)
        else:
            print('Loading concordance distribution for', f'run {run+1}', f'chunk {k+1}')
            prediction = np.load(BASE_DIR + 'predictions/' + f'run{run+1}_concordance{k+1}.npy', mmap_mode='r')
        r.append(prediction)
    return np.concatenate(r, axis=0).flatten()

# Delete everything saved for one run, so it can be trained again from scratch
def discard_run(run, reason):
    print(f'Discarding run {run+1}: {reason}')
    with open(BASE_DIR + 'discarded_runs.txt', 'a') as f:
        f.write(f'run{run+1}: {reason}\n')
    for fname in ([BASE_DIR + f'run{run+1}_weights.keras', BASE_DIR + f'run{run+1}_bad_training.txt']
                  + glob.glob(BASE_DIR + 'predictions/' + f'run{run+1}_concordance*.npy')
                  + glob.glob(BASE_DIR + f'run{run+1}_loss_*.npy')
                  + glob.glob(BASE_DIR + f'run{run+1}_accuracy_*.npy')):
        if os.path.exists(fname):
            os.remove(fname)

# ---------------------------------------------------------------------------
# 3. Train (or load) each NRE, and replace any whose in concordance distribution is unphysical
# ---------------------------------------------------------------------------
for run in range(runs):
    discards = 0
    while True:

        # 3a. Train or load the model
        if not os.path.exists(BASE_DIR + f'run{run+1}_weights.keras'):
            print('Training model weights...')
            if 'model' in locals():
                del model
            model = build_model() # fresh random initial weights for every run

            # Setup early stopping
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=args.patience,
                restore_best_weights=True)

            # Training loop that will abort if training is not going well (does not meet target accuracy after 2 epochs)
            max_retries = 50
            for attempt in range(max_retries):
                abort_callback = AbortAndRestartCallback(min_acc=minimum_accuracy_after_2_epochs, check_epoch=2)

                # Try to find latest snapshot to resume from
                pattern = os.path.join(SNAPSHOT_DIR, f"run{run+1}_snapshot_epoch*.keras")
                snaps = glob.glob(pattern)
                initial_epoch = 0
                if snaps:
                    # extract epoch numbers and pick latest
                    epochs = []
                    for s in snaps:
                        m = re.search(r"epoch(\d+)\.keras$", s)
                        if m:
                            epochs.append((int(m.group(1)), s))
                    if epochs:
                        last_epoch, last_snap = max(epochs, key=lambda x: x[0])
                        try:
                            print(f"Loading snapshot {last_snap} to resume from epoch {last_epoch}...")
                            model = tf.keras.models.load_model(last_snap, custom_objects={'binary_crossentropy': binary_crossentropy}, safe_mode=False)
                            initial_epoch = last_epoch
                        except Exception as e:
                            print("Failed to load snapshot, starting from scratch:", e)
                            initial_epoch = 0

                snapshot_cb = SaveSnapshotCallback(run, save_every=10)

                # Fit, resuming from initial_epoch if a snapshot was loaded
                history = model.fit(
                    x=train_gen,
                    steps_per_epoch=steps_per_epoch,
                    epochs=args.epochs,
                    validation_data=val_gen,
                    validation_steps=validation_steps,
                    callbacks=[early_stopping, abort_callback, snapshot_cb],
                    verbose=1,
                    initial_epoch=initial_epoch
                )

                if not abort_callback.should_abort:
                    break

            if abort_callback.should_abort:
                print(f"Training repeatedly failed to reach minimum accuracy after {max_retries} attempts.")
                print("Training regardless...")
                # Save a zero-byte file as a note
                with open(BASE_DIR + f'run{run+1}_bad_training.txt', 'w') as f:
                    pass

                history = model.fit(
                    x=train_gen,
                    steps_per_epoch=steps_per_epoch,
                    epochs=args.epochs,
                    validation_data=val_gen,
                    validation_steps=validation_steps,
                    callbacks=[early_stopping],
                    verbose=1
                )

            # Save the model to BASE_DIR
            print('Saving model weights...')
            model.save(BASE_DIR + f'run{run+1}_weights.keras')

        else:
            print(f"Model for run {run+1} already exists. Skipping training.")
            if 'model' not in locals():
                model = build_model()
            model.load_weights(BASE_DIR + f'run{run+1}_weights.keras')

        # If model weights exist, delete all snapshots for this run
        if os.path.exists(BASE_DIR + f'run{run+1}_weights.keras'):
            pattern = os.path.join(SNAPSHOT_DIR, f"run{run+1}_snapshot_epoch*.keras")
            snaps = glob.glob(pattern)
            for snap in snaps:
                try:
                    os.remove(snap)
                    print(f"Deleted snapshot: {snap}")
                except Exception as e:
                    print(f"Failed to delete snapshot {snap}: {e}")

        # 3b. Predict the in concordance log R distribution
        r = predict_concordance(model, run)

        # 3c. Check the distribution is physical; if not, discard these weights and train a new NRE
        passed, peak_density, median, reason = check_concordance_distribution(r, args.min_peak, args.max_peak, args.min_median)
        print(f'Run {run+1} in concordance distribution: peak density = {peak_density:.3f}, median = {median:.3f} -> {reason}')
        if passed:
            break
        discard_run(run, reason)
        discards += 1
        if discards >= args.max_discards:
            raise RuntimeError(f'Run {run+1}: {discards} NREs in a row had unphysical in concordance distributions. '
                               f'Check the simulations and training settings.')

    tf.keras.backend.clear_session() # Clear session to reset memory use

# Delete snapshot directory if empty
if os.path.exists(SNAPSHOT_DIR) and not os.listdir(SNAPSHOT_DIR):
    try:
        os.rmdir(SNAPSHOT_DIR)
        print(f"Deleted empty snapshot directory: {SNAPSHOT_DIR}")
    except Exception as e:
        print(f"Failed to delete snapshot directory {SNAPSHOT_DIR}: {e}")

# ---------------------------------------------------------------------------
# 4. Evaluate the models
# ---------------------------------------------------------------------------
# Check if stats have been computed
evaluation_needed = False
if not os.path.exists(STATS_FILE):
    evaluation_needed = True
if force_evaluation:
    evaluation_needed = True

if evaluate and evaluation_needed:
    print('Starting evaluation...')
    if 'model' in locals():
        del model
    model = build_model()

    # Compute accuracy and loss on the training and validation data (the validation loss weights the ensemble)
    for run in range(runs):
        model.load_weights(BASE_DIR + f'run{run+1}_weights.keras')

        # Set generators to finite for evaluation
        if 'train_gen' in locals():
            del train_gen
        if 'val_gen' in locals():
            del val_gen

        if not os.path.exists(BASE_DIR + f'run{run+1}_loss_train.npy'):
            print(f'Evaluating run {run+1} on training data...')
            train_gen = tf_static_batch_generator(train_batches_data, train_batches_labels, finite_batch_generator)
            loss_train, accuracy_train = model.evaluate(
                x=train_gen
            )
            np.save(BASE_DIR + f'run{run+1}_loss_train.npy', loss_train)
            np.save(BASE_DIR + f'run{run+1}_accuracy_train.npy', accuracy_train)

        if not os.path.exists(BASE_DIR + f'run{run+1}_loss_test.npy'):
            print(f'Evaluating run {run+1} on validation data...')
            if 'train_gen' in locals():
                del train_gen
            val_gen = tf_static_batch_generator(val_batches_data, val_batches_labels, finite_batch_generator)
            loss_test, accuracy_test = model.evaluate(
                x=val_gen
            )
            np.save(BASE_DIR + f'run{run+1}_loss_test.npy', loss_test)
            np.save(BASE_DIR + f'run{run+1}_accuracy_test.npy', accuracy_test)

    # Clear up space before loading concordant data
    print('Clearing up space before loading concordant data...')
    to_delete = [
        'train_data_files', 'val_data_files',
        'labels',
        'train_samples', 'val_samples', 'steps_per_epoch', 'validation_steps',
        'train_batches_data', 'train_batches_labels',
        'val_batches_data', 'val_batches_labels',
        'train_gen', 'val_gen',
        'get_reshaped_views', 'premake_batches_inplace',
        'static_batch_generator', 'tf_static_batch_generator'
    ]
    for var in to_delete:
        if var in globals():
            del globals()[var]

    # In concordance distribution, and T and C, for every run
    sigmaD, sigmaA = [], []
    widths = []
    for run in range(runs):
        model.load_weights(BASE_DIR + f'run{run+1}_weights.keras')
        r = predict_concordance(model, run)
        mask = np.isfinite(r)

        print('Plotting distribution for', f'run {run+1}')
        # Plot in concordance distribution and cumulative distribution
        fig, axes = plt.subplots(1, 2, figsize=(4.5, 3))
        axes[0].hist(r[mask], bins=25, density=True)
        axes[0].set_xlabel(r'$\log R$')
        axes[0].set_ylabel('Density')
        axes[0].axvline(R, ls='--', c='r')
        axes[0].axvspan(R - errorR, R + errorR, alpha=0.1, color='r')
        axes[0].set_xlim(-20, 20)
        rsort  = np.sort(r[mask])
        c = ecdf(rsort)
        axes[1].plot(rsort, c.cdf.evaluate(rsort))
        axes[1].axhline(c.cdf.evaluate(R), ls='--', color='r')
        axes[1].axhspan(c.cdf.evaluate(R - errorR), c.cdf.evaluate(R + errorR), alpha=0.1, color='r')
        axes[1].set_xlabel(r'$\log R$')
        axes[1].set_ylabel(r'$P(\log R < \log R^\prime)$')
        axes[1].set_xlim(-20, 20)
        plt.tight_layout()
        plt.savefig(BASE_DIR + 'evaluations/' + f'run{run+1}_distribution.pdf', bbox_inches='tight')
        plt.close()

        # Compute width of the in concordance R distribution
        mu, width_std = norm.fit(r[mask]) # width is taken as the standard deviation of the fitted Gaussian
        widths.append(width_std)

        # Calculate the T and C statistics
        stats = calcualte_stats(R, errorR, c)
        sigmaD.append(stats[:3])
        sigmaA.append(stats[3:6])

    sigmaA = np.array(sigmaA)
    sigmaD = np.array(sigmaD)
    mean_sigmaA = sigmaA[:, 0].mean()
    mean_sigmaD = sigmaD[:, 0].mean()
    sigmaAs = sigmaA[:, 0]
    sigmaA_lower = sigmaA[:, 0] - sigmaA[:, 2]
    sigmaA_upper = sigmaA[:, 1] - sigmaA[:, 0]
    sigmaDs = sigmaD[:, 0]
    sigmaD_lower = sigmaD[:, 0] - sigmaD[:, 1]
    sigmaD_upper = sigmaD[:, 2] - sigmaD[:, 0]
    lower_mean_sigmaD_error = 1/np.sqrt(runs) * np.sqrt(np.sum((sigmaD_lower)**2))
    upper_mean_sigmaD_error = 1/np.sqrt(runs) * np.sqrt(np.sum((sigmaD_upper)**2))
    lower_mean_sigmaA_error = 1/np.sqrt(runs) * np.sqrt(np.sum((sigmaA_lower)**2))
    upper_mean_sigmaA_error = 1/np.sqrt(runs) * np.sqrt(np.sum((sigmaA_upper)**2))

    print('Plotting tension and concordance figures...')
    # Tension figure
    fig, axes = plt.subplots(1, 1, figsize=(3.5, 3))
    axes.errorbar(np.arange(len(sigmaDs)) + 1, sigmaDs, yerr=[sigmaD_lower, sigmaD_upper], fmt='o')
    axes.set_xticks(np.arange(len(sigmaDs)) + 1)
    axes.set_ylabel(r'$T$')
    axes.set_xlabel('Run')
    axes.axhline(mean_sigmaD, ls='--', c='r')
    axes.axhspan(mean_sigmaD - lower_mean_sigmaD_error, mean_sigmaD + upper_mean_sigmaD_error, alpha=0.1, color='r')
    plt.tight_layout()
    plt.savefig(BASE_DIR + 'evaluations/T.pdf', bbox_inches='tight')
    plt.close()

    # Concordance figure
    fig, axes = plt.subplots(1, 1, figsize=(3.5, 3))
    axes.errorbar(np.arange(len(sigmaAs)) + 1, sigmaAs, yerr=[sigmaA_lower, sigmaA_upper], fmt='o')
    axes.set_xticks(np.arange(len(sigmaAs)) + 1)
    axes.set_ylabel(r'$C$')
    axes.set_xlabel('Run')
    axes.axhline(mean_sigmaA, ls='--', c='r')
    axes.axhspan(mean_sigmaA - lower_mean_sigmaA_error, mean_sigmaA + upper_mean_sigmaA_error, alpha=0.1, color='r')
    plt.tight_layout()
    plt.savefig(BASE_DIR + 'evaluations/C.pdf', bbox_inches='tight')
    plt.close()

    # Save the statistics
    print('Saving statistics to', STATS_FILE)
    np.save(STATS_FILE, np.array([sigmaD, sigmaA], dtype=object))
    print('Done!')

tf.keras.backend.clear_session() # Clear session to reset memory use
