"""03a_optuna_search1.py: first Optuna hyperparameter search (Appendix B).

OPTIONAL: not needed to train the NREs or to run anything else in this
repository. 04_train_nre.py already defaults to the hyperparameters chosen from
these searches for the paper; this is included so the process can be repeated.

*** HPC-GRADE *** Trains one Planck-RACS NRE per trial (43 in the paper) on 4 GPUs
for up to 48 hours (see pbs/optuna_search.pbs).

Searches over the number of filters, dense layers, neurons and the dropout rate,
training each NRE for at most 15 epochs on 200 000 simulation pairs (patience 10).
Every trial is compared with the nested sampling ground truth (from 02c), saving
the R^2 score, KL divergence and KS statistic between the predicted and true in
concordance log R distributions to its own directory in outputs/optuna/search1/.
The first search heavily favoured one filter, which was fixed for the second search.

Needs: the Planck (first) and RACS-low (second) simulations from 01a, and
outputs/validation/distribution/r_actual.npy from 02c.

Usage:
    python 03a_optuna_search1.py
"""

# Import packages
print('Importing packages...')
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 0 = All messages, 1 = Filter out INFO messages, 2 = Filter out WARNING, 3 = Filter out all messages (Fatal errors remain)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import healpy as hp
import nnhealpix.layers
import config
from utils import pair_simulations_for_optuna
import matplotlib as mpl
from sklearn.metrics import r2_score
import optuna
import scipy as sp
mpl.rcParams['text.usetex'] = False # Fix latex errors on cluster

# Tensorflow GPU configuration
print('Configuring TensorFlow...')
physical_devices = tf.config.list_physical_devices()
print("Available physical devices:", physical_devices) # Print TensorFlow device info
ngpus = len(tf.config.list_physical_devices('GPU'))
if ngpus == 1:
    print(f"{ngpus} GPU available.")
    strategy = tf.distribute.get_strategy()
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
tf.config.set_soft_device_placement(True) # allow TensorFlow to fall back to CPU if GPU is not available
if os.uname().sysname == 'Darwin':
    print('Running on macOS - disabling XLA')
    os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
    os.environ["TF_DISABLE_MLIR_BRIDGE"] = "1"
# else:
#     print('Running on Linux - enabling XLA')
#     tf.config.optimizer.set_jit(True) # enable XLA (Accelerated Linear Algebra) for performance improvements

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--nsamples', type=int, default=200000, help='simulation pairs used by this search (paper: 200000)')
parser.add_argument('--nchunks', type=int, default=20, help='chunks they are split into (paper: 20)')
parser.add_argument('--full-nchunks', type=int, default=config.N_CHUNKS, help='--nchunks used for 01a_make_simulations.py')
parser.add_argument('--epochs', type=int, default=15, help='maximum epochs per trial (paper: 15)')
parser.add_argument('--n-trials', type=int, default=400, help='number of trials in this job')
parser.add_argument('--batch-size', type=int, default=None, help='override the batch size')
parser.add_argument('--skip-plots', action='store_true', help='skip the Optuna visualisations at the end')
args = parser.parse_args()

nChunks = args.nchunks
nSamples = args.nsamples
dtype = np.float32
nside_in = 64
patience = 10
epochs = args.epochs
minimum_accuracy_after_2_epochs = 0.55 # This is used to abort training and restart if it is not going well, for whatever reason (I have observed this occuring)

# Set directories
search = "search1"
BASE_DIR = os.path.join(config.OPTUNA_DIR, search) + '/'
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)
SIMULATION_DIR = os.path.join(config.OPTUNA_DIR, 'paired_simulations') + '/' # shared by both searches

# Pair up the Planck and RACS-low simulations (from 01a) into the chunks read below
print('Pairing up the Planck and RACS-low simulations...')
pair_simulations_for_optuna(SIMULATION_DIR, nSamples, nChunks, config.SIMULATION_DIR, args.full_nchunks)

# Load the nested sampling ground truth log R of the first in concordance chunk (from 02c)
NS_STATS_DIR = config.VALIDATION_DIR
print('Loading R values for evaluation...')
r_actual = np.load(NS_STATS_DIR + 'r_actual.npy')
rerr_actual_lower = np.load(NS_STATS_DIR + 'rerr_actual_lower.npy')
rerr_actual_upper = np.load(NS_STATS_DIR + 'rerr_actual_upper.npy')

# Check that the simulation data exists
files = []
# Calculate chunk sizes as in the simulation function
num_Samples = nSamples
chunk_size = num_Samples // nChunks
concordant_chunks = int(nChunks * 0.1)
training_chunks = int(nChunks * 0.6)
test_chunks = nChunks - concordant_chunks - training_chunks
# Add training data and label files
for k in range(training_chunks):
    files.append(SIMULATION_DIR + f'nside{nside_in}_data_train_chunk{k+1}.npy')
    files.append(SIMULATION_DIR + f'nside{nside_in}_labels_train_chunk{k+1}.npy')
# Add test data and label files (validation)
for k in range(test_chunks):
    files.append(SIMULATION_DIR + f'nside{nside_in}_data_test_chunk{k+1}.npy')
    files.append(SIMULATION_DIR + f'nside{nside_in}_labels_test_chunk{k+1}.npy')
# Add concordant data files
for k in range(concordant_chunks):
    files.append(SIMULATION_DIR + f'nside{nside_in}_data_concordant_chunk{k+1}.npy')
if all(os.path.exists(f) for f in files):
    print('All simulation files found.')
else:
    print('Some simulation files are missing. Please generate them.')
    exit()

# Loading data for evaluation
test_chunk = np.load(SIMULATION_DIR + f'nside{nside_in}_data_concordant_chunk1.npy', mmap_mode='r')
test_chunk = test_chunk.reshape(test_chunk.shape[0], -1, 1) # Reshape to (N, 2*49152, 1) for model input
test_chunk = test_chunk[:len(r_actual)] # the nested-sampled pairs are the first pairs of this chunk

# Set NBB function
def NBB(x, nside_in, nside_out, filters=1):
    nside = nside_in
    while nside > nside_out:
        x = nnhealpix.layers.ConvNeighbours(nside, filters=filters, kernel_size=9)(x)
        nside = nside // 2
        x = nnhealpix.layers.MaskedAveragePooling(nside*2, nside, mask_value=0.0)(x)
    return x

batch_size_scales = np.zeros(11)
batch_size_scales[1] = 1
batch_size_scales[4] = 2
batch_size_scales[7] = 2.5
batch_size_scales[10] = 3

# Hyperparameter optimization with Optuna
print('Starting hyperparameter optimization with Optuna...')
def objective(trial):

    # Hyperparameter suggestions:
    print('Generating hyperparameters for trial...')
    filters = trial.suggest_int('filters', 1, 10, step=3) # x4
    nside_out = 4 # x1
    layers = trial.suggest_int('layers', 2, 5) # x4
    neurons = trial.suggest_categorical('neurons', [32, 64, 128, 256, 512]) # x5
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.20, step=0.05) # x5

    NETWORK_DIR = BASE_DIR + f'nsideout{nside_out}_filters{filters}_layers{layers}_neurons{neurons}_dropout{dropout_rate:.2f}/'
    if not os.path.exists(NETWORK_DIR):
        os.makedirs(NETWORK_DIR)
    print(f'Network directory: {NETWORK_DIR}')
    if 'model' in locals():
        del model

    # Data streaming setup for training and validation
    if 'val_gen' in locals():
        del val_gen
    if 'train_gen' in locals():
        del train_gen
    batch_size = int(1500 * 4 / batch_size_scales[filters]) # Adjust batch size based on number of filters
    if args.batch_size is not None:
        batch_size = args.batch_size
    print('Setting up data streaming...')
    # Get the list of training and validation chunk files
    train_data_files = [SIMULATION_DIR + f'nside{nside_in}_data_train_chunk{i+1}.npy' for i in range(training_chunks)]
    train_label_files = [SIMULATION_DIR + f'nside{nside_in}_labels_train_chunk{i+1}.npy' for i in range(training_chunks)]
    val_data_files = [SIMULATION_DIR + f'nside{nside_in}_data_test_chunk{i+1}.npy' for i in range(test_chunks)]
    val_label_files = [SIMULATION_DIR + f'nside{nside_in}_labels_test_chunk{i+1}.npy' for i in range(test_chunks)]
    # Load all data into memory using memory mapping
    print('Loading data into memory with memory mapping...')
    train_data = [np.load(f, mmap_mode='r') for f in train_data_files]
    train_labels = [np.load(f, mmap_mode='r') for f in train_label_files]
    val_data = [np.load(f, mmap_mode='r') for f in val_data_files]
    val_labels = [np.load(f, mmap_mode='r') for f in val_label_files]
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
    # Generator yielding pre-made batches (no processing during training)
    def static_batch_generator(batches_data, batches_labels):
        while True:
            for data, labels in zip(batches_data, batches_labels):
                yield data, labels
    def tf_static_batch_generator(batches_data, batches_labels):
        num_features = batches_data[0].shape[1]
        output_signature = (
            tf.TensorSpec(shape=(None, num_features, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
        )
        gen = lambda: static_batch_generator(batches_data, batches_labels)
        return tf.data.Dataset.from_generator(gen, output_signature=output_signature).prefetch(tf.data.AUTOTUNE)
    # Build generators using pre-made batches
    print('Building static data generators...')
    train_gen = tf_static_batch_generator(train_batches_data, train_batches_labels)
    val_gen = tf_static_batch_generator(val_batches_data, val_batches_labels)

    with strategy.scope():
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
        # Fully connected layers based on the hyperparameters
        for i in range(layers):
            x = tf.keras.layers.Dense(neurons)(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.Dropout(dropout_rate)(x)
        # Final output layer
        x = tf.keras.layers.Dense(1)(x)
        out = tf.keras.layers.Activation('linear')(x) # linear, to output logR

        # Variable learning rate
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-3,
            decay_steps=1000,
            decay_rate=0.9,
            staircase=True
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        # Define custom binary cross-entropy loss function
        def binary_crossentropy(y_true, logR):
            y_pred = tf.sigmoid(logR) # sigmoid activation
            epsilon = 1e-7  # To avoid log(0)
            y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)  # Clip predictions to avoid log(0)
            loss = -tf.reduce_mean(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
            return loss

        # Compile model
        model = tf.keras.models.Model(inputs=inputs, outputs=out)
        model.compile(loss=binary_crossentropy, optimizer=optimizer, metrics=['accuracy'])

    model.summary()

    if os.path.exists(NETWORK_DIR + 'weights.keras'):
        print('Loading existing model weights...')
        model.load_weights(NETWORK_DIR + 'weights.keras')
    else:
        # Setup early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True)

        print('Training model weights...')
        # Train the model using generators
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

        # Training loop that will abort if training is not going well (does not meet target accuracy after 2 epochs)
        max_retries = 50
        for attempt in range(max_retries):
            abort_callback = AbortAndRestartCallback(min_acc=minimum_accuracy_after_2_epochs, check_epoch=2)

            history = model.fit(
                x=train_gen,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                validation_data=val_gen,
                validation_steps=validation_steps,
                callbacks=[early_stopping, abort_callback],
                verbose=1
            )

            if not abort_callback.should_abort:
                break

        if abort_callback.should_abort:
            print(f"Training repeatedly failed to reach minimum accuracy after {max_retries} attempts.")
            print("Training regardless...")
            # Save a zero-byte file as a note
            with open(NETWORK_DIR + 'bad_training.txt', 'w') as f:
                pass

            history = model.fit(
                x=train_gen,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                validation_data=val_gen,
                validation_steps=validation_steps,
                callbacks=[early_stopping],
                verbose=1
            )

        # Save the model to BASE_DIR
        print('Saving model weights...')
        model.save(NETWORK_DIR + 'weights.keras')

    # Compare Actual R with predicted R
    if not os.path.exists(NETWORK_DIR + 'R2_score.npy'):
        if not os.path.exists(NETWORK_DIR + 'prediction.npy'):
            print('Predicting logR values for', NETWORK_DIR)
            r_predicted = model.predict(test_chunk)
            np.save(NETWORK_DIR + 'prediction.npy', r_predicted)
        else:
            print('Loading logR predictions for', NETWORK_DIR)
            r_predicted = np.load(NETWORK_DIR + 'prediction.npy', mmap_mode='r')
        print('Plotting predicted vs actual logR...')
        plt.figure(figsize=(4, 4))
        plt.errorbar(r_actual, r_predicted, xerr=np.vstack([rerr_actual_lower, rerr_actual_upper]), fmt='o', color='gray', markersize=2, alpha=0.2, linewidth=1)
        plt.xlabel(r'True $\log R$')
        plt.ylabel(r'Predicted $\log R$')
        print('Calculating R^2 score...')
        r2 = r2_score(r_actual, r_predicted)
        plt.title(r"R$^2$ = "+f"{r2:.4f}")
        minR = min([np.nanmin(r_actual), np.nanmin(r_predicted)])
        maxR = max([np.nanmax(r_actual), np.nanmax(r_predicted)])
        plt.plot([minR, maxR], [minR, maxR], linestyle='--', color='gray', linewidth=1)
        plt.legend()
        plt.xlim(minR, maxR)
        plt.ylim(minR, maxR)
        plt.tight_layout()
        plt.savefig(NETWORK_DIR + 'R2_plot.pdf', bbox_inches='tight')
        np.save(NETWORK_DIR + 'R2_score.npy', r2)

        # Compute KL divergence between true and predicted logR distributions
        r_true = np.asarray(r_actual).flatten()
        r_pred = np.asarray(r_predicted).flatten()
        # Ensure matched lengths
        n = min(len(r_true), len(r_pred))
        r_true = r_true[:n]
        r_pred = r_pred[:n]
        # Remove non-finite entries
        mask = np.isfinite(r_true) & np.isfinite(r_pred)
        r_true = r_true[mask]
        r_pred = r_pred[mask]
        # Use KDEs on a shared grid and compute continuous KL: ∫ p(x) log(p(x)/q(x)) dx
        lo = min(r_true.min(), r_pred.min())
        hi = max(r_true.max(), r_pred.max())
        pad = 0.1 * (hi - lo) if hi > lo else 1.0
        grid = np.linspace(lo - pad, hi + pad, 1000)
        kde_true = sp.stats.gaussian_kde(r_true)
        kde_pred = sp.stats.gaussian_kde(r_pred)
        p = kde_true(grid)
        q = kde_pred(grid)
        # Normalize PDFs to integrate to 1 on the grid
        p /= np.trapz(p, grid)
        q /= np.trapz(q, grid)
        eps = 1e-12
        kl = np.trapz(p * np.log((p + eps) / (q + eps)), grid)
        kl_str = f"{kl:.6e}"
        print(f'KL divergence:', kl)

        # Compute KS statistic between true and predicted logR distributions
        ks_stat, ks_pvalue = sp.stats.ks_2samp(r_true, r_pred)
        ks_stat_str = f"{ks_stat:.6e}"
        print(f'KS statistic:', ks_stat)
        ks_pvalue_str = f"{ks_pvalue:.6e}"
        print(f'KS p-value:', ks_pvalue)

        # Plot KDEs for visual comparison
        pt = kde_true(grid)
        qt = kde_pred(grid)
        # Normalize for plotting
        pt /= np.trapz(pt, grid)
        qt /= np.trapz(qt, grid)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(grid, pt, label='True Distribution (KDE)', color='C0', lw=1.5)
        ax.plot(grid, qt, label='Predicted Distribution (KDE)', color='C1', lw=1.5)
        ax.fill_between(grid, 0, pt, color='C0', alpha=0.2)
        ax.fill_between(grid, 0, qt, color='C1', alpha=0.2)
        ax.set_xlabel(r'$\log R$')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.set_xlim(grid[0], grid[-1])
        ax.set_title(f'KL Divergence: {kl_str}, KS Stat: {ks_stat_str}, KS p-value: {ks_pvalue_str}', fontsize=8)
        plt.tight_layout()
        plt.savefig(NETWORK_DIR + 'distribution_comparison.pdf', bbox_inches='tight')
        print('Saved distribution comparison to', NETWORK_DIR + 'distribution_comparison.pdf')
    else:
        print('Loading existing R^2 score...')
        r2 = np.load(NETWORK_DIR + 'R2_score.npy')

    tf.keras.backend.clear_session() # Clear session to reset memory use
    return r2

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))