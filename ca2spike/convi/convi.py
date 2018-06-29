# this file is not type annotated because tensorflow doesn't document it
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from .utils import training_data, testing_data, prep_data

from keras.models import Model
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import Concatenate
from keras.layers import Activation, Dropout, Input, LSTM
from keras.layers.convolutional import Conv1D
from keras.callbacks import TensorBoard
from keras import backend as K

__all__ = ["create_model", "model_train", "model_test"]

def _pearson_corr(y_true, y_pred, pool=True):
    """Calculates Pearson correlation as a metric.
    Args:
        y_true: true tensor with shape (batch_size, num_timesteps, 1)
        y_pred: predition tensor with shape (batch_size, num_timesteps, 1)
        pool: whether pool together timepoints in both y values
    Returns:
        tensor of one float
    """
    def pool1d(x, length=4):  # because there is no backend.pool1d
        """Adds groups of `length` over the time dimension in x.
        Args:
            x: 3D Tensor with shape (batch_size, time_dim, feature_dim).
            length: the pool length.
        Returns:
            3D Tensor with shape (batch_size, time_dim // length, feature_dim).
        """
        return K.squeeze(K.pool2d(
            K.expand_dims(x, 2), (length, 1), (length, 1), 'same'), 2) * length
    if pool:
        y_true = pool1d(y_true, length=4)
        y_pred = pool1d(y_pred, length=4)
    mask = K.cast(y_true >= 0., K.floatx())
    samples = K.sum(mask, axis=1, keepdims=True)
    x_mean = y_true - K.sum(mask * y_true, axis=1, keepdims=True) / samples
    y_mean = y_pred - K.sum(mask * y_pred, axis=1, keepdims=True) / samples
    # Numerator and denominator.
    n = K.sum(x_mean * y_mean * mask, axis=1)
    d = (K.sum(K.square(x_mean) * mask, axis=1) *
         K.sum(K.square(y_mean) * mask, axis=1))
    return 1. - K.mean(n / (K.sqrt(d) + 1e-12))

def create_model():
    '''Create a mixed network.
    ca+ wave ---> conv * 2 + conv + LSTM + conv * 5 ---> sigmoid
                           ↑
        dataset labels ----」
    Dropout for layers 1, 2, 3, 5
    pearson correlation loss for the spike train
    '''
    main_input = Input(shape=(None, 1), name='main_input')
    dataset_input = Input(shape=(None, 10), name='dataset_input')
    x = Conv1D(10, 300, padding='same', input_shape=(None, 1))(main_input)
    x = Dropout(0.3)(Activation('tanh')(x))
    x = Dropout(0.2)(Activation('relu')(Conv1D(10, 10, padding='same')(x)))
    x = Concatenate()([x, dataset_input])
    x = Dropout(0.1)(Activation('relu')(Conv1D(10, 5, padding='same')(x)))
    z = Bidirectional(LSTM(10, return_sequences=True), merge_mode='concat', weights=None)(x)
    x = Concatenate()([x, z])
    x = Activation('relu')(Conv1D(8, 5, padding='same')(x))
    x = Dropout(0.1)(x)
    x = Activation('relu')(Conv1D(4, 5, padding='same')(x))
    x = Activation('relu')(Conv1D(2, 5, padding='same')(x))
    x = Activation('relu')(Conv1D(2, 5, padding='same')(x))
    output = Activation('sigmoid')(Conv1D(1, 5, padding='same')(x))
    model = Model(inputs=[main_input, dataset_input], outputs=output)
    model.compile(loss=_pearson_corr, optimizer='adam')
    return model

def model_train(model: Model, data_folder: str, target_folder: str) -> Model:
    try:
        tf_board = TensorBoard(log_dir='./logtest2', histogram_freq=0, write_graph=True, write_images=True)
        block_callback = [tf_board]
    except ImportError:
        block_callback = []
    params = prep_data(training_data(data_folder), 5)
    model.fit([params["calcium"], params["id_mat"]], params["spikes"], epochs=1,
              batch_size=5, validation_split=0.2, sample_weight=params["sample_weight"], callbacks=block_callback)
    model.save_weights(join(target_folder, 'model-{}.h5'.format(K.backend())))
    return model

def model_test(model: Model, data_folder: str) -> np.ndarray:
    data = testing_data(data_folder)
    preped_data = prep_data(data)
    pred_test = model.predict([preped_data["calcium"], preped_data["id_mat"]])
    for data_id in range(5):
        pred_test[prep_data["id_train"] == data_id, 0: data["calcium"][data_id].shape[0]].squeeze()

def plot_kernels(model: Model, layer: int = 0):
    srate = 100.
    weights = model.get_weights()[layer]
    t = np.arange(-weights.shape[0] / srate / 2, weights.shape[0] / srate / 2, 1. / srate)
    for j in range(weights.shape[2]):
        plt.plot(t, weights[:, 0, j] + .3 * j)
    plt.xlabel('Time [s]')
    plt.ylabel('Kernel amplitudes')
    plt.title('Convolutional kernels of the input layer')
    plt.show()
