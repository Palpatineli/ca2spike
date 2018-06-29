from pkg_resources import Requirement, resource_filename
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from keras.model import load_model as _load_model, Model
from keras import backend as K
from .utils import prep_data_one

MODEL_FREQ = 100  # see https://github.com/codeneuro/spikefinder-datasets/issues/1

def load_model() -> Model:
    return _load_model(resource_filename(Requirement.parse("ca2spike"), "data/weights_{}.h5".format(K.backend())))

def _resample(series: np.ndarray, source_freq: float, target_freq: float, axis: int = -1) -> np.ndarray:
    """resample the time series from timestamp1 to timestamp2
    Args:
        series: the input series or a bunch of series
        source_freq: the sampling frequency of the original series in Hz
        target_freq: the target sampling frequency
        axis: resample on which axis
    Returns:
        new series in target sampling frequency
    """
    source_timestamp = np.arange(series.shape[axis]) * (1.0 / source_freq)
    target_timestamp = np.arange(series.shape[axis] * target_freq / source_freq) * (1.0 / target_freq)

    def interpolate(x: np.ndarray) -> np.ndarray:
        return InterpolatedUnivariateSpline(source_timestamp, x, ext=0)(target_timestamp)
    return np.apply_along_axis(interpolate, axis, series)

def predict(data: np.ndarray, sample_rate: float, target_rate: float = None) -> np.ndarray:
    """Data is a array with rows for samples and columns for cells."""
    data = _resample(data, sample_rate, MODEL_FREQ, -1)
    traces, id_mat = prep_data_one(data)
    model = load_model()
    if target_rate is None:
        target_rate = sample_rate
    return _resample(model.predict([traces, id_mat]).squeeze(), MODEL_FREQ, target_rate, -1)
