from pkg_resources import Requirement, resource_filename
from keras.model import load_model
from .utils import prep_data

def predict(data: np.ndarray, sample_rate: float) -> np.ndarray:
    load_model
