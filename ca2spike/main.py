from typing import List, Tuple
from os.path import join, isfile, split
import numpy as np
from uifunc import FoldersSelector
from .convi.apply import predict

__all__ = ["deconvolve"]

def _find_config_file(end_folder: str, datetime: str) -> str:
    temp = end_folder
    while temp != "/":
        cfg_path = join(temp, datetime + '.cfg')
        if isfile(cfg_path):
            return cfg_path
        temp = split(temp)[0]
    raise IOError("scanning config file {0} doesn't exist".format(datetime + '.cfg'))

def deconvolve(data: np.ndarray, sample_rate: float=10.039) -> Tuple[np.ndarray, np.ndarray, int]:
    """deconvolve data
    Args:
        data: trace data, each row is the trace of one neuron, columns are samples
        sample_rate: frame rate of data_in
    Returns:
        spike_train, time_axis, new_sample_rate
    """
    target_rate = int(round(sample_rate))
    prediction = predict(data, sample_rate, target_rate)
    return prediction, np.arange(prediction.shape[-1]) / target_rate, target_rate

try:  # The following applies only to data files stored in noformat https://pypi.org/project/noformat/
    from noformat import File

    @FoldersSelector
    def convert(folder_paths: List[str]):
        for folder in folder_paths:
            data = File(folder, 'w+')
            if 'spike' not in data:  # not deconvolved
                print("starting file: \n\t{0}\nat frame rate {1}".format(split(folder)[1], data.attrs['frame_rate']))
                spike_train, time_axis, sample_rate = deconvolve(data['measurement'], data.attrs['frame_rate'])
                data['spike'] = {"data": spike_train, 'x': time_axis, 'y': data['measurement']['y']}
                data.attrs['spike_resolution'] = sample_rate
except ImportError:
    print("package 'noformat' not found. Script convert not available. "
          "Please read your data to numpy array and call deconvolve")
