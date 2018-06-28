"""strategy to deconvolve calcium imaging using Theis2015 method"""
from typing import Dict, Tuple
import c2s
import numpy as np
from algorithm.time_series.resample import resample

DataFrame = Dict[str, np.ndarray]


def resample_spikes(time_series: np.ndarray, old_fps: float, new_fps: float) -> np.ndarray:
    return np.diff(resample(np.cumsum(time_series), old_fps, new_fps)) * (new_fps / old_fps)


def deconvolve(data: DataFrame, fps: float=10.039) -> Tuple[DataFrame, float]:
    """use c2s to deconvolve data
    Args:
        data: dict
            row: neuron ids
            data: trace data, each row is the trace of one neuron, columns are samples
        fps: frame rate of data_in
    """
    trace, row_id = (data[x] for x in ('data', 'x'))
    data_list = [{'calcium': row, 'fps': fps} for row in trace]
    # I'm using the default model. Output has fps==100
    result = c2s.predict(c2s.preprocess(data_list))
    new_fps = result[0]['fps']
    target_fps = int(round(fps))
    new_list = [resample_spikes(cell['predictions'][0, :], new_fps, target_fps) for cell in result]
    column_id = np.arange(new_list[0].shape[0]) / fps
    return {'data': np.array(new_list), 'x': row_id, 'y': column_id}, target_fps
