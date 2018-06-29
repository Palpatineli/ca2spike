from typing import Dict, List, Tuple
import numpy as np
from os.path import join, split
from glob import iglob

DATASET_DEPTH = 10

Data = Dict[str, List[np.ndarray]]

__all__ = ["training_data", "testing_data", "prep_data", "prep_data_one"]

def training_data(data_loc: str) -> Data:
    data_list: Dict[str, list] = {"calcium": list(), "spikes": list()}
    id_lists: Dict[str, list] = {"calcium": list(), "spikes": list()}
    path_fmt = join(data_loc, "spikefinder.train", "*.train.{0}.csv")
    for record_type in ("calcium", "spikes"):
        for file_path in iglob(path_fmt.format(record_type)):
            data_list[record_type].append(np.genfromtxt(file_path, delimiter=','))
            file_basename = split(file_path)[-1]
            id_lists[record_type].append(int(file_basename[0: file_basename.index('.')]) - 1)
    # only take files where both calcium and spikes are available
    id_list = sorted(set(id_lists["spikes"]) & set(id_lists["calcium"]))
    for record_type in ("calcium", "spikes"):
        data_list[record_type] = np.take(data_list[record_type], _search_ar(id_list, id_lists[record_type]))
    data_list["id_train"] = [np.full((train.shape[1],), idx) for idx, train in zip(id_list, data_list["calcium"])]
    return data_list

def testing_data(data_loc: str) -> Data:
    path_fmt = join(data_loc, "spikefinder.test", "*.test.calcium.csv")
    test_list: List[np.ndarray] = list()
    id_list = list()
    for file_path in iglob(path_fmt):
        test_list.append(np.genfromtxt(file_path, delimiter=','))
        file_basename = split(file_path)[-1]
        id_list.append(int(file_basename[0: file_basename.index('.')]) - 1)
    index = np.argsort(id_list)
    id_list = np.take(id_list, index)
    test_list = np.take(test_list, index)
    return {"id_train": [np.full((test.shape[1],), idx) for idx, test in zip(id_list, test_list)],
            "calcium": test_list}

TRACE_FILL = {"spikes": -1, "calcium": 0}

def prep_data(data: Data, focus_1st_n: int = 0) -> Dict[str, np.ndarray]:
    """Prepare data for training or prediction.
    Args:
        data: read from csv files, with "calcium" and "id_train", optionally with "spikes"
            "calcium" and "spikes" have cells in columns and samples in rows
    """
    result = {"id_train": np.hstack(data['id_train'])}
    max_len = max([x.shape[0] for x in data["calcium"]])
    for trace_type, fill in TRACE_FILL.items():
        if trace_type not in data:
            continue
        result[trace_type] = np.hstack([np.pad(x, (0, max_len - x.shape[0]), (0, 0), "constant", constant_values=fill)
                                        for x in data[trace_type]])
    if "spikes" in result:
        bad_spikes = result["spikes"] < -1
        result["spikes"][np.logical_and(bad_spikes, np.isnan(result["spikes"]))] = TRACE_FILL["spikes"]
        result["calcium"][np.logical_and(bad_spikes, np.isnan(result["calcium"]))] = TRACE_FILL["calcium"]
        result["spikes"] = result["spikes"].T[:, :, np.newaxis]
    else:
        result["calcium"][np.isnan(result["calcium"])] = TRACE_FILL["calcium"]
    result["calcium"] = result["calcium"].T[:, :, np.newaxis]
    result["id_mat"] = _expand2bool(result["id_train"], (*result["calcium"][0: 2], DATASET_DEPTH))
    result["sample_weight"] = _sample_weight(result["id_train"], focus_1st_n) if focus_1st_n\
        else np.full_like(result["id_train"], 1)
    return result

def prep_data_one(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    data[np.isnan(data)] = TRACE_FILL["calcium"]
    id_train = np.full((data.shape[1],), 0)
    data = data[:, :, np.newaxis]
    id_mat = _expand2bool(id_train, (*data.shape[0: 2], DATASET_DEPTH))
    return data, id_mat

def _expand2bool(data: np.ndarray, shape) -> np.ndarray:
    """Convert a index array to a boolean mask.
    This is a dimension specific version.
    Args:
        data: 1-d array with length N
    Returns:
        3-d matrix
    """
    result = np.zeros(shape, dtype=np.float_)
    result[np.arange(data.size), :, data.ravel()] = 1.0
    return result

def _sample_weight(id_train: np.ndarray, test_set_n: int) -> float:
    """Give higher weight to the first 5 training sets.
    I have no idea why the authors did that.
    """
    sample_weight = 1.5 * (id_train < test_set_n) + 1.0
    return sample_weight / sample_weight.mean()

def _search_ar(array1: np.ndarray, array2: np.ndarray) -> np.ndarray:
    """Find the locations of array1 elements in array2"""
    arg2 = np.argsort(array2)
    arg1 = np.argsort(array1)
    rev_arg1 = np.argsort(arg1)
    sorted2_to_sorted1 = np.searchsorted(array2[arg2], array1[arg1])
    return arg2[sorted2_to_sorted1[rev_arg1]]
