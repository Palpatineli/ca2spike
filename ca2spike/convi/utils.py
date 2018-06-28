from typing import List
import numpy as np
from os.path import join

dataloc = '../data/'
dataloc = '/opt/spikefinder/data/'

def prep_data(data: np.ndarray) -> np.ndarray:
    return data

def find_files() -> List[str]:
    path_fmt = join(dataloc, "spikefinder.train", "{data_id}.train.spikes.csv")
    return [path_fmt.format(data_id=data_id + 1) for data_id in range(10)]

def load_data(load_test=True):
    calcium_train = []
    spikes_train = []
    ids = []
    calcium_test = []
    ids_test = []
    for dataset in range(10):
        calcium_train.append(np.genfromtxt(dataloc + 
            'spikefinder.train/' + str(dataset+1) + 
            '.train.calcium.csv'), delimiter=',')
        spikes_train.append(np.array(pd.read_csv(dataloc + 
            'spikefinder.train/' + str(dataset+1) + 
            '.train.spikes.csv')))
        ids.append(np.array([dataset]*calcium_train[-1].shape[1]))
        if load_test and dataset < 5:
            calcium_test.append(np.array(pd.read_csv(dataloc +
                'spikefinder.test/' + str(dataset+1) +
                '.test.calcium.csv')))
            ids_test.append(np.array([dataset]*calcium_test[-1].shape[1]))

    maxlen = max([c.shape[0] for c in calcium_train])
    maxlen_test = max([c.shape[0] for c in calcium_test])
    calcium_train_padded = \
        np.hstack([np.pad(c, ((0, maxlen-c.shape[0]), (0, 0)),
            'constant', constant_values=np.nan) for c in calcium_train])
    spikes_train_padded = \
        np.hstack([np.pad(c, ((0, maxlen-c.shape[0]), (0, 0)),
            'constant', constant_values=np.nan) for c in spikes_train])
    calcium_test_padded = \
        np.hstack([np.pad(c, ((0, maxlen_test-c.shape[0]), (0, 0)),
        'constant', constant_values=np.nan) for c in calcium_test])
    ids_stacked = np.hstack(ids)
    if load_test:
        ids_test_stacked = np.hstack(ids_test)
    else:
        ids_test_stacked = []
    sample_weight = 1. + 1.5*(ids_stacked<5)
    sample_weight /= sample_weight.mean()
    calcium_train_padded[spikes_train_padded<-1] = np.nan
    spikes_train_padded[spikes_train_padded<-1] = np.nan

    calcium_train_padded[np.isnan(calcium_train_padded)] = 0.
    spikes_train_padded[np.isnan(spikes_train_padded)] = -1.

    calcium_train_padded = calcium_train_padded.T[:, :, np.newaxis]
    spikes_train_padded = spikes_train_padded.T[:, :, np.newaxis]
    calcium_test_padded = calcium_test_padded.T[:, :, np.newaxis]

    ids_oneshot = np.zeros((calcium_train_padded.shape[0],
        calcium_train_padded.shape[1], 10))
    ids_oneshot_test = np.zeros((calcium_test_padded.shape[0],
        calcium_test_padded.shape[1], 10))
    for n,i in enumerate(ids_stacked):
        ids_oneshot[n, :, i] = 1.
    for n,i in enumerate(ids_test_stacked):
        ids_oneshot_test[n, :, i] = 1.

    return calcium_train, calcium_train_padded, spikes_train_padded,\
            calcium_test_padded, ids_oneshot, ids_oneshot_test,\
            ids_stacked, ids_test_stacked, sample_weight
