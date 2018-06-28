from typing import List
from os.path import join, isfile, split
from noformat import File
from uifunc import FoldersSelector
from .theis import deconvolve

def _find_config_file(end_folder: str, datetime: str) -> str:
    temp = end_folder
    while temp != "/":
        cfg_path = join(temp, datetime + '.cfg')
        if isfile(cfg_path):
            return cfg_path
        temp = split(temp)[0]
    raise IOError("scanning config file {0} doesn't exist".format(datetime + '.cfg'))

@FoldersSelector
def convert(folder_paths: List[str]):
    for folder in folder_paths:
        data = File(folder, 'w+')
        if 'spike' not in data:  # not deconvolved
            print("starting file: \n\t{0}\nat frame rate {1}".format(split(folder)[1], data.attrs['frame_rate']))
            data['spike'], data.attrs['spike_resolution'] = deconvolve(data['measurement'], data.attrs['frame_rate'])
