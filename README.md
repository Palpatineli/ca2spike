# Ca2Spike

Deconvolve calcium imaging data to spikes. A wrapper around Chenkov and McColgan so it's easier to use their model and weights.
Assuming you have a imageJ roi measurement csv file, and a known sample_rate of your two photon recording:
```python
import numpy as np
from ca2spike import deconvolve
calcium_data = np.genfromtxt(file_path, delimiter=',')[1:, 1::4].T
spike_data, timestamps, new_sample_rate = deconvolve(data, sample_rate)
```
Here the new sample rate is just a rounded int, coz otherwise your analysis experience will suck.
