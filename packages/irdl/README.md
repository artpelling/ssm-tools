# `irdl`: Impulse Response Downloader
Python package to download and process impulse response datasets.

## Available datasets
- [FABIAN-HRTF](https://doi.org/10.14279/depositonce-5718.5) via `irdl.get_fabian`

## Example

``` python
from irdl import get_fabian

data = get_fabian(kind='measured', hato=10)
print(data)
```

``` shell
{'impulse_response': time domain energy Signal:
(11950, 2) channels with 256 samples @ 44100.0 Hz sampling rate and none FFT normalization
,
 'receiver_coordinates': 2D Coordinates object with 2 points of cshape (2, 1)

Does not contain sampling weights,
 'source_coordinates': 1D Coordinates object with 11950 points of cshape (11950,)

Does not contain sampling weights}
```

## Key features
- Returns a dictionary of [`pyfar`](https://pyfar.org)-objects in a standardised way.
- Leverages [`pooch`](https://www.fatiando.org/pooch/latest/) to download impulse response datasets and verifies their integrity with a checksum. 
- Only downloads, extracts and processes what is needed.
- Adds `pooch`-support for dSpace repositories, such as [depositonce](https://depositonce.tu-berlin.de/home).
