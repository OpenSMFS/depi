from pathlib import Path
import json
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

stream_map = {0: 'DexDem', 1: 'DexAem', 2: 'AexDem', 3: 'AexAem'}
stream_dtype = CategoricalDtype(categories=stream_map.values())


def fname_from_pattern(pattern, folder):
    matching_files = [f for f in Path(folder).glob(pattern)]
    if len(matching_files) == 0:
        msg = f'No file matching the pattern "{pattern}"'
        raise FileNotFoundError(msg)
    elif len(matching_files) > 1:
        msg = f'Multiple files matching the pattern "{pattern}":\n'
        for f in matching_files:
            msg += f'- {f}\n'
        raise FileNotFoundError(msg)
    else:
        return matching_files[0]


def load_burst_and_ph_data(name, folder, pop):
    suffixb = '_%s_bursts'
    suffixph = '_%s_burst_photons'
    if isinstance(name, str):
        patternph = f'{name}_merge{suffixph % pop}.csv'
    else:
        conc = name
        patternph = f'*{conc:.1f}M_merge{suffixph % pop}.csv'
    fname = fname_from_pattern(patternph, folder)
    burstsph = pd.read_csv(fname, skiprows=1, index_col=(0, 1))
    burstsph.stream = burstsph.stream.astype(stream_dtype)
    header = fname.read_text().split('\n')[0]
    meta = json.loads(header)
    meta['timestamp_unit_hw'] = meta['timestamp_unit']
    scale = meta['timestamp_unit'] * 1e9
    scale = int(scale) if round(scale) == scale else scale
    burstsph.timestamp *= scale
    meta['timestamp_unit'] = 1e-9
    print(f'- Loaded photon-data "{fname}"')

    if isinstance(name, str):
        patternb = f'{name}_merge{suffixb % pop}.csv'
    else:
        patternb = f'*{conc:.1f}M_merge{suffixb % pop}.csv'
    fname = fname_from_pattern(patternb, folder)
    print(fname)
    bursts = pd.read_csv(fname, index_col=0)
    istart = np.hstack([[0], np.cumsum(bursts.size_raw)[:-1]])
    istop = np.cumsum(bursts.size_raw.values)
    assert istart.size == istop.size
    bursts['istart'] = istart
    bursts['istop'] = istop
    print(f'- Loaded burst data "{fname}"')

    # Tests
    assert (np.diff(burstsph.timestamp) >= 0).all()
    num_bursts = np.unique(burstsph.reset_index('burst').burst).size
    assert bursts.shape[0] == num_bursts
    assert (burstsph.groupby('burst')['timestamp'].count() == bursts.size_raw).all()
    return bursts, burstsph, meta
