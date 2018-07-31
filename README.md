# DEPI

Monte-Carlo Diffusion-Enhanced Photon Inference (MC-DEPI) is a method to simulate solution-based single-molecule FRET experiments 
and to estimate physical parameters to the experimental data.

The input is the burst-selected photon data from a smFRET experiment. DEPI simulates photon color and nanotimes, based on 
donor-acceptor self-diffusion. This allows to accurately simulate smFRET experiment taking into account the FRET-enhancenment
due to the donor-acceptor diffusion happening during the fluorescence lifetime.

# Installation

The most update version of depi can be installed from github:

```
pip install git+https://github.com/OpenSMFS/depi/ --upgrade
```

# Dependencies

- python >=3.6
- numpy >=1.9
- scipy >=1.0
- matplotlib >=2.0
- cython >= 0.23
- pandas >= 0.22
- joblib >= 0.11
