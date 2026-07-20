# Auditory–motor interactions in the early auditory system
Frequency-following response (FFR) analysis code for the EAM1 project

### Manuscript details
Motor signals modulate cortical but not subcortical processing of self-initiated sounds
Laura Raiff, Gabrielle Butler, Kailyn McFarlane, Bharath Chandrasekaran, Kevin R. Sitek
bioRxiv 2026.07.10.737812; doi: [https://doi.org/10.64898/2026.07.10.737812](https://doi.org/10.64898/2026.07.10.737812) 

### Data availability
Availble in BIDS format: https://openneuro.org/datasets/ds008161

### Analysis steps
1. Extract precise sound onset timing from the EEG ergo channel: [get_event_onsets_stimtrack.ipynb](get_event_onsets_stimtrack.ipynb)
2. Convert the EEG data to the Brain Imaging Data Structure (EEG-BIDS): [bids-conversion.ipynb](bids-conversion.ipynb)
3. Preprocess and analyze the FFRs: [ffr_preprocessing_analysis.ipynb](ffr_preprocessing_analysis.ipynb), [onset_latency_ffr.py](onset_latency_ffr.py), and [onset_latency_analysis.py](onset_latency_ffr.py)
4. Preprocess and analyze the cortical ERPs: [erp_preprocessing_analysis.ipynb](erp_preprocessing_analysis.ipynb)
5. Analyse the phase consistency of the FFRs and cortical ERPs [plot_phase_consistency.ipynb](plot_phase_consistency.ipynb)


### How to contribute
Before running any of the notebooks, please set up the following gitconfig filter to avoid metadata clutter:
- install required packages: `pip install -r requirements.txt`
- Make sure nbconvert is installed globally (`pip install nbconvert` or `conda install nbconvert`)
- Windows: in a bash terminal, run: `bash setup-git-jupyter-filters.sh`
- Mac: run `chmod +x setup-git-jupyter-filters.sh` then `./setup-git-jupyter-filters.sh `
