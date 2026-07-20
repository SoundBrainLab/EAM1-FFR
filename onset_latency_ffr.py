import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from glob import glob
from scipy.stats import ttest_rel
from matplotlib.widgets import Button
import pickle

out_file_name = 'onset_times_laura_3.pkl'
# define directoires
bids_root = '/Users/xjf5193/Library/CloudStorage/OneDrive-NorthwesternUniversity/SoundBrain Lab - EAM1/data-bids'
deriv_dir = os.path.join(bids_root, 'derivatives', 'epochs-python-stimtrack-closer-onsets-longer-baseline')
if not os.path.exists(deriv_dir):
    os.makedirs(deriv_dir)
events_dir = '/Users/xjf5193/Library/CloudStorage/OneDrive-NorthwesternUniversity/PhD/Soundbrain lab/EAM/stimtrack-compare/python_closer_onsets'

# read in epochs
act_pos_avgs = []
act_neg_avgs = []
act_eps = [sorted(glob(deriv_dir+f'/sub-{sub_num:02d}_task-active_run-all_event-stimtrack_epochs.fif') 
                  for sub_num in range(2,35))]
for act_ep in act_eps[0]:
    try:
        sub_active_epochs = mne.read_epochs(act_ep[0], verbose=False)
        act_pos_avgs.append(sub_active_epochs['1'].average())
        act_neg_avgs.append(sub_active_epochs['2'].average())
    except:
        pass

pas_pos_avgs = []
pas_neg_avgs = []
pas_eps = [sorted(glob(deriv_dir+f'/sub-{sub_num:02d}_task-passive_run-all_event-stimtrack_epochs.fif') 
                  for sub_num in range(2,35))]
for pas_ep in pas_eps[0]:
    try:
        sub_passive_epochs = mne.read_epochs(pas_ep[0], verbose=False)
        pas_pos_avgs.append(sub_passive_epochs['1'].average())
        pas_neg_avgs.append(sub_passive_epochs['2'].average())
    except:
        pass

# combien polarities
passive_avgs = [mne.combine_evoked([pas_pos_avgs[x], pas_neg_avgs[x]], weights='nave')
                for x in range(len(pas_pos_avgs))]
active_avgs = [mne.combine_evoked([act_pos_avgs[x], act_neg_avgs[x]], weights='nave')
                for x in range(len(act_pos_avgs))]

all_avgs_labeled = []
for i, (passive_avg, active_avg) in enumerate(zip(passive_avgs, active_avgs)):
    sub_num = i + 2
    all_avgs_labeled.append(('passive', passive_avg.times.copy(), passive_avg.data[0].copy(), sub_num))
    all_avgs_labeled.append(('active', active_avg.times.copy(), active_avg.data[0].copy(), sub_num))

# shuffle list to ensure random order of rows
random.shuffle(all_avgs_labeled)

# set up plot
global exclude_bool, exclude_trials, onset_times
exclude_bool = False
exclude_trials = 0
onset_times =[]

def onclick(event):
    """ Event handler for mouse clicks on the plot. 
    Records the x and y coordinates of the click if it occurs within the axes bounds. """
    if event.inaxes is not None and event.inaxes != button_ax:
        x = event.xdata
        y = event.ydata
        print(f"Clicked at x={x:.3f}, y={y:.3f}")
        onset_times.append({'x': x, 'y': y, 'task': '', 'sub_num': ''})
        plt.close(event.canvas.figure)
        return
    else:
        print("Clicked outside axes bounds but inside plot window")
        return 

def on_exclude(event):
    """ Event handler for the 'Exclude Trial' button. 
    Sets the exclude_bool to True and closes the current figure. """
    global exclude_bool, exclude_trials
    exclude_trials += 1
    exclude_bool = True
    plt.close(event.canvas.figure)
    return

# plot for selecting onset times and excluding trials
for (i, evk) in enumerate(all_avgs_labeled):
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(evk[1], evk[2])
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title(f"Trial {i + 1} / {len(all_avgs_labeled)}")

    button_ax = fig.add_axes([0.8, 0.01, 0.1, 0.05])
    exit_button = Button(button_ax, 'Exclude Trial', color='mistyrose', hovercolor='lightcoral')
    exit_button.on_clicked(on_exclude)

    fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show(block=True)

    if exclude_bool:
        exclude_bool = False
    else:
        onset_times[-1]['task'] = evk[0]  # Assign the task label to the last recorded onset time
        onset_times[-1]['sub_num'] = evk[3]  # Assign the subject number to the last recorded onset time

pickle.dump(onset_times, open(os.path.join(deriv_dir, out_file_name), 'wb'))
