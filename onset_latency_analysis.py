import pickle
import numpy as np
from scipy.stats import ttest_rel
import pingouin as pg
import pandas as pd
import pingouin as pg

def d_to_raw_bound(x, y, d):
    diff = np.array(x) - np.array(y)
    sd_diff = np.std(diff, ddof=1)
    return d * sd_diff

rater1 = pickle.load(open('/Users/xjf5193/Library/CloudStorage/OneDrive-NorthwesternUniversity/SoundBrain Lab - EAM1/data-bids/derivatives/epochs-python-stimtrack-closer-onsets-longer-baseline/onset_times_kevin.pkl', 'rb'))
rater2 = pickle.load(open('/Users/xjf5193/Library/CloudStorage/OneDrive-NorthwesternUniversity/SoundBrain Lab - EAM1/data-bids/derivatives/epochs-python-stimtrack-closer-onsets-longer-baseline/onset_times_laura_3.pkl', 'rb'))

# rater 1 data
active_times_by_subject_1 = {onset['sub_num']: onset['x'] for onset in rater1 if onset['task'] == 'active'}
passive_times_by_subject_1 = {onset['sub_num']: onset['x'] for onset in rater1 if onset['task'] == 'passive'}

common_subjects = sorted(set(active_times_by_subject_1.keys()) & set(passive_times_by_subject_1.keys()))
dropped_subjects = sorted(set(active_times_by_subject_1.keys()) ^ set(passive_times_by_subject_1.keys()))

print("===== Rater 1 Data =====")
print("Active onset times:", np.mean(list(active_times_by_subject_1.values())))
print("Passive onset times:", np.mean(list(passive_times_by_subject_1.values())))
print(f"Common subjects: {common_subjects}")
print(f"Dropped subjects: {dropped_subjects}")

active_times = [active_times_by_subject_1[sub] for sub in common_subjects]
passive_times = [passive_times_by_subject_1[sub] for sub in common_subjects]

t_stat, p_value = ttest_rel(active_times, passive_times)
print(f"Paired t-test results: t-statistic = {t_stat:.3f}, p-value = {p_value:.3f}")

# tost
equiv_range = d_to_raw_bound(active_times, passive_times, d=0.5)
tost_result = pg.tost(passive_times, active_times, paired=True, bound=equiv_range)
print(tost_result)

# bayes factor
bf_result = pg.ttest(passive_times, active_times, paired=True)
print(bf_result)

# rater 2 data
active_times_by_subject_2 = {onset['sub_num']: onset['x'] for onset in rater2 if onset['task'] == 'active'}
passive_times_by_subject_2 = {onset['sub_num']: onset['x'] for onset in rater2 if onset['task'] == 'passive'}

common_subjects_2 = sorted(set(active_times_by_subject_2.keys()) & set(passive_times_by_subject_2.keys()))
dropped_subjects_2 = sorted(set(active_times_by_subject_2.keys()) ^ set(passive_times_by_subject_2.keys()))

print("===== Rater 2 Data =====")
print("Active onset times:", np.mean(list(active_times_by_subject_2.values())))
print("Passive onset times:", np.mean(list(passive_times_by_subject_2.values())))
print(f"Common subjects: {common_subjects_2}")
print(f"Dropped subjects: {dropped_subjects_2}")

active_times_2 = [active_times_by_subject_2[sub] for sub in common_subjects_2]
passive_times_2 = [passive_times_by_subject_2[sub] for sub in common_subjects_2]
t_stat_2, p_value_2 = ttest_rel(active_times_2, passive_times_2)
print(f"Paired t-test results (Rater 2): t-statistic = {t_stat_2:.3f}, p-value = {p_value_2:.3f}")

# tost
equiv_range = d_to_raw_bound(active_times_2, passive_times_2, d=0.5)
tost_result = pg.tost(passive_times_2, active_times_2, paired=True, bound=equiv_range)
print(tost_result)

# bayes factor
bf_result = pg.ttest(passive_times_2, active_times_2, paired=True)
print(bf_result)

# assess intra-rater reliability

rows = []
rater_df_all = pd.DataFrame()
rater_df_all['Subject'] = common_subjects + common_subjects
for sub in common_subjects:
    rows.append({'Subject': sub, 'Rater': '1', 'onset_time': active_times_by_subject_1[sub], 'trial_type': 'active'})
    rows.append({'Subject': sub, 'Rater': '1', 'onset_time': passive_times_by_subject_1[sub], 'trial_type': 'passive'})
for sub in common_subjects_2:
    rows.append({'Subject': sub, 'Rater': '2', 'onset_time': active_times_by_subject_2[sub], 'trial_type': 'active'})
    rows.append({'Subject': sub, 'Rater': '2', 'onset_time': passive_times_by_subject_2[sub], 'trial_type': 'passive'})

rater_df_all = pd.DataFrame(rows)
icc = pg.intraclass_corr(data=rater_df_all, targets='Subject', raters='Rater', ratings='onset_time', nan_policy='omit')

print("Intra-class correlation (ICC) results:")
print(icc)

# average onset times across raters for each subject and trial type
print("===== Average Onset Times Across Raters =====")
active_times_avg = [(active_times_by_subject_1[sub] + active_times_by_subject_2[sub]) / 2 for sub in common_subjects]
passive_times_avg = [(passive_times_by_subject_1[sub] + passive_times_by_subject_2[sub]) / 2 for sub in common_subjects]

print("Average Active onset times:", np.mean(active_times_avg))
print("Average Passive onset times:", np.mean(passive_times_avg))

t_stat_avg, p_value_avg = ttest_rel(active_times_avg, passive_times_avg)
print(f"Paired t-test results (Average): t-statistic = {t_stat_avg:.3f}, p-value = {p_value_avg:.3f}")

# tost
equiv_range = d_to_raw_bound(active_times_avg, passive_times_avg, d=0.5)
tost_result = pg.tost(passive_times_avg, active_times_avg, paired=True, bound=equiv_range)
print(tost_result)

# bayes factor
bf_result = pg.ttest(passive_times_avg, active_times_avg, paired=True)
print(bf_result)