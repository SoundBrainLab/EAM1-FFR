import pickle
import numpy as np
from scipy.stats import ttest_rel
import pingouin as pg
import pandas as pd
import pingouin as pg
import os
import matplotlib.pyplot as plt

def d_to_raw_bound(x, y, d):
    diff = np.array(x) - np.array(y)
    sd_diff = np.std(diff, ddof=1)
    return d * sd_diff

bids_root = "/projects/p52964/EAM1/data-bids/"
deriv_dir = os.path.join(bids_root, 'derivatives', 'epochs-ffr-2')

# load rater data
rater1 = pickle.load(open(os.path.join(deriv_dir, 'onset_times_kevin_2.pkl'), 'rb'))
rater2 = pickle.load(open(os.path.join(deriv_dir, 'onset_times_laura.pkl'), 'rb'))

long_active_subs = [2, 4, 6, 7, 8, 10, 12, 13, 20, 22, 25, 28, 31, 33, 34]

# Rater 1 data
active_times_by_subject_1 = {onset['sub_num']: onset['x'] for onset in rater1 if onset['task'] == 'active' and onset['sub_num'] not in long_active_subs}
passive_times_by_subject_1 = {onset['sub_num']: onset['x'] for onset in rater1 if onset['task'] == 'passive'and onset['sub_num'] not in long_active_subs}

common_subjects = sorted(set(active_times_by_subject_1.keys()) & set(passive_times_by_subject_1.keys()))
dropped_subjects = sorted(set(active_times_by_subject_1.keys()) ^ set(passive_times_by_subject_1.keys()))

print("===== Rater 1 Data =====")
print("Active onset times:", np.mean(list(active_times_by_subject_1.values())))
print("Passive onset times:", np.mean(list(passive_times_by_subject_1.values())))
print(f"Common subjects: {common_subjects}")
print(f"Dropped subjects: {dropped_subjects}")
print()

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

# Rater 2 data
active_times_by_subject_2 = {onset['sub_num']: onset['x'] for onset in rater2 if onset['task'] == 'active' and onset['sub_num'] not in long_active_subs}
passive_times_by_subject_2 = {onset['sub_num']: onset['x'] for onset in rater2 if onset['task'] == 'passive' and onset['sub_num'] not in long_active_subs}

common_subjects_2 = sorted(set(active_times_by_subject_2.keys()) & set(passive_times_by_subject_2.keys()))
dropped_subjects_2 = sorted((set(active_times_by_subject_2.keys()) ^ set(passive_times_by_subject_2.keys())) & (set(passive_times_by_subject_2.keys()) ^ set(active_times_by_subject_2.keys())))

print()
print("===== Rater 2 Data =====")
print("Active onset times:", np.mean(list(active_times_by_subject_2.values())))
print("Passive onset times:", np.mean(list(passive_times_by_subject_2.values())))
print(f"Common subjects: {common_subjects_2}")
print(f"Dropped subjects: {dropped_subjects_2}")
print()

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

# subjects for both raters
common_subjects_both = sorted(set(common_subjects) & set(common_subjects_2))
dropped_subjects_both = sorted((set(common_subjects) ^ set(common_subjects_2)) & (set(common_subjects_2) ^ set(common_subjects)))

print(f"Common subjects across raters: {common_subjects_both}")
print(f"Dropped subjects across raters: {dropped_subjects_both}")

# Differences between raters for each subject
active_diff_times = [(active_times_by_subject_1[sub] * 1000 - active_times_by_subject_2[sub] * 1000) for sub in common_subjects_both]
passive_diff_times = [(passive_times_by_subject_1[sub] * 1000 - passive_times_by_subject_2[sub] * 1000) for sub in common_subjects_both]

print()
print("Average differences:")
print(np.mean(active_diff_times))
print(np.mean(passive_diff_times))
print('% within 1 ms:', np.mean(np.abs(active_diff_times) <= 1) * 100)
print('% within 1 ms:', np.mean(np.abs(passive_diff_times) <= 1) * 100)

# Disagreement between raters 1 and 2 determined by a 5 or more ms differnce in onset time. these subjects where then looked by a third rater
# for tiebreaking. For the FFRs where all three raters disagreed, a consensus was made through a discussion between raters.
active_disagreement_idx = np.where(np.array(np.abs(active_diff_times)) >= 5)[0]
active_disagreement = [common_subjects[i] for i in active_disagreement_idx]
passive_disagreement_idx = np.where(np.array(np.abs(passive_diff_times)) >=5 )[0]
passive_disagreement = [common_subjects[i] for i in passive_disagreement_idx]

print(f"Subjects with active disagreement: {active_disagreement}")
print(f"Subjects with passive disagreement: {passive_disagreement}")

fig, ax = plt.subplots()
plt.scatter(common_subjects_both, active_diff_times, label='active')
plt.scatter(common_subjects_both, passive_diff_times, label='passive')
plt.legend()
plt.savefig('latency_diffs.png')
plt.close("all")

# Assess intra-rater reliability
rows = []
rater_df_all = pd.DataFrame()
rater_df_all['Subject'] = common_subjects_both
for sub in common_subjects:
    rows.append({'Subject': sub, 'Rater': '1', 'onset_time': active_times_by_subject_1[sub], 'trial_type': 'active'})
    rows.append({'Subject': sub, 'Rater': '1', 'onset_time': passive_times_by_subject_1[sub], 'trial_type': 'passive'})
for sub in common_subjects_2:
    rows.append({'Subject': sub, 'Rater': '2', 'onset_time': active_times_by_subject_2[sub], 'trial_type': 'active'})
    rows.append({'Subject': sub, 'Rater': '2', 'onset_time': passive_times_by_subject_2[sub], 'trial_type': 'passive'})

rater_df_all = pd.DataFrame(rows)
icc = pg.intraclass_corr(data=rater_df_all, targets='Subject', raters='Rater', ratings='onset_time', nan_policy='omit')

print()
print("Intra-class correlation (ICC) results:")
print(icc)

# Average onset times across raters for each subject and trial type
print("===== Average Onset Times Across Raters =====")
active_times_avg = [(active_times_by_subject_1[sub] + active_times_by_subject_2[sub]) / 2 for sub in common_subjects_both]
passive_times_avg = [(passive_times_by_subject_1[sub] + passive_times_by_subject_2[sub]) / 2 for sub in common_subjects_both]

active_times_avg_s = [a * 1000 for a in active_times_avg]
passive_times_avg_s = [p * 1000 for p in passive_times_avg]

# look at distributions
plt.scatter(common_subjects_both, active_times_avg_s)
plt.scatter(common_subjects_both, passive_times_avg_s)
plt.savefig('latency_distributions.png')

print("Average Active onset times:", np.mean(active_times_avg_s))
print("Average Passive onset times:", np.mean(passive_times_avg_s))

t_stat_avg, p_value_avg = ttest_rel(active_times_avg_s, passive_times_avg_s)
print(f"Paired t-test results (Average): t-statistic = {t_stat_avg:.3f}, p-value = {p_value_avg:.3f}")

# tost
equiv_range = d_to_raw_bound(active_times_avg_s, passive_times_avg_s, d=0.5)
tost_result = pg.tost(passive_times_avg_s, active_times_avg_s, paired=True, bound=equiv_range)
print(tost_result)

# bayes factor
bf_result = pg.ttest(passive_times_avg_s, active_times_avg_s, paired=True)
print(bf_result)

lat = pd.DataFrame({'sub': common_subjects_both,
                    'active': active_times_avg_s,
                    'passive': passive_times_avg_s})
