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
deriv_dir = os.path.join(bids_root, 'derivatives', 'epochs-ffr')

# load rater data
rater1 = pickle.load(open(os.path.join(deriv_dir, 'onset_times_kevin.pkl'), 'rb'))
rater2 = pickle.load(open(os.path.join(deriv_dir, 'onset_times_laura_3.pkl'), 'rb'))
rater3 = pickle.load(open(os.path.join(deriv_dir, 'onset_times_tiebreaker_2.pkl'), 'rb'))
consensus34 = pickle.load(open(os.path.join(deriv_dir, 'onset_times_34.pkl'), 'rb'))

# Rater 1 data
active_times_by_subject_1 = {onset['sub_num']: onset['x'] for onset in rater1 if onset['task'] == 'active'}
passive_times_by_subject_1 = {onset['sub_num']: onset['x'] for onset in rater1 if onset['task'] == 'passive'}

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
active_times_by_subject_2 = {onset['sub_num']: onset['x'] for onset in rater2 if onset['task'] == 'active'}
passive_times_by_subject_2 = {onset['sub_num']: onset['x'] for onset in rater2 if onset['task'] == 'passive'}

common_subjects_2 = sorted(set(active_times_by_subject_2.keys()) & set(passive_times_by_subject_2.keys()))
dropped_subjects_2 = sorted(set(active_times_by_subject_2.keys()) ^ set(passive_times_by_subject_2.keys()))

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

# Tiebreaker data - used for peaks that had more than 5 ms disagreement (half a cycle difference)
active_times_by_subject_3 = {onset['sub_num']: onset['x'] for onset in rater3 if onset['task'] == 'active'}
passive_times_by_subject_3 = {onset['sub_num']: onset['x'] for onset in rater3 if onset['task'] == 'passive'}

common_subjects_active_3 = [s for s in set(active_times_by_subject_3.keys())]
common_subjects_passive_3 = [s for s in set(passive_times_by_subject_3.keys())]
# dropped_subjects_3 = sorted(set(active_times_by_subject_3.keys()) ^ set(passive_times_by_subject_3.keys()))

active_times_3 = [active_times_by_subject_3[sub] for sub in common_subjects_active_3]
passive_times_3 = [passive_times_by_subject_3[sub] for sub in common_subjects_passive_3]

# Active subject 34 was agreed upon by a consensus amongst raters
active_times_by_subejct_c = {34: onset['x'] for onset in consensus34 if onset['task'] == 'active'}

# Differences between raters for each subject
active_diff_times = [(active_times_by_subject_1[sub] * 1000 - active_times_by_subject_2[sub] * 1000) for sub in common_subjects]
passive_diff_times = [(passive_times_by_subject_1[sub] * 1000 - passive_times_by_subject_2[sub] * 1000) for sub in common_subjects]

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
print()
print(f"Subjects with active disagreement: {active_disagreement}")
print(f"Subjects with passive disagreement: {passive_disagreement}")
print()
fig, ax = plt.subplots()
plt.scatter(common_subjects, active_diff_times, label='active')
plt.scatter(common_subjects, passive_diff_times, label='passive')
plt.axhline(5)
plt.axhline(-5)
plt.legend()
plt.savefig('latency_diffs.png')
plt.close("all")

long_actives_2 = len(np.where(np.array(active_times_2) >= 0.013)[0]) 
long_actives_1 = len(np.where(np.array(active_times) >= 0.013)[0])

long_passives_2 = len(np.where(np.array(passive_times_2) >= 0.013)[0]) 
long_passives_1 = len(np.where(np.array(passive_times) >= 0.013)[0])
print()
print(f"Long actives, rater 1: {long_actives_1} and rater 2: {long_actives_2}")
print(f"Long passives, rater 1: {long_passives_1} and rater 2: {long_passives_2}")
print()
print(f"Tiebreaker subjects: active = {common_subjects_active_3} and passive = {common_subjects_passive_3}")
active_diff_times_ties_1 = [(active_times_by_subject_1[sub] * 1000 - active_times_by_subject_3[sub] * 1000) for sub in common_subjects_active_3]
active_diff_times_ties_2 = [(active_times_by_subject_2[sub] * 1000 - active_times_by_subject_3[sub] * 1000) for sub in common_subjects_active_3]
passive_diff_times_ties_1 = [(passive_times_by_subject_1[sub] * 1000 - passive_times_by_subject_3[sub] * 1000) for sub in common_subjects_passive_3]
passive_diff_times_ties_2 = [(passive_times_by_subject_2[sub] * 1000 - passive_times_by_subject_3[sub] * 1000) for sub in common_subjects_passive_3]

plt.scatter(common_subjects_active_3, active_diff_times_ties_1, label="Active Rater 1")
plt.scatter(common_subjects_active_3, active_diff_times_ties_2, label="Active Rater 2")
plt.ylabel('ms difference')
plt.xlabel('subject')
plt.legend()
plt.savefig("latency_ties_active.png")
plt.close("all")

plt.scatter(common_subjects_passive_3, passive_diff_times_ties_1, label="Passive Rater 1")
plt.scatter(common_subjects_passive_3, passive_diff_times_ties_2, label="Passive Rater 2")
plt.ylabel('ms difference')
plt.xlabel('subject')
plt.legend()
plt.savefig("latency_ties_passive.png")
plt.close("all")

# Assess intra-rater reliability
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

print()
print("Intra-class correlation (ICC) results:")
print(icc)

# Average onset times across raters for each subject and trial type
print("===== Average Onset Times Across Raters =====")
active_times_avg = [(active_times_by_subject_1[sub] + active_times_by_subject_2[sub]) / 2 for sub in common_subjects]
passive_times_avg = [(passive_times_by_subject_1[sub] + passive_times_by_subject_2[sub]) / 2 for sub in common_subjects]

active_times_avg_s = [a * 1000 for a in active_times_avg]
passive_times_avg_s = [p * 1000 for p in passive_times_avg]

# look at distributions
plt.scatter(common_subjects, active_times_avg_s)
plt.scatter(common_subjects, passive_times_avg_s)
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

lat = pd.DataFrame({'sub': common_subjects,
                    'active': active_times_avg_s,
                    'passive': passive_times_avg_s})
# print(lat.sort_values('passive', ascending=False).round(2))
# print(lat.sort_values('active', ascending=False).round(2))

# Average onset times across raters, including tiebreaks and consensus
# active_tiebreaker_subs = [5, 31]
# passive_tiebreaker_subs = [14, 21, 25]

active_rater1_win_subs = [5, 21, 31]
active_rater2_win_subs = [17, 19]
passive_rater1_win_subs = [14, 21, 25, 31]
passive_rater2_win_subs = [17]

active_sub_exceptions =  active_rater1_win_subs + [34]
passive_sub_exceptions =  passive_rater1_win_subs


if len(common_subjects) < len(common_subjects_2):
    subs_to_iter = common_subjects
else:
    subs_to_iter = common_subjects_2

active_times_avg_new = []
for sub in subs_to_iter:
    if sub in active_rater1_win_subs:
        active_times_avg_new.append(active_times_by_subject_1[sub])
    elif sub in active_rater2_win_subs:
        active_times_avg_new.append(active_times_by_subject_2[sub])
    elif sub == 34:
        active_times_avg_new.append(active_times_by_subejct_c[sub])
    else:
        active_times_avg_new.append((active_times_by_subject_1[sub] + active_times_by_subject_2[sub]) / 2)

passive_times_avg_new = []
for sub in subs_to_iter:
    if sub in passive_rater1_win_subs:
        passive_times_avg_new.append(passive_times_by_subject_1[sub])
    elif sub in passive_rater2_win_subs:
        passive_times_avg_new.append(passive_times_by_subject_2[sub])
    else:
        passive_times_avg_new.append((passive_times_by_subject_1[sub] + passive_times_by_subject_2[sub]) / 2)

active_times_avg_new_ms = [a * 1000 for a in active_times_avg_new]
passive_times_avg_new_ms = [p * 1000 for p in passive_times_avg_new]

print()
print("=== Average Onset Times Across Raters (tiebreaks) ===")
print("Average Active onset times:", np.mean(active_times_avg_new_ms))
print("Average Passive onset times:", np.mean(passive_times_avg_new_ms))
t_stat_avg, p_value_avg = ttest_rel(active_times_avg_new_ms, passive_times_avg_new_ms)
print(f"Paired t-test results (Average): t-statistic = {t_stat_avg:.3f}, p-value = {p_value_avg:.3f}")

# tost
equiv_range = d_to_raw_bound(active_times_avg_new_ms, passive_times_avg_new_ms, d=0.5)
tost_result = pg.tost(passive_times_avg_new_ms, active_times_avg_new_ms, paired=True, bound=equiv_range)
print(tost_result)

# bayes factor
bf_result = pg.ttest(passive_times_avg_new_ms, active_times_avg_new_ms, paired=True)
print(bf_result)

bins = np.arange(np.floor(min(active_times_avg_new_ms + passive_times_avg_new_ms)),
                 np.ceil(max(active_times_avg_new_ms + passive_times_avg_new_ms)) + 1, 1)  # 1 ms bins

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.hist(active_times_avg_new_ms, bins=bins, alpha=0.6, label='active')
ax.hist(passive_times_avg_new_ms, bins=bins, alpha=0.6, label='passive')
ax.set_xlabel('Onset latency (ms)')
ax.set_ylabel('Number of subjects')
ax.legend()
plt.tight_layout()
plt.savefig('latency_hist_tiebreaks.png')
plt.show()

lat = pd.DataFrame({'sub': subs_to_iter,
                    'active': active_times_avg_new_ms,
                    'passive': passive_times_avg_new_ms})
# print(lat.sort_values('passive', ascending=False).round(2))
# print(lat.sort_values('active', ascending=False).round(2))