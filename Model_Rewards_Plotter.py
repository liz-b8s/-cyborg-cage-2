import numpy as np
import os
import glob
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True

# Global graph styles set:
sns.set_style('whitegrid') # darkgrid, white grid, dark, white and ticks
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
plt.rc('legend', fontsize=13)    # legend fontsize
plt.rc('font', size=13)          # controls default text sizes
sns.color_palette('deep')
plt.figure(figsize=(8,6), tight_layout=True)

disprop_scale = []
norm_mag = []
scaleup_mag = []
icm_b = []
icm_m = []
baseline_m = []
baseline_b = []
small_pos_b = []
small_pos_m = []
big_pos_b = []
big_pos_m = []
curiosity_a = []
curiosity_b = []


for i in range(8):
    i_new = str(i+1)
    # Baseline paths
    path_baseline_m = '/Model_rewards/Reward_shaping_vm/RS_baseline_meander_vm_' + i_new + '/Reward_shaping_real_iter_0'
    path_baseline_b = '/Model_rewards/Reward_shaping_vm/RS_baseline_vm__' + i_new + '/Reward_shaping_real_iter_0'

    # Experiment 1 paths (first set meander, 2nd bline, uncomment the relevant ones)
    path_disprop = '/Model_rewards/Reward_shaping_vm/RS_disprop_vm_meander_' + i_new + '/Reward_shaping_real_iter_0'
    path_norm = '/Model_rewards/Reward_shaping_vm/RS_norm_vm_meander_' + i_new + '/Reward_shaping_real_iter_0'
    path_scaleup = '/Model_rewards/Reward_shaping_vm/RS_scaleup_vm_meander_' + i_new + '/Reward_shaping_real_iter_0'

    # path_disprop = '/Model_rewards/Reward_shaping_vm/RS_disprop_scale_mag_vm_' + i_new + '/Reward_shaping_real_iter_0'
    # path_norm = '/Model_rewards/Reward_shaping_vm/RS_norm_mag_vm_' + i_new + '/Reward_shaping_real_iter_0'
    # path_scaleup = '/Model_rewards/Reward_shaping_vm/RS_scaleup_mag_vm_' + i_new + '/Reward_shaping_real_iter_0'

    path_new_cur_B = '/Model_rewards/Reward_shaping_vm/Even_Sparser_ICM_D_' + i_new + '/Reward_shaping_real_iter_0'
    path_new_cur_A = '/Model_rewards/Reward_shaping_vm/Sparse_ICM_D_' + i_new + '/Reward_shaping_real_iter_0'

    # Experiment 2 paths
    path_small_pos_b = '/Model_rewards/Reward_shaping_vm/RS_small_pos_vm_' + i_new + '/Reward_shaping_real_iter_0'
    path_small_pos_m = '/Model_rewards/Reward_shaping_vm/RS_small_pos_vm_meander_' + i_new + '/Reward_shaping_real_iter_0'
    path_big_pos_b = '/Model_rewards/Reward_shaping_vm/RS_big_pos_vm_' + i_new + '/Reward_shaping_real_iter_0'
    path_big_pos_m = '/Model_rewards/Reward_shaping_vm/RS_big_pos_meander_vm_' + i_new + '/Reward_shaping_real_iter_0'

    # Experiment 3 paths
    path_ICM_m = '/Model_rewards/Reward_shaping_vm/RS_ICM_Meander_' + i_new + '/Reward_shaping_real_iter_0'
    path_ICM_b = '/Model_rewards/Reward_shaping_vm/RS_ICM_' + i_new + '/Reward_shaping_real_iter_0'

    # Appending paths to array
    # baseline_m.append(path_baseline_m)
    baseline_b.append(path_baseline_b)

    disprop_scale.append(path_disprop)
    norm_mag.append(path_norm)
    scaleup_mag.append(path_scaleup)

    big_pos_b.append(path_big_pos_b)
    small_pos_b.append(path_small_pos_b)
    small_pos_m.append(path_small_pos_m)
    big_pos_m.append(path_big_pos_m)

    icm_b.append(path_ICM_b)
    icm_m.append(path_ICM_m)

    curiosity_a.append(path_new_cur_A)
    curiosity_b.append(path_new_cur_B)

# To plot on the x axis
x_vals = list(range(0, 50000, 10))

### Baseline ###

# Baseline_m
df_array_bl = []
df_concat_bl = pd.DataFrame()
for rewards_f in baseline_m:
    df_new = pd.read_csv(f"{rewards_f}")
    df_concat_bl = pd.concat((df_concat_bl, df_new))
    df_array_bl.append(df_new)

by_row_index_bl = df_concat_bl.groupby(df_concat_bl.index)
df_means_bl = by_row_index_bl.mean()

# Baseline_b
df_array_bl_b = []
df_concat_bl_b = pd.DataFrame()
for rewards_f in baseline_b:
    df_new = pd.read_csv(f"{rewards_f}")
    df_concat_bl_b = pd.concat((df_concat_bl_b, df_new))
    df_array_bl_b.append(df_new)

by_row_index_bl_b = df_concat_bl_b.groupby(df_concat_bl_b.index)
df_means_bl_b = by_row_index_bl_b.mean()

### Exp1 ###

# Normalised
df_array_norm = []
df_concat_norm = pd.DataFrame()
for rewards_f in norm_mag:
    df_new = pd.read_csv(f"{rewards_f}")
    df_concat_norm = pd.concat((df_concat_norm, df_new))
    df_array_norm.append(df_new)

by_row_index_norm = df_concat_norm.groupby(df_concat_norm.index)
df_means_norm = by_row_index_norm.mean()
# print(df_means_norm)

# scaled up
df_array_scaleup = []
df_concat_scaleup = pd.DataFrame()
for rewards_f in scaleup_mag:
    df_new = pd.read_csv(f"{rewards_f}")
    df_concat_scaleup = pd.concat((df_concat_scaleup, df_new))
    df_array_scaleup.append(df_new)

by_row_index_scaleup = df_concat_scaleup.groupby(df_concat_scaleup.index)
df_means_scaleup = by_row_index_scaleup.mean()
# print(df_means_scaleup)

# disproportionally scaled up
df_array_disprop = []
df_concat_disprop = pd.DataFrame()
for rewards_f in disprop_scale:
    df_new = pd.read_csv(f"{rewards_f}")
    df_concat_disprop = pd.concat((df_concat_disprop, df_new))
    df_array_disprop.append(df_new)

by_row_index_disprop = df_concat_disprop.groupby(df_concat_disprop.index)
df_means_disprop = by_row_index_disprop.mean()
# print(df_means_norm)

### Exp2 ###

# small pos b
df_array_sposb = []
df_concat_sposb = pd.DataFrame()
for rewards_f in small_pos_b:
    df_new = pd.read_csv(f"{rewards_f}")
    df_concat_sposb = pd.concat((df_concat_sposb, df_new))
    df_array_sposb.append(df_new)

by_row_index_sposb = df_concat_sposb.groupby(df_concat_sposb.index)
df_means_sposb = by_row_index_sposb.mean()

# big pos b
df_array_bposb = []
df_concat_bposb = pd.DataFrame()
for rewards_f in big_pos_b:
    df_new = pd.read_csv(f"{rewards_f}")
    df_concat_bposb = pd.concat((df_concat_bposb, df_new))
    df_array_bposb.append(df_new)

by_row_index_bposb = df_concat_bposb.groupby(df_concat_bposb.index)
df_means_bposb = by_row_index_bposb.mean()

# small pos m
df_array_sposm = []
df_concat_sposm = pd.DataFrame()
for rewards_f in small_pos_m:
    df_new = pd.read_csv(f"{rewards_f}")
    df_concat_sposm = pd.concat((df_concat_sposm, df_new))
    df_array_sposm.append(df_new)

by_row_index_sposm = df_concat_sposm.groupby(df_concat_sposm.index)
df_means_sposm = by_row_index_sposm.mean()

# big pos m
df_array_bposm = []
df_concat_bposm = pd.DataFrame()
for rewards_f in big_pos_m:
    df_new = pd.read_csv(f"{rewards_f}")
    df_concat_bposm = pd.concat((df_concat_bposm, df_new))
    df_array_bposm.append(df_new)

by_row_index_bposm = df_concat_bposm.groupby(df_concat_bposm.index)
df_means_bposm = by_row_index_bposm.mean()

### Exp3 ###

# icm_b
df_array_icmb = []
df_concat_icmb = pd.DataFrame()
for rewards_f in icm_b:
    df_new = pd.read_csv(f"{rewards_f}")
    df_concat_icmb = pd.concat((df_concat_icmb, df_new))
    df_array_icmb.append(df_new)

by_row_index_icmb = df_concat_icmb.groupby(df_concat_icmb.index)
df_means_icmb = by_row_index_icmb.mean()

# icm_m
df_array_icmm = []
df_concat_icmm = pd.DataFrame()
for rewards_f in icm_m:
    df_new = pd.read_csv(f"{rewards_f}")
    df_concat_icmm = pd.concat((df_concat_icmm, df_new))
    df_array_icmm.append(df_new)

by_row_index_icmm = df_concat_icmm.groupby(df_concat_icmm.index)
df_means_icmm = by_row_index_icmm.mean()

# curiosity_a
df_array_icmA = []
df_concat_icmA = pd.DataFrame()
for rewards_f in curiosity_a:
    df_new = pd.read_csv(f"{rewards_f}")
    df_concat_icmA = pd.concat((df_concat_icmA, df_new))
    df_array_icmA.append(df_new)

by_row_index_icmA = df_concat_icmA.groupby(df_concat_icmA.index)
df_means_icmA = by_row_index_icmA.mean()

# curiosity_b
df_array_icmB = []
df_concat_icmB = pd.DataFrame()
for rewards_f in curiosity_b:
    df_new = pd.read_csv(f"{rewards_f}")
    df_concat_icmB = pd.concat((df_concat_icmB, df_new))
    df_array_icmB.append(df_new)

by_row_index_icmB = df_concat_icmB.groupby(df_concat_icmB.index)
df_means_icmB = by_row_index_icmB.mean()

### Plotting ###

# Plotting the curves all on one graph. Uncomment and recomment the parts that are relevant to your plt.

# for mean_val in [df_means_bl]:
#     plt.plot(x_vals, mean_val[:5000], label='Meander Baseline')


for mean_val in [df_means_icmb]:
    plt.plot(x_vals, mean_val[:5000], label='Bline ICM - rewards 0, -0.1, -1.0, -10')

for mean_val in [df_means_bl_b]:
    plt.plot(x_vals, mean_val[:5000], label='Bline Baseline')

for mean_val in [df_means_icmA]:
    plt.plot(x_vals, mean_val[:5000], label='CuriosityA - rewards 0, 0, -1.0, -10')

for mean_val in [df_means_icmB]:
    plt.plot(x_vals, mean_val[:5000], label='CuriosityB - rewards 0, 0, 0, -10')

# for mean_val in [df_means_sposm]:
#     plt.plot(y_vals, mean_val[:5000], label='Small Positive Reward')
#
# for mean_val in [df_means_bposm]:
#     plt.plot(y_vals, mean_val[:5000], label='Large Positive Reward')
#

# for mean_val in [df_means_norm]:
#     plt.plot(y_vals, mean_val[:5000], label='Normalised Rewards')
#
# for mean_val in [df_means_scaleup]:
#     plt.plot(y_vals, mean_val[:5000], label='Scaled Up Rewards')
#
# for mean_val in [df_means_disprop]:
#     plt.plot(y_vals, mean_val[:5000], label='Disproportionally Scaled Up Rewards')



# for mean_val in [df_means_icmm]:
#     plt.plot(x_vals, mean_val[:5000], label='Meander ICM')


plt.xlabel('Episodes of Training')
plt.ylabel('Average Score')
plt.ylim(-250, 0)
plt.legend(loc='right')

plt.savefig('figures/CurisoityAB_learning_curves.png')

plt.show()

