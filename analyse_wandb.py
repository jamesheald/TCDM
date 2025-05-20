import wandb
import pandas as pd

api = wandb.Api()

# run = api.run('james-gatsby/dummy_proj/52l0hj4m')
# history = run.history()
# print(history.columns)
# Index(['train/stdz', 'train/explained_variance', '_step', 'env/obj_success',
#        'train/clip_range', 'rollout/ep_len_mean', '_runtime',
#        'train/value_loss', 'train/entropy_loss', 'train/loss', 'env/time_frac',
#        'train/std', 'train/learning_rate', 'env/obj_err_scale',
#        'train/policy_gradient_loss', 'env/obj_err', 'train/dynamics_loss',
#        'rollout/ep_rew_mean', 'train/clip_fraction', 'time/fps', 'global_step',
#        'train/approx_kl', '_timestamp', 'env/step_obj_err',
#        'eval/rollout_video', 'eval/mean_reward', 'eval/mean_step_obj_err',
#        'eval/time', 'eval/mean_obj_err', 'eval/mean_length',
#        'eval/mean_obj_err_scale', 'eval/mean_time_frac',
#        'eval/mean_obj_success'],
#       dtype='object')
# history["global_step"]
# history["rollout/ep_rew_mean"]

# to get each row, not downsampled
# import pandas as pd
# data = []
# for row in run.scan_history():
#     data.append(row)
# df = pd.DataFrame(data)

runids = []

runids.extend([
# objex - diag entropy touch_table, ObjCvelForceTable, nEpochs40
'ftxi63se',
'nxuug5u3',
'1tasnrpw',
'v8nw06b2',
'dz4900xg',
'q5s7aa4f',
'l2jywkm3',
'9p2ox8p8',
'bomoenls',
'yeu6pmto',
'89ucjht7',
'jxywagr4',
'j3f8x5ae',
'35pecx2o',
'rxwde54m',
# objex - diag entropy always
# finished
# '52l0hj4m',
'5c579czj',
'frgd1db8',
'tynoy6zy',
'hwnalngp',
'lilsjhbj',
# still running
'8sse1iu8', # (lyric-firefly-819) CRASHED/TERMINATED EARLY?
'p466d32f', # (playful-armadillo-844)
'f0ucdppp', # (revived-snow-845)
# objex - diag entropy pre-touching only, GS true and False
'ppxli3ot',
'sbefnesx',
'22zcgi2v',
'i4awt0sd',
'rhkfnx5g',
'v6sc66ub',
'c13px9uq',
'bwvt12kh',
'jrld3c7k',
'creu1g8z',
'qloyzz33',
'wve66dl2',
'72it3d72',
# still running
'f66u1g7b', # (vague-armadillo-834)
# standard ppo
# 3fhg7qet # same experiment type as below for standard_PPO hammer-use1 (but better)
'wb9ehj9a',
'caltv0ec',
'sf4o4a86',
'afmfmp5q',
'tvrdftp6',
'3pfnuhz8',
'jbzuhfpv',
'doichnvp', # (no progress - entropy only optimized)
'e72p8oqp',
# still running
'i43fi8f0',
'z9tkl1jf',
'q7nl6wnk',
'7knmabgo',
'ldlmpjlu',
'd31guvkg']) # (dulcet-voice-812) CRASHED/TERMINATED EARLY?

training_data = {}

def add_to_dict(D, key, history=None):
    if key not in D:
        D[key] = {}
    if history is not None:
        D[key] = history[key]
    return D

for runid in runids:
    run = api.run(f'james-gatsby/dummy_proj/{runid}')
    env_name = run.config['env']['name']
    # if env_name != "mug-drink3":
    # if env_name != "flashlight-on2" and env_name != "mug-drink3":
    # if env_name != "flashlight-on2":
    history = run.history()
    training_data = add_to_dict(training_data, env_name)
    if run.config['agent']['standard_PPO']:
        algo_name = 'PPO'
    elif run.config['agent']['controlled_variables'] == 'ObjCvelForceTable':
        if run.config['agent']['params']['diagonal_entropy'] == 'touch_table' \
            and run.config['agent']['params']['dynamics_n_epochs'] == 40:
            algo_name = 'OBJEX_table_40'
    else:
        if run.config['agent']['params']['diagonal_entropy_when_touching']:
            algo_name = 'OBJEX_always'
        else:
            algo_name = 'OBJEX_touching'
        if 'use_gram_schmidt' in run.config['agent']['params']:
            gram_schmidt = run.config['agent']['params']['use_gram_schmidt']
        else:
            gram_schmidt = True
        algo_name += f'_GS{gram_schmidt}'
    training_data[env_name] = add_to_dict(training_data[env_name], algo_name)
    training_data[env_name][algo_name] = add_to_dict(training_data[env_name][algo_name], 'global_step', history)
    training_data[env_name][algo_name] = add_to_dict(training_data[env_name][algo_name], 'rollout/ep_rew_mean', history)
    training_data[env_name][algo_name] = add_to_dict(training_data[env_name][algo_name], 'env/obj_success', history)
    training_data[env_name][algo_name] = add_to_dict(training_data[env_name][algo_name], 'env/step_obj_err', history)

# for env in training_data.keys(): print(env,training_data[env]['PPO']['global_step'].max()/1e7)

from collections import defaultdict

algo_env_counts = defaultdict(int)

for env_name, env_data in training_data.items():
    for algo_name in env_data.keys():
        algo_env_counts[algo_name] += 1

# Print results
for algo_name, count in algo_env_counts.items():
    print(f"{algo_name}: {count} environments")

from matplotlib import pyplot as plt

colors = {'OBJEX_always_GSTrue': 'r',
          'OBJEX_touching_GSTrue': 'g',
          'OBJEX_touching_GSFalse': 'b',
          'OBJEX_table_40': 'c',
          'PPO': 'k'}

for env_name, env_data in training_data.items():
    fig, axs = plt.subplots(1, 3, figsize=(12, 5))  # 3 rows, 1 column
    fig.suptitle(f"{env_name}", fontsize=16)
    for algo_name, algo_data in env_data.items():

        # Plot data on each subplot
        axs[0].plot(algo_data['global_step'], algo_data['rollout/ep_rew_mean'], c=colors[algo_name],label=f"{algo_name}")
        axs[0].set_title("ep_rew_mean")
        axs[0].legend()

        axs[1].plot(algo_data['global_step'], algo_data['env/obj_success'], c=colors[algo_name], label=f"{algo_name}")
        axs[1].set_title("obj_success")
        axs[1].set_ylim(0, 1)
        axs[1].legend()

        axs[2].plot(algo_data['global_step'], algo_data['env/step_obj_err'], c=colors[algo_name], label=f"{algo_name}")
        axs[2].set_title("step_obj_err")
        axs[2].legend()

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig(f"TCDM_{env_name}.png")

    # Close the figure
    plt.close(fig)

from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# Constants
num_bins = 50
step_min = 0
step_max = 50_000_000
bin_edges = np.linspace(step_min, step_max, num_bins + 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# Accumulate binned results for each algorithm across environments
algo_feature_bins = defaultdict(lambda: defaultdict(list))

for env_name, env_data in training_data.items():
    for algo_name, algo_data in env_data.items():
        global_steps = np.array(algo_data['global_step'])  # shape: (T,)
        bin_indices = np.digitize(global_steps, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)

        for feature_name, values in algo_data.items():
            if feature_name == 'global_step':
                continue
            values = np.array(values)

            bin_sums = np.zeros(num_bins)
            bin_counts = np.zeros(num_bins)

            for idx, val in zip(bin_indices, values):
                bin_sums[idx] += val
                bin_counts[idx] += 1

            bin_means = np.divide(
                bin_sums, bin_counts,
                out=np.full_like(bin_sums, np.nan),
                where=bin_counts > 0
            )

            algo_feature_bins[algo_name][feature_name].append(bin_means)

# Final averaging and SEM
final_result = {}

for algo_name, features in algo_feature_bins.items():
    final_result[algo_name] = {}
    for feature_name, all_env_bin_means in features.items():
        stacked = np.stack(all_env_bin_means, axis=0)  # shape: (num_envs, num_bins)
        
        # Compute mean and SEM
        mean = np.nanmean(stacked, axis=0)
        std = np.nanstd(stacked, axis=0)
        count = np.sum(~np.isnan(stacked), axis=0)
        sem = np.divide(
            std, np.sqrt(count),
            out=np.full_like(std, np.nan),
            where=count > 0
        )

        final_result[algo_name][feature_name] = {
            'mean': mean,
            'sem': sem  # NEW: store SEM
        }

# Store bin centers
final_result["bin_centers"] = bin_centers

# Plotting
fig, axs = plt.subplots(1, 3, figsize=(12, 5))
fig.suptitle("all_data", fontsize=16)

# for algo_name in ['OBJEX_always_GSTrue', 'OBJEX_touching_GSTrue', 'PPO', 'OBJEX_touching_GSFalse', 'OBJEX_table_40']:
# for algo_name in ['OBJEX_touching_GSTrue', 'PPO', 'OBJEX_touching_GSFalse', 'OBJEX_table_40']:
for algo_name in ['PPO', 'OBJEX_table_40']:
    for ax, feature_key, title in zip(
        axs,
        ['rollout/ep_rew_mean', 'env/obj_success', 'env/step_obj_err'],
        ['ep_rew_mean', 'obj_success', 'step_obj_err']
    ):
        mean = final_result[algo_name][feature_key]['mean']
        sem = final_result[algo_name][feature_key]['sem']
        x = final_result["bin_centers"]

        # Plot mean line
        ax.plot(x, mean, label=algo_name, color=colors[algo_name])

        # Plot shaded region for ±SEM
        ax.fill_between(
            x,
            mean - sem,
            mean + sem,
            color=colors[algo_name],
            alpha=0.3,
            linewidth=0
        )

        ax.set_title(title)
        ax.legend()
        if title == "obj_success":
            ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig("TCDM_All_Envs.png")
plt.close(fig)

# print(run.config['agent']['std_network'])