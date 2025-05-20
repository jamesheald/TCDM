import os
import re

def classify_run_paths(base_dir, allowed_runids):
    to_delete = []
    to_keep = []

    for date_folder in os.listdir(base_dir):
        date_path = os.path.join(base_dir, date_folder)
        if not os.path.isdir(date_path):
            continue

        for time_folder in os.listdir(date_path):
            time_path = os.path.join(date_path, time_folder)
            if not os.path.isdir(time_path):
                continue

            wandb_dir = os.path.join(time_path, "wandb")
            if not os.path.isdir(wandb_dir):
                continue

            # Search for subdir in wandb/ matching run-...-runid pattern
            for run_dir in os.listdir(wandb_dir):
                if run_dir.startswith("run-"):
                    runid_found =run_dir.split("-")[-1]
                    if runid_found in allowed_runids:
                        to_keep.append(time_path)
                    else:
                        to_delete.append(time_path)
                    break  # Assume one run dir per time_path; stop looking further
                # match = re.match(r"run-\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}-(.+)", run_dir)
                # if match:
                #     runid_found = match.group(1)
                #     if runid_found in allowed_runids:
                #         to_keep.append(time_path)
                #     else:
                #         to_delete.append(time_path)
                #     break  # Assume one run dir per time_path; stop looking further

    return to_delete, to_keep

allowed_runids = []
allowed_runids.extend([
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
# objex - diag entropy always
# finished
'52l0hj4m',
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
'3fhg7qet', # same experiment type as below for standard_PPO hammer-use1 (but better)
'wb9ehj9a',
'caltv0ec',
'sf4o4a86',
'afmfmp5q',
'tvrdftp6',
'3pfnuhz8',
'jbzuhfpv',
'doichnvp', # (no progress - entropy only optimized)
# still running
'e72p8oqp', # (scarlet-wildflower-822) CRASHED/TERMINATED EARLY?
'd31guvkg']) # (dulcet-voice-812) CRASHED/TERMINATED EARLY?

base_dir = "/nfs/nhome/live/jheald/TCDM/outputs"

to_delete, to_keep = classify_run_paths(base_dir, allowed_runids)

print("Paths to delete:")
for path in to_delete:
    print(path)

print("\nPaths to keep:")
for path in to_keep:
    print(path)

print(len(to_delete))
print(len(allowed_runids) == len(to_keep))
print(len(allowed_runids))
print(len(to_keep))

breakpoint()

# import shutil
# for time_path in to_delete: shutil.rmtree(time_path)