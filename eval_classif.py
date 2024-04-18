# %%
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import scipy
from sklearn.metrics import accuracy_score

# %%
path = Path("/data/parietal/store/data/HCP900/glm/fsaverage5/")
assert path.exists()

N_SUBJECTS_TRAIN = 10
N_SUBJECTS_TEST = 100
METRIC = "correlation"

# %%
# Get the list of subjects
subjects = [
    x.name
    for x in path.iterdir()
]
subjects = subjects[N_SUBJECTS_TRAIN:N_SUBJECTS_TRAIN + N_SUBJECTS_TEST]
interest = {
    'LANGUAGE': ['MATH', 'STORY'],
    'WM': ['2BK_BODY', '2BK_PLACE', '2BK_TOOL', '2BK_FACE',
           '0BK_BODY', '0BK_PLACE', '0BK_TOOL', '0BK_FACE'],
    'EMOTION': ['SHAPES', 'FACES'],
    'SOCIAL': ['TOM', 'RANDOM'],
    'RELATIONAL': ['REL', 'MATCH'],
    'GAMBLING': ['PUNISH', 'REWARD'],
    'MOTOR': ['RH', 'RF', 'T', 'LH', 'LF']
}
tasks = list(interest.keys())
hemi = 'left'

# %%
# Load subject gifti files
features = list()
for subject in tqdm(subjects):
    z_maps = list()
    save_subject = True
    for task in tasks:
        current_path = path / subject / task / "level2" / "z_maps"
        for contrast in interest[task]:
            try:
                z_map = nib.load(current_path / f"z_{contrast}_{hemi[0]}h.gii").agg_data()
                z_maps.append(z_map)
            except FileNotFoundError:
                save_subject = False
    if save_subject:
        features.append(np.array(z_maps))
features = np.array(features)
features_norm_factor = np.max(np.abs(features))
features_normalized = features / features_norm_factor

# %%
# Load euclidean and fugw barycenters
barycenter_path = Path("barycenters")
assert barycenter_path.exists()

# Load euclidean barycenter gifti files
barycenter_euc = dict()
for task in tasks:
    barycenter_path_euc = barycenter_path / "euclidean" / task / "level2" / "z_maps"
    barycenter_euc[f"{task}"] = list()
    for contrast in interest[task]:
        barycenter_euc[f"{task}"].append(nib.load(barycenter_path_euc / f"z_{contrast}_{hemi[0]}h.gii").agg_data())

# Load FUGW barycenter gifti files
# get all paths that begin with alpha_* in the directory
barycenter_paths_fugw = list(barycenter_path.glob("alpha_*"))
barycenters_fugw = dict()
for barycenter_path_fugw in barycenter_paths_fugw:
    barycenter_fugw = dict()
    for task in tasks:
        barycenter_path_fugw_task = barycenter_path_fugw / task / "level2" / "z_maps"
        barycenter_fugw[f"{task}"] = list()
        for contrast in interest[task]:
            barycenter_fugw[f"{task}"].append(nib.load(barycenter_path_fugw_task / f"z_{contrast}_{hemi[0]}h.gii").agg_data())
    barycenters_fugw[barycenter_path_fugw.name] = barycenter_fugw

# %%
# For each task-contrast, perform a K-nearest neighbors classification using the euclidean and FUGW barycenters

def evaluate_barycenter(barycenter_name, barycenter):
    results = list()
    idx = 0
    for task in tasks:
        idx_end = idx + len(interest[task])
        features_task = features[:, idx:idx_end]
        barycenter_task = barycenter[task]

        labels, predictions = list(), list()
        n_subjects = features_task.shape[0]
        for i in range(n_subjects):
            features_task_subject = features_task[i]
            dist = scipy.spatial.distance.cdist(
                features_task_subject, barycenter_task, METRIC)
            predictions.append(np.argmin(dist, axis=1))
            labels.append(np.arange(features_task_subject.shape[0]))
        labels = np.array(labels).flatten()
        predictions = np.array(predictions).flatten()

        results.append([barycenter_name, f"{task}", accuracy_score(predictions, labels) * 100])
        idx += len(interest[task])
    return results

results = evaluate_barycenter("euclidean", barycenter_euc)
for barycenter_name, barycenter in tqdm(barycenters_fugw.items()):
    results += evaluate_barycenter(barycenter_name, barycenter)

# %%
df = pd.DataFrame(results, columns=['Barycenter', 'Task', 'val'])

# %%
df_euc = df[df['Barycenter'] == 'euclidean']
df_fugw = df[df['Barycenter'].str.contains("alpha_")]
df_fugw = df_fugw.groupby('Task').agg(
    {'Barycenter': 'last', 'val': 'max'}).reset_index()

# %%
df_fugw = df_fugw.merge(df, on=['Task', 'val'], how='inner')
df_fugw = df_fugw.drop_duplicates(subset='Task', keep='first')
df_fugw.reset_index(drop=True, inplace=True)
df_fugw['Barycenter'] = 'fugw'

# %%
df = pd.concat([df_euc, df_fugw])
df.reset_index(drop=True, inplace=True)

# %%
# Plot
df.pivot(columns='Barycenter', values='val', index='Task').plot(kind='bar')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
plt.suptitle(METRIC)

# %%
