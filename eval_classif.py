# %%
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import scipy
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier

# %%
path = Path("/data/parietal/store/data/HCP900/glm/fsaverage5/")
assert path.exists()

N_SUBJECTS_TRAIN = 10
N_SUBJECTS_TEST = 100
ALPHA = 0.5
RHO = 1
EPS = 1e-4

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

barycenter_euc, barycenter_fugw = list(), list()
for task in tasks:
    barycenter_path_euc = barycenter_path / "euclidean" / task / "level2" / "z_maps"
    barycenter_path_fugw = barycenter_path / f"alpha_{ALPHA}_rho_{RHO}_eps_{EPS}"
    barycenter_path_fugw = barycenter_path_fugw / task / "level2" / "z_maps"
    for contrast in interest[task]:
        # Load euclidean barycenter gifti file
        barycenter_euc.append(nib.load(barycenter_path_euc / f"z_{contrast}_{hemi[0]}h.gii").agg_data())

        # Load fugw barycenter gifti file
        barycenter_fugw.append(nib.load(barycenter_path_fugw / f"z_{contrast}_{hemi[0]}h.gii").agg_data())
barycenter_euc = np.array(barycenter_euc)
barycenter_fugw = np.array(barycenter_fugw)

# %%
# For each task-contrast, perform a K-nearest neighbors classification using the euclidean and FUGW barycenters
results = list()
idx = 0
for task in tasks:
    idx_end = idx + len(interest[task])
    features_task = features[:, idx:idx_end]
    barycenter_euc_task = barycenter_euc[idx:idx_end]
    barycenter_fugw_task = barycenter_fugw[idx:idx_end]

    labels, predictions_euc, predictions_fugw = list(), list(), list()
    for i in range(len(features_task)):
        features_task_subject = features_task[i]
        dist_euc = scipy.spatial.distance.cdist(
            features_task_subject, barycenter_euc_task, metric='correlation')
        dist_fugw = scipy.spatial.distance.cdist(
            features_task_subject, barycenter_fugw_task, metric='correlation')
        predictions_euc.append(np.argmin(dist_euc, axis=1))
        predictions_fugw.append(np.argmin(dist_fugw, axis=1))
        labels.append(np.arange(features_task_subject.shape[0]))
    labels = np.array(labels).flatten()
    predictions_euc = np.array(predictions_euc).flatten()
    predictions_fugw = np.array(predictions_fugw).flatten()
    predictions_dummy = DummyClassifier().fit(None, labels).predict(np.arange(len(labels)))

    results.append(['Euclidean', f"{task}", accuracy_score(predictions_euc, labels)])
    results.append(['FUGW', f"{task}", accuracy_score(predictions_fugw, labels)])
    # results.append(['Dummy classifier', f"{task}", accuracy_score(predictions_dummy, labels)])
    idx += len(interest[task])

# %%
# Plot
df = pd.DataFrame(results, columns=['Barycenter', 'Task', 'val'])
df.pivot(columns='Barycenter', values='val', index='Task').plot(kind='bar')
plt.ylabel('Accuracy (%)')

# %%
