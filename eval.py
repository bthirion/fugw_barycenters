#%%
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm

#%%
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
    'LANGUAGE': ['MATH-STORY'],
    'WM': ['2BK-0BK', 'BODY-AVG', 'PLACE-AVG', 'TOOL-AVG', 'FACE-AVG'],
    'EMOTION': ['SHAPES-FACES'],
    # 'SOCIAL': ['TOM-RANDOM'],
    'RELATIONAL': ['REL-MATCH'],
    'GAMBLING': ['PUNISH-REWARD'],
    'MOTOR': ['RH-AVG', 'RF-AVG', 'T-AVG', 'LH-AVG', 'LF-AVG']
}
tasks = list(interest.keys())
hemi = 'left'

#%%
# Load subject gifti files
features = list()
for subject in tqdm(subjects):
    z_maps = list()
    for task in tasks:
        current_path = path / subject / task / "level2" / "z_maps"
        for contrast in interest[task]:
            z_map = nib.load(current_path / f"z_{contrast}_{hemi[0]}h.gii").agg_data()
            z_maps.append(z_map)
    features.append(np.array(z_maps))
features = np.array(features)
features_norm_factor = np.max(np.abs(features))
features_normalized = features / features_norm_factor

#%%
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

# For each task/contrast, compute the distance between the barycenter and each subject
# Compute euclidean distances
barycenter_euc = barycenter_euc.reshape(1, barycenter_euc.shape[0], barycenter_euc.shape[1])
distances_euc = np.linalg.norm(features_normalized - barycenter_euc, axis=2)
distances_euc = np.mean(distances_euc, axis=0)

# Compute FUGW distances
barycenter_fugw = barycenter_fugw.reshape(1, barycenter_fugw.shape[0], barycenter_fugw.shape[1])
distances_fugw = np.linalg.norm(features_normalized - barycenter_fugw, axis=2)
distances_fugw = np.mean(distances_fugw, axis=0)

#%%
# Plot the results with grouped bar plot per task/contrast
# Create a dataframe
idx = 0
data = list()
for task in tasks:
    for contrast in interest[task]:
        data.append(['FUGW', f"{task} - {contrast}", distances_fugw[idx]])
        data.append(['Euclidean', f"{task} - {contrast}", distances_euc[idx]])
        idx += 1
df = pd.DataFrame(data, columns=['Barycenter', 'Task', 'val'])
df.pivot(columns='Barycenter', values='val', index='Task').plot(kind='bar')
plt.ylabel('Mean Euclidean distance to barycenter')

# %%
