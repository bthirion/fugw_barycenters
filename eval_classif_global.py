# %%
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed

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
barycenter_euc = list()
for task in tasks:
    barycenter_path_euc = barycenter_path / "euclidean" / task / "level2" / "z_maps"
    barycenter_euc.append(list())
    for contrast in interest[task]:
        barycenter_euc[-1].append(nib.load(barycenter_path_euc / f"z_{contrast}_{hemi[0]}h.gii").agg_data())
    barycenter_euc[-1] = np.array(barycenter_euc[-1])
barycenter_euc = np.concatenate(barycenter_euc)

# Load FUGW barycenter gifti files
# get all paths that begin with alpha_* in the directory
barycenter_paths_fugw = list(barycenter_path.glob("alpha_*"))
barycenters_fugw = dict()
for barycenter_path_fugw in barycenter_paths_fugw:
    barycenter_fugw = list()
    for task in tasks:
        barycenter_path_fugw_task = barycenter_path_fugw / task / "level2" / "z_maps"
        barycenter_fugw.append(list())
        for contrast in interest[task]:
            barycenter_fugw[-1].append(nib.load(barycenter_path_fugw_task / f"z_{contrast}_{hemi[0]}h.gii").agg_data())
        barycenter_fugw[-1] = np.array(barycenter_fugw[-1])
    barycenters_fugw[barycenter_path_fugw.name] = np.concatenate(barycenter_fugw)
# remove barycenters which have nan
barycenters_fugw = {k: v for k, v in barycenters_fugw.items() if not np.isnan(v).any()}

# %%

def covariance_to_bary(barycenter_name, barycenter):
    # barycenter /= np.max(np.abs(barycenter))
    barycenter /= np.linalg.norm(barycenter, axis=1).mean()
    covariances = np.sum(features_normalized * barycenter[None, :, :], axis=1)
    return [[barycenter_name, np.mean(covariances), covariances]]

results = covariance_to_bary("euclidean", barycenter_euc)
df_euc = pd.DataFrame(results, columns=['Barycenter', 'mean_cov', 'cov'])

results = list()
for barycenter_name, barycenter in barycenters_fugw.items():
    results += covariance_to_bary(barycenter_name, barycenter)
df_fugw = pd.DataFrame(results, columns=['Barycenter', 'mean_cov', 'cov'])
df_fugw = df_fugw[df_fugw['mean_cov'] == df_fugw['mean_cov'].max()]
df = pd.concat([df_euc, df_fugw])

# %%
# Plot the covariance of each barycenter on fsaverage5
cov_euc = df_euc['cov'].values[0]
cov_fugw = df_fugw['cov'].values[0]
TH = 0
from nilearn import datasets
from nilearn.plotting import plot_surf
fsaverage5 = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
hemi = 'left'
fig = plt.figure(figsize=(6, 20))
q = 1
for i in range(10):
    data = cov_euc[i]
    ax = fig.add_subplot(N_SUBJECTS_TEST, 2, q, projection='3d')
    plot_surf(
        fsaverage5[f'pial_{hemi}'],
        data,
        fsaverage5[f'sulc_{hemi}'],
        hemi=hemi,
        view='lateral',
        engine='matplotlib',
        threshold=TH,
        axes=ax
    )
    q += 1

    data = cov_fugw[i]
    ax = fig.add_subplot(N_SUBJECTS_TEST, 2, q, projection='3d')
    plot_surf(
        fsaverage5[f'pial_{hemi}'],
        data,
        fsaverage5[f'sulc_{hemi}'],
        hemi=hemi,
        view='lateral',
        engine='matplotlib',
        threshold=TH,
        axes=ax
    )
    q += 1
# %%
