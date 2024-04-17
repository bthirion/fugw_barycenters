import itertools
from nilearn import datasets, surface
import nibabel as nib
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm
import torch

from fugw.scripts.lmds import compute_lmds_mesh
from fugw.mappings.barycenter import FUGWBarycenter


path = Path("/data/parietal/store/data/HCP900/glm/fsaverage5/")
assert path.exists()

N_SUBJECTS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALPHA_LIST = [0.5]
RHO = [1]
EPS_LIST = [1e-4]

# Get the list of subjects
subjects = [
    x.name
    for x in path.iterdir()
]
subjects = subjects[:N_SUBJECTS]
interest = {
    'LANGUAGE': ['MATH-STORY'],
    'WM': ['2BK-0BK', 'BODY-AVG', 'PLACE-AVG', 'TOOL-AVG', 'FACE-AVG'],
    'EMOTION': ['SHAPES-FACES'],
    'SOCIAL': ['TOM-RANDOM'],
    'RELATIONAL': ['REL-MATCH'],
    'GAMBLING': ['PUNISH-REWARD'],
    'MOTOR': ['RH-AVG', 'RF-AVG', 'T-AVG', 'LH-AVG', 'LF-AVG']
}
tasks = list(interest.keys())
hemi = 'left'

# Load gifti files
features = list()
for subject in tqdm(subjects):
    z_maps = list()
    for task in tasks:
        current_path = path / subject / task / "level2" / "z_maps"
        for contrast in interest[task]:
            z_map = nib.load(current_path / f"z_{contrast}_{hemi[0]}h.gii").agg_data()
            z_maps.append(z_map)
    features.append(np.array(z_maps))
features = torch.tensor(np.array(features))

# Compute the geodesic distance matrix
fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
(coordinates, triangles) = surface.load_surf_mesh(fsaverage.pial_left)
geometry = compute_lmds_mesh(coordinates, triangles, k=20)
geometry = torch.cdist(geometry, geometry)

# Normalize features and geometries
features = features.to(DEVICE)
geometry = geometry.to(DEVICE)
features_norm_factor = torch.max(torch.abs(features))
features_normalized = features / features_norm_factor
geometry_normalized = geometry / torch.max(geometry)

# Compute fugw barycenter
d = features_normalized.shape[-1]
n_subjects = features_normalized.shape[0]
weights = torch.ones(d, device=DEVICE) / d
weights_list = [weights for _ in range(n_subjects)]
features_list = [features_normalized[i] for i in range(n_subjects)]
barycenter_path = Path("barycenters")
barycenter_path.mkdir(exist_ok=True)

print("Total number of combinations: ", len(ALPHA_LIST) * len(RHO) * len(EPS_LIST))

# Compute and save euclidean barycenter gifti file
barycenter_path_hyp = barycenter_path / "euclidean"
euclidean_barycenter_features = torch.mean(features_normalized, dim=0)
idx = 0
for task in tasks:
    task_path = barycenter_path_hyp / task / "level2" / "z_maps"
    task_path.mkdir(exist_ok=True, parents=True)
    for contrast in interest[task]:
        barycenter_feature = euclidean_barycenter_features[idx].cpu().numpy()
        barycenter_feature = nib.gifti.GiftiImage(
            darrays=[nib.gifti.GiftiDataArray(barycenter_feature)]
        )
        nib.save(
            barycenter_feature,
            task_path / f"z_{contrast}_{hemi[0]}h.gii"
        )
        idx += 1

for alpha, rho, eps in itertools.product(ALPHA_LIST, RHO, EPS_LIST):
    print()
    start_time = time.time()

    fugw_bary = FUGWBarycenter(
        alpha=alpha,
        rho=rho,
        eps=eps
    )

    (
        barycenter_weights,
        barycenter_features,
        barycenter_geometry,
        _,
        _,
        _
    ) = fugw_bary.fit(
        weights_list=weights_list,
        features_list=features_list,
        geometry_list=[geometry_normalized],
        init_barycenter_features=euclidean_barycenter_features,
        nits_barycenter=3,
        solver='mm',
        solver_params={
            'nits_bcd': 3,
            'nits_uot': 50,
            'eval_bcd': 1,
            'eval_uot': 1,
        },
        device=DEVICE,
        verbose=False
    )

    barycenter_weights = barycenter_weights.cpu().numpy()
    barycenter_features = barycenter_features.cpu().numpy()
    barycenter_geometry = barycenter_geometry.cpu().numpy()

    # Save barycenter gifti files
    barycenter_path_hyp = barycenter_path / f"alpha_{alpha}_rho_{rho}_eps_{eps}"
    idx = 0
    for task in tasks:
        task_path = barycenter_path_hyp / task / "level2" / "z_maps"
        task_path.mkdir(exist_ok=True, parents=True)
        for contrast in interest[task]:
            barycenter_feature = barycenter_features[idx]
            barycenter_feature = nib.gifti.GiftiImage(
                darrays=[nib.gifti.GiftiDataArray(barycenter_feature)]
            )
            nib.save(
                barycenter_feature,
                task_path / f"z_{contrast}_{hemi[0]}h.gii"
            )
            idx += 1

    # Save barycenter geometry
    np.save(barycenter_path_hyp / "weights.npy", barycenter_weights)
    np.save(barycenter_path_hyp / "geometry.npy", barycenter_geometry)

    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")
