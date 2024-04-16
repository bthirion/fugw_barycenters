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

N_SUBJECTS = 50
DEVICE = "cuda"
# ALPHA_LIST = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
ALPHA_LIST = [0.5]
RHO = [1]
# EPS_LIST = [1e-4, 1e-3, 1e-2, 1e-1]
EPS_LIST = [1e-3, 1e-2, 1e-1]

# Get the list of subjects
subjects = [
    x.name
    for x in path.iterdir()
]
subjects = subjects[:N_SUBJECTS]
tasks = [
    "EMOTION",
    "GAMBLING",
    "LANGUAGE",
    "MOTOR",
    "RELATIONAL",
    "SOCIAL",
    "WM",
]

# Load gifti files
features = list()
for subject in tqdm(subjects):
    z_maps = list()
    for task in tasks:
        current_path = path / subject / task / "level2" / "z_maps"
        filnames_z_maps = list(current_path.glob("*.gii"))
        # keep only left hemisphere
        filnames_z_maps = [x for x in filnames_z_maps if "lh" in x.name]
        for filename in filnames_z_maps:
            tmp = nib.load(filename)
            z_maps.append(tmp.darrays[0].data)

    features.append(np.array(z_maps))

features = torch.tensor(np.array(features))

# Compute the geodesic distance matrix


def compute_geometry_from_mesh(mesh_path):
    """Util function to compute matrix of geodesic distances of a mesh."""
    (coordinates, triangles) = surface.load_surf_mesh(mesh_path)
    geometry = compute_lmds_mesh(coordinates, triangles, k=20)
    geometry = torch.cdist(geometry, geometry)
    return geometry


fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
geometry = compute_geometry_from_mesh(fsaverage.pial_left)

# Normalize features and geometries
features_normalized = features / torch.linalg.norm(
    features, axis=1, keepdims=True
)
geometry_normalized = geometry / torch.max(geometry)
features_normalized = features_normalized.to(DEVICE)
geometry_normalized = geometry_normalized.to(DEVICE)

# Compute fugw barycenter
d = features_normalized.shape[-1]
n_subjects = features_normalized.shape[0]
weights = torch.ones(d, device=DEVICE) / d
weights_list = [weights for _ in range(n_subjects)]
features_list = [features_normalized[i] for i in range(n_subjects)]
barycenter_path = Path("barycenters")
barycenter_path.mkdir(exist_ok=True)

print("Total number of combinations: ", len(ALPHA_LIST) * len(RHO) * len(EPS_LIST))

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
        nits_barycenter=3,
        solver='mm',
        solver_params={
            'nits_bcd': 5,
            'nits_uot': 50,
            'eval_bcd': 1,
            'eval_uot': 1,
        },
        device=DEVICE,
        verbose=False
    )

    # Save the barycenter
    barycenter_path_hyp = barycenter_path / f"alpha_{alpha}__rho_{rho}__eps_{eps}"
    barycenter_path_hyp.mkdir(exist_ok=True)
    barycenter_weights = barycenter_weights.cpu().numpy()
    barycenter_features = barycenter_features.cpu().numpy()
    barycenter_geometry = barycenter_geometry.cpu().numpy()
    np.save(barycenter_path_hyp / "weights.npy", barycenter_weights)
    np.save(barycenter_path_hyp / "features.npy", barycenter_features)
    np.save(barycenter_path_hyp / "geometry.npy", barycenter_geometry)

    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")
