# %%
# Useful imports
import nibabel as nib
from nilearn import datasets, image, plotting, surface
import numpy as np
from pathlib import Path
import torch

from fugw.mappings import FUGWBarycenter
from fugw.scripts import lmds

# Set device
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:3")

contrasts_path = Path("/data/parietal/store/data/HCP900/glm/fsaverage5/")
assert contrasts_path.exists()


# Compute the geometry embeddings
def load_geometry_fsaverage(mesh="fsaverage5", hemi="left"):
    fsaverage5 = datasets.fetch_surf_fsaverage(mesh=mesh)

    if hemi == "left":
        (coordinates, triangles) = surface.load_surf_mesh(fsaverage5.pial_left)
    elif hemi == "right":
        (coordinates, triangles) = surface.load_surf_mesh(
            fsaverage5.pial_right
        )

    fs_hemi_embedding = lmds.compute_lmds_mesh(
        coordinates,
        triangles,
        n_landmarks=100,
        k=12,
        n_jobs=2,
        verbose=True,
    )

    source_embeddings_normalized = fs_hemi_embedding / torch.norm(
        fs_hemi_embedding, p=2, dim=1, keepdim=True
    )

    return source_embeddings_normalized


# List of subjects
subjects = [x.name for x in contrasts_path.iterdir() if x.is_dir()]
# Keep only 10 subjects
subjects = subjects[:3]

# Keep the same geometry for all subjects
geometry_embedding = load_geometry_fsaverage(mesh="fsaverage5")
print(f"Loaded geometry embedding of shape {geometry_embedding.shape}")


TRAINING_TASKS = ["WM"]
TESTING_TASKS = ["LANGUAGE"]

features_list = []


# Load the contrasts
def load_contrasts(pathlist, device):
    contrasts_list = []
    for contrast in pathlist:
        zmap_img = nib.load(contrast)
        zmap_data = zmap_img.darrays[0].data
        zmap_data = np.nan_to_num(zmap_data)
        contrasts_list.append(torch.tensor(zmap_data, dtype=torch.float32))

    features = torch.stack(contrasts_list, dim=0).to(device)
    return features


for sub in subjects:
    contrasts_subject = []
    for task in TRAINING_TASKS:
        contrasts_subject += list(
            contrasts_path.glob(f"{sub}/{task}/level2/z_maps/*_lh.gii")
        )
    features = load_contrasts(contrasts_subject, device)
    features /= torch.norm(features, p=2, dim=1, keepdim=True)
    features_list.append(features)

print(f"Loaded {len(features_list)} subjects")

# Choose uniform weights
n_voxels = geometry_embedding.shape[0]
weights_list = [
    torch.ones(n_voxels, device=device) / n_voxels for _ in features_list
]

# Initialize the barycenter
fugw_barycenter = FUGWBarycenter()

# Fit the barycenter
fugw_barycenter.fit(
    weights_list,
    features_list,
    [
        geometry_embedding @ geometry_embedding.T
    ],  # TODO : add low-rank approximation
    nits_barycenter=2,
    solver_params={"nits_bcd": 2, "nits_uot": 10},
    device=device,
    verbose=False,
)
