# Useful imports
import nibabel as nib
from nilearn import datasets, surface
import numpy as np
from pathlib import Path
import torch

from fugw.mappings import FUGWSparseBarycenter
from fugw.scripts import coarse_to_fine
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
        n_landmarks=1000,
        k=12,
        n_jobs=5,
        verbose=True,
    )

    source_embeddings_normalized, d_max = coarse_to_fine.random_normalizing(
        fs_hemi_embedding
    )

    return source_embeddings_normalized, d_max


# List of subjects
subjects = [x.name for x in contrasts_path.iterdir() if x.is_dir()]
# Keep the first 600 subjects
subjects = subjects[:3]

# Keep the same geometry for all subjects
geometry_embedding, d_max = load_geometry_fsaverage(mesh="fsaverage5")
print(f"Loaded geometry embedding of shape {geometry_embedding.shape}")


# Compute the mesh_sample
coordinates, triangles = surface.load_surf_mesh(
    datasets.fetch_surf_fsaverage()["pial_left"]
)
mesh_sample = coarse_to_fine.sample_mesh_uniformly(
    coordinates,
    triangles,
    embeddings=geometry_embedding,
    n_samples=1000,
)


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
fugw_barycenter = FUGWSparseBarycenter(
    alpha_coarse=0.5,
    alpha_fine=0.5,
    rho_coarse=1.0,
    rho_fine=1.0,
    eps_coarse=1.0,
    eps_fine=1.0,
    selection_radius=0.5 / d_max,
)

# Fit the barycenter
(barycenter_weights, barycenter_features, barycenter_geometry, _, _, _) = (
    fugw_barycenter.fit(
        weights_list,
        features_list,
        [geometry_embedding],
        nits_barycenter=1,
        mesh_sample=mesh_sample,
        coarse_mapping_solver_params={"nits_bcd": 1, "nits_uot": 1000},
        fine_mapping_solver_params={"nits_bcd": 1, "nits_uot": 1000},
        device=device,
        verbose=True,
    )
)


# Save the barycenter
barycenter_path = Path("barycenters")
barycenter_path.mkdir(exist_ok=True)
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
            barycenter_feature, task_path / f"z_{contrast}_{hemi[0]}h.gii"
        )
        idx += 1

# Save barycenter geometry
np.save(barycenter_path_hyp / "weights.npy", barycenter_weights)
np.save(barycenter_path_hyp / "geometry.npy", barycenter_geometry)
