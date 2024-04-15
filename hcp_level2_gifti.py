from nilearn import datasets, image, surface
import nibabel as nib
import glob
from pathlib import Path
from tqdm import tqdm

hcp_path = Path("/data/parietal/store/data/HCP900/glm/")
output_path = Path("/data/parietal/store/data/HCP900/glm/fsaverage5/")

# Assert that the hcp_path exists
assert hcp_path.exists()
if not output_path.exists():
    output_path.mkdir(parents=True)

# Get the list of subjects
subjects = [
    x.name
    for x in hcp_path.iterdir()
    if (x.is_dir() and x.name != "fsaverage5")
]
tasks = [
    "EMOTION",
    "GAMBLING",
    "LANGUAGE",
    "MOTOR",
    "RELATIONAL",
    "SOCIAL",
    "WM",
]


# Fetch the fsaverage5 surface
fsaverage5 = datasets.fetch_surf_fsaverage(mesh="fsaverage5")


# Save skipped subjects
skipped = []

for sub in tqdm(subjects):
    for task in tasks:
        zmaps_path = hcp_path / sub / task / "level2/z_maps"
        if not zmaps_path.exists():
            skipped.append((sub, task))
        else:
            # Load the zmaps
            zmaps = glob.glob(str(zmaps_path / "*.nii.gz"))
            for zmap in zmaps:
                zmap_img = image.load_img(zmap)
                # Project the zmap to fsaverage5
                zmap_lh = surface.vol_to_surf(
                    zmap_img,
                    fsaverage5.pial_left,
                ).astype("float32")
                zmap_rh = surface.vol_to_surf(
                    zmap_img,
                    fsaverage5.pial_right,
                ).astype("float32")
                # Convert to gifti
                zmap_lh = nib.gifti.GiftiImage(
                    darrays=[nib.gifti.GiftiDataArray(data=zmap_lh)]
                )
                zmap_rh = nib.gifti.GiftiImage(
                    darrays=[nib.gifti.GiftiDataArray(data=zmap_rh)]
                )

                # Edit the output path
                output_folder = output_path / sub / task / "level2" / "z_maps"
                output_folder.mkdir(parents=True, exist_ok=True)
                output_path_lh = (
                    output_folder / f"{Path(zmap).stem[:-4]}_lh.gii"
                )
                output_path_rh = (
                    output_folder / f"{Path(zmap).stem[:-4]}_rh.gii"
                )

                # Save the gifti files
                nib.save(zmap_lh, output_path_lh)
                nib.save(zmap_rh, output_path_rh)


print(f"Skipped {len(skipped)} subjects")
print(skipped)
