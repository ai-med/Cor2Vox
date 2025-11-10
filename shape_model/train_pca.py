# Script for fitting PCA on cortex geometry (meshes & cortical thickness)

import tqdm
import os
import trimesh
import nibabel as nib
import numpy as np
import glob
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance

latent_dim = 8192
hemi = 'lh'
include_cth = True

path_prefix = '/path/prefix/'
outdir = '/output/dir/'
subjects_dir = '/subjects/dir/'

if not os.path.exists(outdir):
    os.makedirs(outdir)

print("Looking for meshes...")
mesh_files = glob.glob(f'{subjects_dir}/*/{hemi}_midthickness.ply')

mesh_files = sorted(mesh_files)

print(f"Found {len(mesh_files)} meshes.")

print("Using fixed split for training and testing.")
train_ids = [str(l.strip()) for l in open(
                f'{path_prefix}/train_ids.txt', 'r'
            ).readlines()]
train_mesh_files = [f for f in mesh_files if f.split('/')[-2].split('_')[0] in train_ids]
internal_test_ids = [str(l.strip()) for l in open(
                f'{path_prefix}/test_ids.txt', 'r'
            ).readlines()]
internal_test_mesh_files = [f for f in mesh_files if f.split('/')[-2].split('_')[0] in internal_test_ids]
print(f"Number of train files: {len(train_mesh_files)}")
print(f"Number of test files: {len(internal_test_mesh_files)}")

# Only need train mesh files from now on
del mesh_files

meshes = []
train_vertices = []
found_mesh_files = []
for f in tqdm.tqdm(train_mesh_files):
    m = trimesh.load(f, process=False)
    f_cth = f.replace(
        f'{hemi}_midthickness.ply',
        f'surf/{hemi}.thickness'
    )
    cth = nib.freesurfer.io.read_morph_data(
        f_cth
    ).astype(np.float32).reshape((-1, 1))
    train_vertices.append(np.concatenate([m.vertices, cth], axis=1))
    meshes.append(m)

train_vertices = np.stack(train_vertices)

scaler_clean = StandardScaler()
scaler_clean.fit(train_vertices.reshape(len(train_mesh_files), -1))
# Save clean scaler
with open(f'{outdir}/scaler_clean.pkl', 'wb') as f:
    pickle.dump(scaler_clean, f)
train_vertices_scaled = scaler_clean.transform(train_vertices.reshape(len(train_mesh_files), -1))
pca_clean = PCA(n_components=latent_dim)
pca_clean.fit(train_vertices_scaled)
with open(f'{outdir}/pca_clean_{pca_clean.n_components_}.pkl', 'wb') as f:
    pickle.dump(pca_clean, f)
print("Fitted clean PCA")
# Transform and recover the vertices
train_vertices_transformed = pca_clean.transform(train_vertices_scaled)
train_vertices_recovered_scaled = pca_clean.inverse_transform(train_vertices_transformed)
train_vertices_recovered = scaler_clean.inverse_transform(train_vertices_recovered_scaled)
del train_vertices_recovered_scaled
if include_cth:
    train_vertices_recovered = train_vertices_recovered.reshape(len(train_mesh_files), -1, 4)
else:
    train_vertices_recovered = train_vertices_recovered.reshape(len(train_mesh_files), -1, 3)
diff = train_vertices - train_vertices_recovered
diff = np.linalg.norm(diff, axis=2)
mean_diffs = np.mean(diff, axis=1)
print(f"Mean diff train: {np.mean(mean_diffs)}")
