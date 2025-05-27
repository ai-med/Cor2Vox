import numpy as np
import os
import nibabel as nib

def cortical_ribbon_mask_generation(sdf1, sdf2):
    # Create a mask where signs are the same
    same_sign_mask = np.sign(sdf1) == np.sign(sdf2)
    sdf_combine = np.zeros_like(sdf1)
    sdf_combine[same_sign_mask] = 0
    differing_sign_mask = ~same_sign_mask
    sdf_combine[differing_sign_mask] = 1  # Or handle differently if needed
    # where sdf1 is 0, sdf_combine is 1
    sdf_combine[sdf1 == 0] = 1
    # where sdf2 is 0, sdf_combine is 1
    sdf_combine[sdf2 == 0] = 1
    return sdf_combine


# Input folders
sdf_pial_folder = '/data/to/sdf_pial/'
sdf_white_folder = '/data/to/sdf_white/'

# Output folder
output_folder = '/data/to/cortex_ribbon_mask/'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get the list of .nii.gz files from the sdf_pial folder
file_list = [f for f in os.listdir(sdf_pial_folder) if f.endswith('.nii.gz')]

flag = 0

# Process each file
for filename in file_list:
    print(f'Processing {filename}...')

    sdf_pial_path = os.path.join(sdf_pial_folder, filename)
    sdf_white_path = os.path.join(sdf_white_folder, filename)

    if not os.path.exists(sdf_white_path):
        print(f'Corresponding sdf_white file for {filename} not found. Skipping.')
        flag += 1
        continue

    # Load the SDF
    sdf_pial_img = nib.load(sdf_pial_path)
    sdf_white_img = nib.load(sdf_white_path)

    # Extract the data arrays
    sdf_pial_data = sdf_pial_img.get_fdata()
    sdf_white_data = sdf_white_img.get_fdata()

    # create the cortical ribbon mask
    cortical_ribbon_mask = cortical_ribbon_mask_generation(sdf_pial_data, sdf_white_data)

    cortical_ribbon_mask_img = nib.Nifti1Image(cortical_ribbon_mask, affine=sdf_pial_img.affine, header=sdf_pial_img.header)

    output_path = os.path.join(output_folder, filename)
    nib.save(cortical_ribbon_mask_img, output_path)

print('Processing completed.')
print(f'{flag} files were skipped due to missing corresponding sdf files.')