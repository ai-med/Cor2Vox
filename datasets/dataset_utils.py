from glob import glob
import re
import os
import numpy as np
import nibabel as nib
import torchio as tio
import torch 

def pair_file(input_folders, target_folder, condition_folders=None):
    """
    Pairs files from two folders based on matching numeric sequences in their filenames.
    
    Args:
        input_folder (str): Path to the folder containing input files.
        target_folder (str): Path to the folder containing target files.
    
    Returns:
        list of tuples: A list of tuples where each tuple consists of a paired 
        input and target file. The pairing is done by matching the numeric portion 
        of the filenames.
    """
    if isinstance(input_folders, str):
        input_folders = [input_folders]

    print('input folders: ', input_folders)
    input_files_list = [sorted(glob(os.path.join(folder, '*'))) for folder in input_folders]
    target_files = sorted(glob(os.path.join(str(target_folder), '*')))

    for folder_files in input_files_list:
        if len(folder_files) != len(target_files):
            raise ValueError("Input folders and target folder must have the same number of files.")

    if condition_folders is not None:
        if isinstance(condition_folders, str):
            condition_folders = [condition_folders]
        condition_files_list = [sorted(glob(os.path.join(folder, '*'))) for folder in condition_folders]
        for folder_files in condition_files_list:
            if len(folder_files) != len(target_files):
                raise ValueError("Condition folders and target folder must have the same number of files.")

    pairs = []

    if condition_folders is not None:

        for files_per_scan in zip(*input_files_list, target_files, *condition_files_list):
            input_count = len(input_files_list)       # Number of input files per scan
            condition_count = len(condition_files_list)  # Number of condition files per scan

            input_files = files_per_scan[:input_count]  # First N are input files
            target_file = files_per_scan[input_count]   # Next one is the target file
            condition_files = files_per_scan[input_count + 1:]  # Remaining are condition files

            assert len(condition_files) == condition_count, "Condition files count mismatch"
            
            # Extract numeric IDs from filenames and ensure they match
            numeric_ids = [int("".join(re.findall("\d", os.path.basename(f)))) for f in list(input_files) + [target_file] + list(condition_files)]
            if not all(id_ == numeric_ids[0] for id_ in numeric_ids):
                raise ValueError(f"File IDs do not match: {files_per_scan}")
            
            scan_id = str(numeric_ids[0])

            pairs.append((input_files, target_file, condition_files, scan_id))
    
    else:

        for files_per_scan in zip(*input_files_list, target_files):
            *input_files, target_file = files_per_scan
            
            # Extract numeric IDs from filenames and ensure they match
            numeric_ids = [int("".join(re.findall("\d", os.path.basename(f)))) for f in input_files + [target_file]]
            if not all(id_ == numeric_ids[0] for id_ in numeric_ids):
                raise ValueError(f"File IDs do not match: {files_per_scan}")
            
            scan_id = str(numeric_ids[0])

            pairs.append((input_files, target_file, scan_id))

    return pairs


def read_image(file_path, scaler=tio.RescaleIntensity(out_min_max=(0, 1)), pass_scaler=False):
    """
    Reads an image file and applies scaling if required.
    
    Args:
        file_path (str): Path to the image file (e.g., NIfTI format).
        scaler (object): A scaler object with a `fit_transform` method to normalize 
        the image data.
        pass_scaler (bool): If True, bypasses scaling and returns the image as is.
    
    Returns:
        numpy.ndarray: The image data, scaled if `pass_scaler` is False.
    """
    img = nib.load(file_path).get_fdata()
    if not pass_scaler:
        img_tensor = torch.tensor(img).unsqueeze(0)
        img = scaler(img_tensor).squeeze().numpy()
    return img

def resize_img(img, h_size, w_size, d_size):
    """
    Resizes a 3D image to the specified dimensions.
    
    Args:
        img (numpy.ndarray): The image to resize.
        h_size (int): Height of the desired output.
        w_size (int): Width of the desired output.
        d_size (int): Depth of the desired output.
    
    Returns:
        numpy.ndarray: The resized image.
    """
    h, w, d = img.shape
    if h != h_size or w != w_size or d != d_size:
        img = tio.ScalarImage(tensor=img[np.newaxis, ...])
        resize = tio.Resize((h_size, w_size, d_size))
        img = np.asarray(resize(img))[0]
    return img
